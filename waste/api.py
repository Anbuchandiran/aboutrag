# module2_rag/src/api.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
import json
from pathlib import Path
from datetime import datetime
import tempfile
import os

import pandas as pd
import cv2
import easyocr
from rapidfuzz import process, fuzz

# ✅ NEW (only additions needed for drugnames Chroma)
import chromadb
from chromadb.utils import embedding_functions

from waste.validate_prescription import run_validation_multi
from module2_rag.src.mongo_memory import (
    mongo_first_search,
    store_solved_case,
    add_visit,
    upsert_patient,
    upsert_doctor,
    get_patient_history,
    get_doctor_history,
    patients,
    doctors,
)

app = FastAPI(title="Medical RAG Backend (Mongo + Chroma + LLM)")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- PATHS --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE2_RAG_DIR = PROJECT_ROOT / "module2_rag"
DATA_DIR = MODULE2_RAG_DIR / "data"

LAST_VALIDATION_PATH = DATA_DIR / "last_validation.txt"
LAST_CONTEXT_PATH = DATA_DIR / "last_context.json"

# -------------------- OCR / DRUGBANK CACHES --------------------
_OCR_READER = None
_DRUG_NAMES = None  # list[str]

# ✅ NEW: Drug-names Chroma (only additions; does not affect other logic)
DRUGNAMES_DIR = str(MODULE2_RAG_DIR / "db" / "chroma_drugnames")
DRUGNAMES_COLLECTION = "drug_names"
_drugnames_client = None
_drugnames_col = None
_drugnames_embed_fn = None
OCR_DISTANCE_THRESHOLD = 0.55
WHISPER_MODEL_NAME = "base"
WHISPER_MODEL_DIR = PROJECT_ROOT / ".models" / "faster-whisper"


def _to_na(x):
    """Convert empty/None values to N/A and format lists nicely."""
    if x is None:
        return "N/A"
    if isinstance(x, list):
        return ", ".join([str(i) for i in x]) if x else "N/A"
    s = str(x).strip()
    return s if s else "N/A"


def write_last_validation(text: str) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LAST_VALIDATION_PATH.write_text(text or "", encoding="utf-8")
    except Exception as e:
        print("Failed to write last_validation.txt:", repr(e), "path:", str(LAST_VALIDATION_PATH))


def write_last_context(context: Dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LAST_CONTEXT_PATH.write_text(json.dumps(context, indent=2), encoding="utf-8")
    except Exception as e:
        print("Failed to write last_context.json:", repr(e), "path:", str(LAST_CONTEXT_PATH))


# -------------------- MODELS --------------------
class PatientIn(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    phone: Optional[str] = ""
    chronic_conditions: Optional[List[str]] = []
    allergies: Optional[List[str]] = []
    notes: Optional[str] = ""


class DoctorIn(BaseModel):
    doctor_id: str
    name: str
    department: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""


class VisitIn(BaseModel):
    patient_id: str
    doctor_id: str
    complaint: str
    diagnosis: Optional[str] = ""
    prescription_text: Optional[str] = ""
    extracted_entities: Optional[Dict[str, Any]] = {}
    notes: Optional[str] = ""


class Ask(BaseModel):
    patient_id: str
    doctor_id: str
    query: str


# -------------------- HELPERS --------------------
def normalize_drug_query(q: str) -> str:
    q = (q or "").lower()
    q = re.sub(r"[\+\.,;/]", " ", q)
    q = re.sub(r"\band\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    parts = [p.strip() for p in q.split(" ") if p.strip()]
    parts = [p for p in parts if len(p) > 2]
    parts = sorted(set(parts))
    return "|".join(parts)


def clean_ocr_text(txt: str) -> str:
    txt = (txt or "").lower()
    txt = re.sub(r"[^a-z0-9+\-\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _normalize_common_typos(token: str) -> str:
    """
    Fix recurring OCR/STT medicine misspellings before matching.
    """
    t = (token or "").strip().lower()
    typo_map = {
        "asprin": "aspirin",
        "aspiin": "aspirin",
        "aspiri": "aspirin",
        "aspinn": "aspirin",
        "warfarin": "warfarin",
        "warforin": "warfarin",
        "warfarln": "warfarin",
        "warferin": "warfarin",
        "dijoxin": "digoxin",
        "digoxln": "digoxin",
        "digoxim": "digoxin",
        "digoxxn": "digoxin",
        "plicamyein": "plicamycin",
        "plicamyin": "plicamycin",
        "plicamyein": "plicamycin",
        "plicamycin": "plicamycin",
    }
    if t in typo_map:
        return typo_map[t]
    return t


def _prepare_ocr_images(img):
    """
    Generate multiple enhanced variants for handwritten text OCR.
    """
    variants = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    variants.append(thr)

    return variants


def _merge_fragmented_tokens(tokens: List[str]) -> List[str]:
    """
    Merge OCR fragments like ['as', 'p', 'n'] -> 'aspn'
    and ['waxfari', 'n'] -> 'waxfarin'.
    """
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # Join trailing single-char suffix to previous long token.
        if merged and len(t) == 1 and t.isalpha() and len(merged[-1]) >= 4:
            merged[-1] = merged[-1] + t
            i += 1
            continue

        # Join runs of short alpha chunks.
        if len(t) <= 2 and t.isalpha():
            j = i
            buf = []
            while j < len(tokens) and len(tokens[j]) <= 2 and tokens[j].isalpha():
                buf.append(tokens[j])
                j += 1
            joined = "".join(buf)
            if len(joined) >= 3:
                merged.append(joined)
                i = j
                continue

        merged.append(t)
        i += 1

    # preserve order, unique
    out: List[str] = []
    seen = set()
    for x in merged:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_local_whisper_model_path() -> str:
    """
    Download/sync faster-whisper model into project-local storage and return the snapshot path.
    """
    from huggingface_hub import snapshot_download

    repo_id = f"Systran/faster-whisper-{WHISPER_MODEL_NAME}"
    WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(WHISPER_MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    return local_path


def _map_tokens_to_known_drugs(tokens: List[str], drug_list: List[str], max_items: int = 5, min_score: int = 80) -> List[str]:
    mapped: List[str] = []
    for t in tokens:
        if t in drug_list and t not in mapped:
            mapped.append(t)
        else:
            fb, sc = fuzzy_match_drug(t, drug_list)
            if fb and sc >= min_score and fb not in mapped:
                mapped.append(fb)
        if len(mapped) >= max_items:
            break
    return mapped


def load_drugbank_names() -> List[str]:
    """
    Loads drug names from module2_rag/data/drugbank_docs.csv (columns: drug1, drug2).
    Cached after first load.
    """
    global _DRUG_NAMES
    if _DRUG_NAMES is not None:
        return _DRUG_NAMES

    csv_path = DATA_DIR / "drugbank_docs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"DrugBank CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "drug1" not in df.columns or "drug2" not in df.columns:
        raise ValueError("drugbank_docs.csv must contain columns: drug1, drug2")

    names = set()
    for col in ("drug1", "drug2"):
        for x in df[col].dropna().astype(str).tolist():
            x = x.strip().lower()
            if x:
                names.add(x)

    _DRUG_NAMES = sorted(names)
    return _DRUG_NAMES


def fuzzy_match_drug(word: str, drug_list: List[str]) -> tuple[str, int]:
    """
    Returns: (best_match, score 0..100)
    """
    if not word:
        return "", 0
    best = process.extractOne(word, drug_list, scorer=fuzz.WRatio)
    if not best:
        return "", 0
    match, score, _ = best
    return match, int(score)


def llm_fix_drug_text(raw_text: str) -> str:
    """
    LLM fallback (Gemini) to clean up messy OCR text.
    If key isn't configured, returns raw_text.
    """
    try:
        from google import genai
        GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not GEMINI_KEY:
            return raw_text

        client = genai.Client(api_key=GEMINI_KEY)

        prompt = f"""
You are helping correct medicine names extracted from an OCR prescription image.

OCR TEXT:
{raw_text}

Task:
- Correct spelling of medicine names
- Remove irrelevant words
- Output ONLY medicine names separated by ' + '
- Do NOT add extra explanation

Example output:
warfarin + aspirin
""".strip()

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return (resp.text or "").strip().lower()
    except Exception:
        return raw_text


# ✅ NEW: Chroma drug-names helpers (ONLY additions; does not change your other actions)
def get_drugnames_collection():
    global _drugnames_client, _drugnames_col, _drugnames_embed_fn

    if _drugnames_col is not None:
        return _drugnames_col

    # Create client for the persistent Chroma folder you built
    _drugnames_client = chromadb.PersistentClient(path=DRUGNAMES_DIR)

    # Use same embedding model you used while building
    _drugnames_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    _drugnames_col = _drugnames_client.get_or_create_collection(
        name=DRUGNAMES_COLLECTION,
        embedding_function=_drugnames_embed_fn,
    )
    return _drugnames_col


def correct_token_with_chroma(token: str) -> tuple[str, float]:
    """
    Returns (best_drug_name, distance).
    Lower distance = better match.
    """
    token = (token or "").strip().lower()
    if not token:
        return "", 999.0

    col = get_drugnames_collection()

    try:
        res = col.query(query_texts=[token], n_results=1)
    except Exception as e:
        # If Chroma fails for any reason, return original token
        print("Chroma query failed:", repr(e))
        return token, 999.0

    best = ""
    dist = 999.0

    if res and res.get("documents") and res["documents"][0]:
        best = (res["documents"][0][0] or "").strip().lower()

    if res and res.get("distances") and res["distances"][0]:
        try:
            dist = float(res["distances"][0][0])
        except Exception:
            dist = 999.0

    return best, dist


def extract_medicine_names(raw_text: str, max_items: int = 5) -> Dict[str, Any]:
    """
    Normalize noisy OCR/STT text into likely medicine names.
    """
    raw_clean = re.sub(r"[^a-zA-Z0-9\s\+\-/,;]", " ", (raw_text or "").lower())
    raw_clean = re.sub(r"\s+", " ", raw_clean).strip()
    raw_split = [t for t in re.split(r"[+\s,;/]+", raw_clean) if t]
    raw_tokens = [_normalize_common_typos(t) for t in _merge_fragmented_tokens(raw_split) if len(t) > 2]

    llm_text = llm_fix_drug_text(raw_clean)
    llm_clean = re.sub(r"[^a-zA-Z0-9\s\+\-/,;]", " ", (llm_text or "").lower())
    llm_clean = re.sub(r"\s+", " ", llm_clean).strip()
    llm_split = [t for t in re.split(r"[+\s,;/]+", llm_clean) if t]
    llm_tokens = [_normalize_common_typos(t) for t in _merge_fragmented_tokens(llm_split) if len(t) > 2]

    stopwords = {
        "take", "tablet", "tablets", "tab", "capsule", "capsules", "cap",
        "morning", "night", "after", "before", "food", "daily", "twice",
        "once", "days", "day", "for", "and", "the",
    }

    candidates: List[str] = []
    for t in llm_tokens + raw_tokens:
        if t not in stopwords and t not in candidates:
            candidates.append(t)

    out: List[str] = []
    debug: List[Dict[str, Any]] = []
    try:
        drug_list = load_drugbank_names()
    except Exception:
        drug_list = []
    llm_out: List[str] = _map_tokens_to_known_drugs(llm_tokens, drug_list, max_items=max_items, min_score=80) if drug_list else []

    for token in candidates:
        best, dist = correct_token_with_chroma(token)
        used = None

        if best and dist < 0.70:
            used = best
            debug.append({"token": token, "via": "chroma", "best": best, "dist": dist})
        elif drug_list:
            f_best, score = fuzzy_match_drug(token, drug_list)
            # LLM-derived tokens can be accepted with slightly lower fuzzy score.
            min_score = 80 if token in llm_tokens else 88
            if f_best and score >= min_score:
                used = f_best
                debug.append({"token": token, "via": "fuzzy", "best": f_best, "score": score})
            else:
                debug.append({"token": token, "via": "none", "best": best, "dist": dist, "score": score if drug_list else None})
        else:
            debug.append({"token": token, "via": "none", "best": best, "dist": dist})

        if used and used not in out:
            out.append(used)
        if len(out) >= max_items:
            break

    # last fallback so UI always gets usable text
    if not out and raw_tokens:
        # Prefer cleaned LLM tokens if they map to known medicines.
        if llm_out:
            out = llm_out[:max_items]
        else:
            out = raw_tokens[:max_items]

    return {
        "text": " + ".join(out[:max_items]),
        "llm_mapped_text": " + ".join(llm_out[:max_items]),
        "raw_tokens": raw_tokens,
        "llm_text": llm_text,
        "matches": debug,
    }


# -------------------- API --------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running. Go to /docs"}


@app.get("/health")
async def health():
    mongo_ok = True
    try:
        await patients.find_one({}, {"_id": 1})
    except Exception:
        mongo_ok = False

    return {
        "api": "ok",
        "mongo_ok": mongo_ok,
        "last_validation_path": str(LAST_VALIDATION_PATH),
        "last_context_path": str(LAST_CONTEXT_PATH),
    }


@app.post("/patients/upsert")
async def patients_upsert(p: PatientIn):
    await upsert_patient(
        patient_id=p.patient_id,
        name=p.name,
        age=p.age,
        gender=p.gender,
        phone=p.phone,
        chronic_conditions=p.chronic_conditions,
        allergies=p.allergies,
        notes=p.notes,
    )
    return {"ok": True, "patient_id": p.patient_id}


@app.get("/patients/{patient_id}/history")
async def patient_history(patient_id: str, limit: int = 20):
    history = await get_patient_history(patient_id, limit=limit)
    return {"patient_id": patient_id, "history": history}


@app.post("/doctors/upsert")
async def doctors_upsert(d: DoctorIn):
    await upsert_doctor(
        doctor_id=d.doctor_id,
        name=d.name,
        department=d.department,
        phone=d.phone,
        email=d.email,
    )
    return {"ok": True, "doctor_id": d.doctor_id}


@app.get("/doctors/{doctor_id}/history")
async def doctor_history(doctor_id: str, limit: int = 20):
    history = await get_doctor_history(doctor_id, limit=limit)
    return {"doctor_id": doctor_id, "history": history}


@app.post("/visits/add")
async def visits_add(v: VisitIn):
    await add_visit(
        patient_id=v.patient_id,
        doctor_id=v.doctor_id,
        complaint=v.complaint,
        diagnosis=v.diagnosis,
        prescription_text=v.prescription_text,
        extracted_entities=v.extracted_entities,
        notes=v.notes,
    )
    return {"ok": True}


@app.post("/ask")
async def ask(data: Ask):
    # patient history
    try:
        history = await get_patient_history(data.patient_id, limit=10)
    except Exception:
        history = []

    # normalized key
    norm_key = normalize_drug_query(data.query)

    # mongo exact match
    try:
        matches = await mongo_first_search(data.patient_id, norm_key)
    except Exception as e:
        print("Mongo search failed:", repr(e))
        matches = []

    previous_answer = None
    if matches:
        previous_answer = {
            "answer": matches[0].get("final_answer", ""),
            "matched_query": matches[0].get("query_text", ""),
            "score": matches[0].get("score"),
            "created_at": matches[0].get("created_at"),
            "normalized_key": matches[0].get("normalized_key"),
        }

    # fetch patient + doctor docs
    p = await patients.find_one({"_id": data.patient_id}) or {}
    d = await doctors.find_one({"_id": data.doctor_id}) or {}

    # history text for validation
    history_text_lines = []
    for v in history[:10]:
        history_text_lines.append(
            f"- Complaint: {v.get('complaint','')}; Diagnosis: {v.get('diagnosis','')}; "
            f"Prescription: {v.get('prescription_text','')}"
        )
    history_text = "\n".join(history_text_lines)

    patient_profile = {
        "age": p.get("age", None),
        "allergies": p.get("allergies", []) or [],
        "conditions": p.get("chronic_conditions", []) or [],
        "current_meds": p.get("current_meds", []) or [],
        "history_text": history_text,
    }

    # run validation
    answer = run_validation_multi(
        data.query,
        patient_profile,
        patient_id=data.patient_id,
        doctor_id=data.doctor_id
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # context JSON (PDF reads this)
    context = {
        "patient_id": data.patient_id,
        "doctor_id": data.doctor_id,
        "patient": {
            "name": p.get("name", "N/A"),
            "age": p.get("age", "N/A"),
            "gender": p.get("gender", "N/A"),
            "phone": p.get("phone", "N/A"),
            "chronic_conditions": p.get("chronic_conditions", []) or [],
            "allergies": p.get("allergies", []) or [],
        },
        "doctor": {
            "name": d.get("name", "N/A"),
            "department": d.get("department", "N/A"),
            "phone": d.get("phone", "N/A"),
            "email": d.get("email", "N/A"),
        },
        "generated_at": generated_at,
        "normalized_key": norm_key,
    }
    write_last_context(context)

    # also write a header + answer into txt
    header = (
        "Patient Details\n"
        f"Patient ID: {_to_na(data.patient_id)}\n"
        f"Patient Name: {_to_na(p.get('name'))}\n"
        f"Age: {_to_na(p.get('age'))}\n"
        f"Gender: {_to_na(p.get('gender'))}\n"
        f"Phone: {_to_na(p.get('phone'))}\n"
        f"Chronic Conditions: {_to_na(p.get('chronic_conditions', []))}\n"
        f"Allergies: {_to_na(p.get('allergies', []))}\n"
        f"Doctor ID: {_to_na(data.doctor_id)}\n"
        f"Doctor Name: {_to_na(d.get('name'))}\n"
        f"Department: {_to_na(d.get('department'))}\n"
        f"Generated At: {_to_na(generated_at)}\n"
        "\n----------------------------------------\n\n"
    )
    write_last_validation(header + (answer or ""))

    # store solved case
    try:
        await store_solved_case(
            patient_id=data.patient_id,
            doctor_id=data.doctor_id,
            query=data.query,
            normalized_key=norm_key,
            answer=answer,
            confidence=0.85,
            source="rag",
        )
    except Exception as e:
        print("Mongo store failed:", repr(e))

    # store visit
    try:
        await add_visit(
            patient_id=data.patient_id,
            doctor_id=data.doctor_id,
            complaint=data.query,
            diagnosis="",
            prescription_text="",
        )
    except Exception as e:
        print("Mongo visit failed:", repr(e))

    return {
        "source": "rag",
        "answer": answer,
        "used_previous_case": bool(previous_answer),
        "previous_answer_from_mongo": previous_answer,
        "normalized_key": norm_key,
        "last_validation_path": str(LAST_VALIDATION_PATH),
        "last_context_path": str(LAST_CONTEXT_PATH),
        "last_validation_written": True,
        "last_context_written": True,
    }


@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    global _OCR_READER

    try:
        if _OCR_READER is None:
            _OCR_READER = easyocr.Reader(["en"], gpu=False)

        suffix = os.path.splitext(file.filename or "")[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        if img is None:
            return {"text": "", "error": "Unable to read uploaded image."}

        # --- OCR (multi-pass for handwriting) ---
        all_lines: List[str] = []
        for variant in _prepare_ocr_images(img):
            try:
                lines = _OCR_READER.readtext(variant, detail=0)
                all_lines.extend(lines)
            except Exception:
                continue

        raw = " ".join([str(x) for x in all_lines if str(x).strip()]).strip()
        if not raw:
            return {"text": "", "error": "OCR returned empty text (handwriting may need preprocessing)."}

        extracted = extract_medicine_names(raw, max_items=5)

        return {
            "text": extracted["text"],
            "raw": raw,
            "tokens": extracted["raw_tokens"],
            "matches": extracted["matches"],
            "llm_text": extracted["llm_text"],
            "mode": "normalized_meds"
        }

    except Exception as e:
        return {"text": "", "error": f"OCR processing failed: {type(e).__name__}: {e}"}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/stt/audio")
async def stt_audio(file: UploadFile = File(...)):
    """
    Receives an audio file and returns transcribed text using faster-whisper.
    Supports wav/mp3/webm (webm is common from browser recording).
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        return {
            "text": "",
            "error": f"STT dependency missing or failed to import: {type(e).__name__}: {e}. Install faster-whisper and ffmpeg.",
        }

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in [".wav", ".mp3", ".m4a", ".webm", ".ogg"]:
        suffix = ".webm"  # safe default for browser recordings

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        try:
            WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            # Use a global model cache (loads once) to make it fast
            if not hasattr(stt_audio, "_model"):
                model_path = _get_local_whisper_model_path()
                stt_audio._model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",
                )

            segments, info = stt_audio._model.transcribe(tmp_path)
            raw_text = " ".join(seg.text.strip() for seg in segments).strip()
            extracted = extract_medicine_names(raw_text, max_items=5)
            final_text = extracted.get("llm_mapped_text") or extracted["text"] or raw_text
            return {
                "text": final_text,
                "raw_text": raw_text,
                "tokens": extracted["raw_tokens"],
                "matches": extracted["matches"],
                "llm_text": extracted["llm_text"],
                "llm_mapped_text": extracted.get("llm_mapped_text", ""),
                "language": getattr(info, "language", None),
                "mode": "faster-whisper+normalized_meds",
            }
        except Exception as e:
            # Auto-recover from a corrupted/incomplete downloaded model.
            if "model.bin" in str(e).lower():
                model_path = _get_local_whisper_model_path()
                stt_audio._model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",
                )
                segments, info = stt_audio._model.transcribe(tmp_path)
                raw_text = " ".join(seg.text.strip() for seg in segments).strip()
                extracted = extract_medicine_names(raw_text, max_items=5)
                final_text = extracted.get("llm_mapped_text") or extracted["text"] or raw_text
                return {
                    "text": final_text,
                    "raw_text": raw_text,
                    "tokens": extracted["raw_tokens"],
                    "matches": extracted["matches"],
                    "llm_text": extracted["llm_text"],
                    "llm_mapped_text": extracted.get("llm_mapped_text", ""),
                    "language": getattr(info, "language", None),
                    "mode": "faster-whisper+normalized_meds",
                }
            return {"text": "", "error": f"STT failed: {type(e).__name__}: {e}"}

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
