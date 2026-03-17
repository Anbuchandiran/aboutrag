# module2_rag/src/api.py

from fastapi import FastAPI, UploadFile, File , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re
import json
from pathlib import Path
from datetime import datetime
import tempfile
import os


from .validate_prescription import run_validation_multi



import pandas as pd
import numpy as np
import cv2
import easyocr
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from .alert_service import generate_and_alert, extract_status
from .ocr_service import (
    extract_text_from_image_bytes,
    preprocess_image,
    normalize_prescription_text,
)

# ✅ NEW (only additions needed for drugnames Chroma)
import chromadb
from chromadb.utils import embedding_functions

from .validate_prescription import run_validation_multi
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
    allow_origin_regex=r"(https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?$)|(chrome-extension://.*)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- PATHS --------------------
# api.py is at <repo>/module2_rag/src/api.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE2_RAG_DIR = PROJECT_ROOT / "module2_rag"
DATA_DIR = MODULE2_RAG_DIR / "data"

LAST_VALIDATION_PATH = DATA_DIR / "last_validation.txt"
LAST_CONTEXT_PATH = DATA_DIR / "last_context.json"

# -------------------- OCR / DRUGBANK CACHES --------------------
_OCR_READER = None
_DRUG_NAMES = None  # list[str]
_DRUG_NAME_SET = None  # set[str]

# ✅ NEW: Drug-names Chroma (only additions; does not affect other logic)
DRUGNAMES_DIR = str(MODULE2_RAG_DIR / "db" / "chroma_drugnames")
DRUGNAMES_COLLECTION = "drug_names"
_drugnames_client = None
_drugnames_col = None
_drugnames_embed_fn = None
OCR_DISTANCE_THRESHOLD = 0.55
WHISPER_MODEL_NAME = "small"
WHISPER_MODEL_DIR = PROJECT_ROOT / ".models" / "faster-whisper"
GENERIC_SYNONYMS = {
    "paracetamol": "acetaminophen",
    "paracetomol": "acetaminophen",
    "paracetmol": "acetaminophen",
    "paracitamol": "acetaminophen",
    "paracatamol": "acetaminophen",
    "fracetoml": "acetaminophen",
    "fracetamol": "acetaminophen",
}
DISPLAY_SYNONYMS = {
    "acetaminophen": "paracetamol",
}


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
        "paracetomol": "paracetamol",
        "paracetmol": "paracetamol",
        "paracitamol": "paracetamol",
        "paracatamol": "paracetamol",
        "fracetoml": "paracetamol",
        "fracetamol": "paracetamol",
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
        "dqoxis": "digoxin",
        "dqoxin": "digoxin",
        "qoxim": "digoxin",
        "plicamyein": "plicamycin",
        "plicamyin": "plicamycin",
        "plicamyein": "plicamycin",
        "plicamycin": "plicamycin",
        "plitamm": "plicamycin",
        "plitamw": "plicamycin",
        "plitawm": "plicamycin",
        "plitewm": "plicamycin",
        "plitamum": "plicamycin",
        "plitamyw": "plicamycin",
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
    variants.append(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    variants.append(cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)
    variants.append(cv2.resize(clahe, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    variants.append(thr)
    variants.append(cv2.resize(thr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.resize(otsu, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))

    return variants


def _extract_text_regions(img) -> List[Any]:
    """
    Detect likely handwritten text bands and crop them before OCR.
    This prevents EasyOCR from re-reading the full page background on every pass.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
    merged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    min_area = max(500, int(h * w * 0.003))

    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area:
            continue
        if bw < w * 0.12 or bh < h * 0.05:
            continue

        pad_x = max(8, int(bw * 0.05))
        pad_y = max(8, int(bh * 0.20))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)
        boxes.append((x0, y0, x1, y1))

    if not boxes:
        return [img]

    boxes.sort(key=lambda box: (box[1], box[0]))

    merged_boxes = []
    for box in boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue

        px0, py0, px1, py1 = merged_boxes[-1]
        x0, y0, x1, y1 = box
        overlap_y = min(py1, y1) - max(py0, y0)
        same_line = overlap_y > -max(10, int(h * 0.02))
        if same_line:
            merged_boxes[-1] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
        else:
            merged_boxes.append(box)

    valid_boxes = []
    for x0, y0, x1, y1 in merged_boxes:
        bw = x1 - x0
        bh = y1 - y0
        if bw >= w * 0.18 and bh >= h * 0.08:
            valid_boxes.append((x0, y0, x1, y1))

    if not valid_boxes:
        return [img]

    return [img[y0:y1, x0:x1] for (x0, y0, x1, y1) in valid_boxes]


def _ocr_region_candidates(reader, region_img) -> List[Dict[str, Any]]:
    """
    Run OCR on one cropped region and keep scored candidates.
    """
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for variant in _prepare_ocr_images(region_img):
        try:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,/-+ ",
            )
        except Exception:
            try:
                results = reader.readtext(variant, detail=1, paragraph=False)
            except Exception:
                continue

        for item in results:
            if len(item) < 3:
                continue
            _, text, confidence = item
            clean = re.sub(r"\s+", " ", str(text)).strip()
            if not clean:
                continue

            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)

            letters = sum(ch.isalpha() for ch in clean)
            total = max(len(clean), 1)
            letter_ratio = letters / total
            noise_penalty = sum(ch.isdigit() or ch in "_&%$#@*=" for ch in clean)
            score = float(confidence) + letter_ratio - (noise_penalty * 0.15)

            candidates.append(
                {
                    "text": clean,
                    "confidence": float(confidence),
                    "score": score,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _looks_like_useful_ocr(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return False
    alpha = sum(ch.isalpha() for ch in clean)
    if alpha < 3:
        return False
    words = [w for w in re.split(r"[\s,;/|+-]+", clean.lower()) if w]
    long_words = [w for w in words if sum(ch.isalpha() for ch in w) >= 4]
    return bool(long_words)


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
        if t in drug_list:
            mapped_name = normalize_known_drug_name(t)
            if mapped_name not in mapped:
                mapped.append(mapped_name)
        else:
            fb, sc = fuzzy_match_drug(t, drug_list)
            if fb and sc >= min_score and fb not in mapped:
                mapped.append(fb)
        if len(mapped) >= max_items:
            break
    return mapped


def _read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def normalize_known_drug_name(name: str) -> str:
    return GENERIC_SYNONYMS.get((name or "").strip().lower(), (name or "").strip().lower())


def display_drug_name(name: str) -> str:
    return DISPLAY_SYNONYMS.get((name or "").strip().lower(), (name or "").strip().lower())


def load_drugbank_names() -> List[str]:
    """
    Loads drug names from local datasets and normalizes common generic aliases.
    Cached after first load.
    """
    global _DRUG_NAMES, _DRUG_NAME_SET
    if _DRUG_NAMES is not None:
        return _DRUG_NAMES

    names = set()
    dataset_specs = [
        (DATA_DIR / "drugbank_docs.csv", ("drug1", "drug2")),
        (DATA_DIR / "drugbank_interactions.csv", ("drug1", "drug2")),
        (DATA_DIR / "real_drug_dataset.csv", ("Drug_Name",)),
    ]

    for csv_path, columns in dataset_specs:
        if not csv_path.exists():
            continue

        try:
            df = _read_csv_safe(csv_path)
        except Exception as e:
            print("Failed to load drug names:", str(csv_path), repr(e))
            continue

        for col in columns:
            if col not in df.columns:
                continue
            for x in df[col].dropna().astype(str).tolist():
                x = x.strip().lower()
                if x and x not in {"drug1", "drug2", "text"}:
                    names.add(x)

    names.update(x.strip().lower() for x in GENERIC_SYNONYMS.keys())
    names.update(normalize_known_drug_name(x) for x in GENERIC_SYNONYMS.values())

    _DRUG_NAMES = sorted(names)
    _DRUG_NAME_SET = set(_DRUG_NAMES)
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
    return normalize_known_drug_name(match), int(score)


def raw_token_support_score(token: str, raw_tokens: List[str]) -> int:
    """
    Score how well a final medicine candidate is supported by the original OCR tokens.
    """
    token = normalize_known_drug_name(token)
    best_score = 0
    for raw in raw_tokens:
        raw = (raw or "").strip().lower()
        if not raw:
            continue

        lev_score = int(Levenshtein.normalized_similarity(raw, token) * 100)
        ratio_score = int(fuzz.ratio(raw, token))
        best_score = max(best_score, lev_score, ratio_score)
    return best_score



def llm_fix_drug_text(raw_text: str) -> str:
    try:
        from google import genai
        GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not GEMINI_KEY: return raw_text
        
        client = genai.Client(api_key=GEMINI_KEY)
        prompt = f"""
        TRANSCRIPT: "{raw_text}"
        TASK: Identify medicine names. Correct phonetic errors (e.g. 'war for in' -> 'warfarin').
        Return ONLY names separated by ' + '. Output nothing else.
        """.strip()

        resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
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


def extract_medicine_names(
    raw_text: str,
    max_items: int = 5,
    use_llm_cleanup: bool = True,
) -> Dict[str, Any]:
    """
    Normalize noisy OCR/STT text into likely medicine names.
    """
    raw_clean = re.sub(r"[^a-zA-Z0-9\s\+\-/,;]", " ", (raw_text or "").lower())
    raw_clean = re.sub(r"\s+", " ", raw_clean).strip()
    raw_split = [t for t in re.split(r"[+\s,;/]+", raw_clean) if t]
    raw_tokens = [_normalize_common_typos(t) for t in _merge_fragmented_tokens(raw_split) if len(t) > 2]

    llm_text = llm_fix_drug_text(raw_clean) if use_llm_cleanup else ""
    llm_clean = re.sub(r"[^a-zA-Z0-9\s\+\-/,;]", " ", (llm_text or "").lower())
    llm_clean = re.sub(r"\s+", " ", llm_clean).strip()
    llm_split = [t for t in re.split(r"[+\s,;/]+", llm_clean) if t]
    llm_tokens = [_normalize_common_typos(t) for t in _merge_fragmented_tokens(llm_split) if len(t) > 2]

    def _is_likely_med_token(token: str) -> bool:
        t = (token or "").strip().lower()
        if len(t) < 3:
            return False
        letters = sum(ch.isalpha() for ch in t)
        digits = sum(ch.isdigit() for ch in t)
        # Reject obvious OCR garbage like "71i", "pk", etc.
        if letters < 3:
            return False
        if digits > 0 and letters <= digits:
            return False
        return True

    raw_tokens = [t for t in raw_tokens if _is_likely_med_token(t)]
    llm_tokens = [t for t in llm_tokens if _is_likely_med_token(t)]

    stopwords = {
        "take", "tablet", "tablets", "tab", "capsule", "capsules", "cap",
        "morning", "night", "after", "before", "food", "daily", "twice",
        "once", "days", "day", "for", "and", "the",
    }

    candidates: List[str] = []
    for t in raw_tokens + llm_tokens:
        if t not in stopwords and t not in candidates:
            candidates.append(t)

    out: List[str] = []
    debug: List[Dict[str, Any]] = []
    soft_votes: Dict[str, Dict[str, Any]] = {}
    try:
        drug_list = load_drugbank_names()
    except Exception:
        drug_list = []
    llm_out: List[str] = _map_tokens_to_known_drugs(llm_tokens, drug_list, max_items=max_items, min_score=80) if drug_list else []

    for token in candidates:
        best, dist = correct_token_with_chroma(token)
        best = normalize_known_drug_name(best)
        used = None
        score = None

        normalized_token = normalize_known_drug_name(token)
        raw_support = raw_token_support_score(normalized_token, raw_tokens)
        token_from_raw = token in raw_tokens
        token_from_llm_only = token in llm_tokens and not token_from_raw

        token_is_known_drug = token in drug_list or normalized_token in drug_list
        has_exact_support = (
            token_from_raw or
            (not token_from_llm_only and raw_support >= 50) or
            (token_from_llm_only and raw_support >= 80)
        )

        if token_is_known_drug and has_exact_support:
            used = normalized_token
            debug.append(
                {
                    "token": token,
                    "via": "exact",
                    "best": normalized_token,
                    "raw_support": raw_support,
                    "source": "raw" if token_from_raw else "llm",
                }
            )

        elif best and dist < 0.25 and len(token) >= 5:
            used = best
            debug.append({"token": token, "via": "chroma", "best": best, "dist": dist})

        elif drug_list:
            f_best, score = fuzzy_match_drug(token, drug_list)
            # LLM-derived tokens can be accepted with slightly lower fuzzy score.
            min_score = 92 if token in llm_tokens else 95
            if f_best and score >= min_score:
                used = f_best
                debug.append({"token": token, "via": "fuzzy", "best": f_best, "score": score})
            else:
                debug.append({"token": token, "via": "none", "best": best, "dist": dist, "score": score if drug_list else None})
        else:
            debug.append({"token": token, "via": "none", "best": best, "dist": dist})

        if not used and drug_list:
            f_best, score = fuzzy_match_drug(token, drug_list)
            if f_best and len(token) >= 6 and score >= 68:
                vote = soft_votes.setdefault(f_best, {"votes": 0, "best_score": 0, "tokens": []})
                vote["votes"] += 1
                vote["best_score"] = max(vote["best_score"], score)
                if token not in vote["tokens"]:
                    vote["tokens"].append(token)

        if not used and best and len(token) >= 6 and dist <= OCR_DISTANCE_THRESHOLD:
            vote = soft_votes.setdefault(best, {"votes": 0, "best_score": 0, "tokens": []})
            vote["votes"] += 1
            vote["best_score"] = max(vote["best_score"], int((1.0 - min(dist, 1.0)) * 100))
            if token not in vote["tokens"]:
                vote["tokens"].append(token)

        if used and used not in out:
            out.append(used)
        if len(out) >= max_items:
            break

    if not out and soft_votes:
        ranked_votes = sorted(
            soft_votes.items(),
            key=lambda item: (item[1]["votes"], item[1]["best_score"], len(item[0])),
            reverse=True,
        )
        best_name, meta = ranked_votes[0]
        runner_up_votes = ranked_votes[1][1]["votes"] if len(ranked_votes) > 1 else 0
        if meta["votes"] >= 2 and (meta["votes"] > runner_up_votes or meta["best_score"] >= 78):
            out.append(best_name)
            debug.append({
                "token": " | ".join(meta["tokens"]),
                "via": "soft_vote",
                "best": best_name,
                "votes": meta["votes"],
                "score": meta["best_score"],
            })

    # last fallback so UI always gets usable text
    if not out and raw_tokens:
        # Prefer cleaned LLM tokens if they map to known medicines.
        if llm_out:
            out = llm_out[:max_items]
        else:
            out = [t for t in raw_tokens if _is_likely_med_token(t)][:max_items]

    return {
        "text": " + ".join(out[:max_items]),
        "llm_mapped_text": " + ".join(llm_out[:max_items]),
        "raw_tokens": raw_tokens,
        "llm_text": llm_text,
        "matches": debug,
        "soft_votes": soft_votes,
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
    try:
        p = await patients.find_one({"_id": data.patient_id}) or {}
    except Exception as e:
        print("Patient fetch failed:", repr(e))
        p = {}

    try:
        d = await doctors.find_one({"_id": data.doctor_id}) or {}
    except Exception as e:
        print("Doctor fetch failed:", repr(e))
        d = {}

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

    # trigger email/pdf/sms alert only for NOT SAFE
    try:
        status = extract_status(answer or "")
        if status == "NOT SAFE":
            patient_info = {
                "patient_id": data.patient_id,
                "name": p.get("name", "N/A"),
                "age": p.get("age", "N/A"),
                "gender": p.get("gender", "N/A"),
                "phone": p.get("phone", "N/A"),
                "chronic_conditions": p.get("chronic_conditions", []) or [],
                "allergies": p.get("allergies", []) or [],
                "doctor_id": data.doctor_id,
                "doctor_name": d.get("name", "N/A"),
                "department": d.get("department", "N/A"),
                "generated_at": generated_at,
            }

            alert_result = generate_and_alert(
                validation_text=answer or "",
                patient_info=patient_info,
                ctx=context,
                always_send=False,
            )

            print("ALERT RESULT:", alert_result)

    except Exception as e:
        print("Alert generation failed:", repr(e))

    # store solved case

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
        file_bytes = await file.read()
        if not file_bytes:
            return {"text": "", "error": "Uploaded image is empty."}

        if _OCR_READER is None:
            _OCR_READER = easyocr.Reader(["en"], gpu=False)

        candidate_payloads: List[Dict[str, Any]] = []

        # Candidate 1: simple in-memory helper.
        try:
            ocr_result = extract_text_from_image_bytes(file_bytes)
            helper_raw = normalize_prescription_text(ocr_result.get("raw_text", ""))
            helper_lines = [
                normalize_prescription_text(line)
                for line in ocr_result.get("lines", [])
                if str(line).strip()
            ]
            if helper_raw:
                candidate_payloads.append(
                    {
                        "source": "helper",
                        "raw_text": helper_raw,
                        "lines": helper_lines,
                        "avg_confidence": float(ocr_result.get("avg_confidence", 0.0) or 0.0),
                    }
                )
        except Exception as e:
            candidate_payloads.append(
                {
                    "source": "helper_error",
                    "raw_text": "",
                    "lines": [],
                    "avg_confidence": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

        # Candidate 2: region-aware OCR over enhanced image variants.
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"text": "", "error": "Unable to read uploaded image."}

        region_lines: List[str] = []
        region_confidences: List[float] = []
        try:
            for region in _extract_text_regions(img):
                region_candidates = _ocr_region_candidates(_OCR_READER, region)
                if region_candidates:
                    region_lines.append(normalize_prescription_text(region_candidates[0]["text"]))
                    region_confidences.append(float(region_candidates[0]["confidence"]))
        except Exception:
            region_lines = []
            region_confidences = []

        if region_lines:
            candidate_payloads.append(
                {
                    "source": "region",
                    "raw_text": normalize_prescription_text("\n".join(region_lines)),
                    "lines": region_lines,
                    "avg_confidence": sum(region_confidences) / max(len(region_confidences), 1),
                }
            )

        # Candidate 3: paragraph OCR on preprocessed + variant images.
        fallback_lines: List[str] = []
        seen_lines = set()

        try:
            preprocessed_img = preprocess_image(file_bytes)
            pre_lines = _OCR_READER.readtext(preprocessed_img, detail=0, paragraph=True) or []
        except Exception:
            pre_lines = []

        for line in pre_lines:
            clean_line = normalize_prescription_text(str(line))
            key = clean_line.lower()
            if clean_line and key not in seen_lines:
                seen_lines.add(key)
                fallback_lines.append(clean_line)

        for variant in _prepare_ocr_images(img):
            try:
                variant_lines = _OCR_READER.readtext(variant, detail=0, paragraph=True) or []
            except Exception:
                try:
                    variant_lines = _OCR_READER.readtext(variant, detail=0) or []
                except Exception:
                    continue

            for line in variant_lines:
                clean_line = normalize_prescription_text(str(line))
                key = clean_line.lower()
                if clean_line and key not in seen_lines:
                    seen_lines.add(key)
                    fallback_lines.append(clean_line)

        if fallback_lines:
            candidate_payloads.append(
                {
                    "source": "variants",
                    "raw_text": normalize_prescription_text("\n".join(fallback_lines)),
                    "lines": fallback_lines,
                    "avg_confidence": 0.0,
                }
            )

        best_payload = None
        best_score = -1.0
        best_extracted: Dict[str, Any] = {}

        for payload in candidate_payloads:
            raw_text = (payload.get("raw_text") or "").strip()
            if not raw_text:
                continue

            extracted = extract_medicine_names(raw_text, max_items=5, use_llm_cleanup=False)
            normalized_text = (extracted.get("text") or "").strip()
            useful = _looks_like_useful_ocr(raw_text)

            score = 0.0
            if normalized_text:
                score += 10.0
            if useful:
                score += 3.0
            score += float(payload.get("avg_confidence", 0.0) or 0.0)

            if score > best_score:
                best_score = score
                best_payload = payload
                best_extracted = extracted

        if not best_payload:
            return {
                "text": "",
                "raw": "",
                "lines": [],
                "error": "OCR returned empty text. Try a clearer image or higher contrast."
            }

        raw_text = best_payload["raw_text"]
        lines = best_payload["lines"]
        extracted = best_extracted or extract_medicine_names(raw_text, max_items=5, use_llm_cleanup=False)
        normalized_text = (extracted.get("text") or "").strip()
        display_text = " + ".join(
            display_drug_name(part.strip())
            for part in normalized_text.split("+")
            if part.strip()
        ).strip()
        output_text = display_text or normalized_text or raw_text

        return {
            "text": output_text,
            "raw": raw_text,
            "lines": lines,
            "normalized_text": normalized_text,
            "display_text": display_text,
            "tokens": extracted.get("raw_tokens", []),
            "matches": extracted.get("matches", []),
            "llm_text": extracted.get("llm_text", ""),
            "ocr_source": best_payload.get("source", "unknown"),
            "avg_confidence": best_payload.get("avg_confidence", 0.0),
            "mode": "ocr-text"
        }

    except Exception as e:
        return {"text": "", "error": f"OCR processing failed: {type(e).__name__}: {e}"}


@app.post("/stt/audio")
async def stt_audio(file: UploadFile = File(...)):
    tmp_path = None
    try:
        from faster_whisper import WhisperModel
        
        # 1. Save the incoming audio
        suffix = os.path.splitext(file.filename or "")[1].lower() or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2. Initialize Model (Using "small" for better accuracy than "base")
        if not hasattr(stt_audio, "_model"):
            model_path = _get_local_whisper_model_path()
            stt_audio._model = WhisperModel(model_path, device="cpu", compute_type="int8")

        # 3. MEDICAL PRIMING: We tell Whisper what words to "expect"
        # This helps it distinguish "Warfarin" from "War for in"
        med_prompt = (
            "Medicine names: Atorvastatin, Warfarin, Digoxin, Clopidogrel, "
            "Metformin, Aspirin, Paracetamol, Amoxicillin, Ibuprofen, Lisinopril."
        )

        segments, info = stt_audio._model.transcribe(
            tmp_path, 
            initial_prompt=med_prompt, 
            beam_size=5
        )
        raw_text = " ".join(seg.text.strip() for seg in segments).strip()

        # 4. PHONETIC CLEANUP: Pass the text to Gemini to fix spelling
        # This is what makes it "Perfect"
        extracted = extract_medicine_names(raw_text, max_items=5, use_llm_cleanup=True)
        final_text = extracted.get("text") or raw_text

        # 5. VALIDATION: Check for interactions
        # Note: Passing as positional argument to avoid "unexpected keyword" error
        answer = run_validation_multi(
            final_text, 
            {
                "age": None, 
                "allergies": [], 
                "conditions": [], 
                "current_meds": [], 
                "history_text": ""
            }
        )

        return {
            "text": final_text,
            "raw_transcription": raw_text,
            "validation": answer,
            "mode": "optimized-medical-voice"
        }

    except Exception as e:
        # Log error to terminal for debugging
        print(f"--- VOICE ERROR --- \n{e}")
        return {"text": "", "error": f"STT failed: {str(e)}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
