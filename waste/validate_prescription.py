import os
import re
import csv
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from google import genai

# ---------------- PATHS / CONFIG ----------------
MODULE2_RAG_BASE = os.getenv("MODULE2_RAG_BASE", "module2_rag")

DDI_PATH = os.path.join(MODULE2_RAG_BASE, "data", "drugbank_interactions.csv")
CHROMA_DIR = os.path.join(MODULE2_RAG_BASE, "db", "chroma_drugbank")

# ✅ API KEY
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("Missing API key. Set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY.")

GEMINI_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=GEMINI_KEY)

SYNONYMS = {
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
}

# ---------------- LOAD STRUCTURED DDI CSV ----------------
def _load_ddi_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path):
            return None

        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"Drug1": "drug1", "Drug2": "drug2", "Effect": "effect"})
        df = df.rename(columns={"drug_1": "drug1", "drug_2": "drug2"})

        if not {"drug1", "drug2", "effect"}.issubset(set(df.columns)):
            return None

        df["drug1"] = df["drug1"].astype(str).str.lower().str.strip()
        df["drug2"] = df["drug2"].astype(str).str.lower().str.strip()
        df["effect"] = df["effect"].astype(str).str.strip()
        return df
    except Exception as e:
        print("Failed to load interaction dataset:", e)
        return None


ddi = _load_ddi_csv(DDI_PATH)

def _reload_ddi():
    global ddi
    ddi = _load_ddi_csv(DDI_PATH)

def norm(x: str) -> str:
    return str(x).strip().lower()

# ---------------- STRUCTURED LOOKUP ----------------
def find_interaction(drug_a: str, drug_b: str) -> Optional[Dict[str, Any]]:
    global ddi
    if ddi is None:
        return None

    a = norm(drug_a)
    b = norm(drug_b)

    # exact
    hit = ddi[
        ((ddi["drug1"] == a) & (ddi["drug2"] == b)) |
        ((ddi["drug1"] == b) & (ddi["drug2"] == a))
    ]
    if len(hit):
        return hit.iloc[0].to_dict()

    # word-boundary contains (avoid aspirin matching nitroaspirin)
    pattern_a = rf"\b{re.escape(a)}\b"
    pattern_b = rf"\b{re.escape(b)}\b"
    hit2 = ddi[
        (ddi["drug1"].str.contains(pattern_a, na=False, regex=True) &
         ddi["drug2"].str.contains(pattern_b, na=False, regex=True)) |
        (ddi["drug1"].str.contains(pattern_b, na=False, regex=True) &
         ddi["drug2"].str.contains(pattern_a, na=False, regex=True))
    ]
    if len(hit2):
        return hit2.iloc[0].to_dict()

    return None

def ddi_exists(drug_a: str, drug_b: str) -> bool:
    return find_interaction(drug_a, drug_b) is not None

# ---------------- CHROMA: SEARCH ALL COLLECTIONS ----------------
def _get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def get_all_chroma_collections():
    """
    Returns all collections present in your CHROMA_DIR.
    This makes sure your RAG searches DrugBank + real_drug + extras (everything).
    """
    client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = _get_embed_fn()

    cols = []
    try:
        items = client_chroma.list_collections()
    except Exception as e:
        print("Chroma list_collections failed:", e)
        return cols

    # Chroma versions differ: list may return objects or names
    for item in items:
        try:
            name = item.name if hasattr(item, "name") else str(item)
            col = client_chroma.get_collection(name=name, embedding_function=embed_fn)
            cols.append(col)
        except Exception:
            continue

    return cols

def multi_query(cols, query: str, n_per_col: int = 3) -> List[str]:
    docs_all: List[str] = []
    for col in cols:
        try:
            res = col.query(query_texts=[query], n_results=n_per_col)
            docs = res["documents"][0] if res.get("documents") else []
            docs_all.extend(docs)
        except Exception:
            pass

    # de-dup preserve order
    seen = set()
    out = []
    for d in docs_all:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out

def retrieve_drug_evidence_all(drug: str, cols) -> str:
    q = f"{drug} contraindications interactions warnings toxicity dosage side effects"
    docs = multi_query(cols, q, n_per_col=3)
    return ("\n\n".join(docs[:10]))[:4000]

def retrieve_interaction_evidence_all(drugs: List[str], cols) -> str:
    if len(drugs) < 2:
        return ""
    chunks = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a, b = drugs[i], drugs[j]
            q = f"{a} {b} interaction contraindication warning adverse effects"
            docs = multi_query(cols, q, n_per_col=2)
            if docs:
                chunks.append(
                    f"### Interaction evidence (ALL collections): {a} + {b}\n" +
                    "\n\n".join(docs[:4])
                )
    return ("\n\n".join(chunks))[:4000]

# ---------------- SELF-LEARNING STORAGE ----------------
def append_ddi_to_csv(drug_a: str, drug_b: str, effect: str, path: str = DDI_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["drug1", "drug2", "effect"])
        w.writerow([norm(drug_a), norm(drug_b), effect.strip()])

def add_ddi_to_chroma(drug_a: str, drug_b: str, effect: str, doctor_id: str = "", patient_id: str = ""):
    """
    Stores the LLM validated DDI into ALL collections?
    ✅ We store into ONE dedicated collection: llm_ddi_knowledge
    (keeps your main datasets clean)
    """
    client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = _get_embed_fn()

    col = client_chroma.get_or_create_collection(
        name="llm_ddi_knowledge",
        embedding_function=embed_fn
    )

    # stable id avoids duplicates forever
    a = norm(drug_a)
    b = norm(drug_b)
    ddi_id = f"llm_ddi::{a}::{b}" if a <= b else f"llm_ddi::{b}::{a}"

    doc = (
        "[LLM_VALIDATED_DDI]\n"
        f"drug1: {a}\n"
        f"drug2: {b}\n"
        f"effect: {effect.strip()}\n"
        f"created_at: {datetime.utcnow().isoformat()}Z\n"
    )

    try:
        col.upsert(
            ids=[ddi_id],
            documents=[doc],
            metadatas=[{
                "type": "llm_validated_ddi",
                "drug1": a,
                "drug2": b,
                "doctor_id": doctor_id,
                "patient_id": patient_id,
            }]
        )
    except Exception:
        # fallback for older chroma that may not have upsert
        col.add(
            ids=[ddi_id],
            documents=[doc],
            metadatas=[{
                "type": "llm_validated_ddi",
                "drug1": a,
                "drug2": b,
                "doctor_id": doctor_id,
                "patient_id": patient_id,
            }]
        )

# ---------------- PARSING GEMINI OUTPUT ----------------
def normalize_status(s: str) -> str:
    s = (s or "").strip().upper()
    if "NOT SAFE" in s or "NOT_SAFE" in s or "NOT-SAFE" in s or s == "NOT":
        return "NOT SAFE"
    if "CAUTION" in s:
        return "CAUTION"
    if s == "SAFE":
        return "SAFE"
    if "INSUFFICIENT" in s:
        return "INSUFFICIENT"
    return s

def parse_overall_status(answer_text: str) -> str:
    for line in (answer_text or "").splitlines():
        if "overall_status" in line.lower():
            parts = line.split(":", 1)
            if len(parts) == 2:
                return normalize_status(parts[1])
    return ""

def parse_interaction_lines(answer_text: str) -> List[Tuple[str, str, str, str]]:
    """
    Parses lines like:
    - drugA + drugB: SAFE/CAUTION/NOT SAFE/INSUFFICIENT (reason)
    Returns: (a,b,status,full_text)
    """
    out: List[Tuple[str, str, str, str]] = []
    for line in (answer_text or "").splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue

        body = line[1:].strip()
        if ":" not in body:
            continue

        left, right = body.split(":", 1)
        left = left.strip()
        right = right.strip()

        if "+" not in left:
            continue

        a, b = [norm(x) for x in left.split("+", 1)]

        r_upper = right.upper()
        if "NOT SAFE" in r_upper or "NOT_SAFE" in r_upper or "NOT-SAFE" in r_upper:
            status_guess = "NOT SAFE"
        elif "CAUTION" in r_upper:
            status_guess = "CAUTION"
        elif r_upper.startswith("SAFE"):
            status_guess = "SAFE"
        elif "INSUFFICIENT" in r_upper:
            status_guess = "INSUFFICIENT"
        else:
            status_guess = right.split()[0] if right else ""

        status = normalize_status(status_guess)
        if a and b and status:
            out.append((a, b, status, right.strip()))
    return out

# ---------------- GEMINI (2-PASS FALLBACK) ----------------
def ask_gemini_multi(drugs: List[str], patient: dict, combined_evidence: str) -> str:
    prompt1 = f"""
You are a clinical decision-support AI.

Prescription drugs: {", ".join(drugs)}

Patient Profile:
Age: {patient.get('age')}
Allergies: {patient.get('allergies')}
Conditions: {patient.get('conditions')}
Current Medications: {patient.get('current_meds')}

Medical Evidence (may be incomplete):
{combined_evidence}

IMPORTANT:
- If a section named STRUCTURED_DDI_MATCHES is present, treat it as high-confidence evidence.
- Evidence may be missing; still use general medical knowledge.
- Only answer INSUFFICIENT if you genuinely cannot assess.

Return ONLY this format:

Overall_Status: SAFE / CAUTION / NOT SAFE / INSUFFICIENT
Key_Reason: 1-2 lines
Interactions:
- drugA + drugB: SAFE/CAUTION/NOT SAFE/INSUFFICIENT (1 short reason)
Per_Drug_Notes:
- drug: SAFE/CAUTION/NOT SAFE/INSUFFICIENT (1 short reason)
Action: 1 short line for doctor
""".strip()

    try:
        r1 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt1)
        ans1 = (r1.text or "").strip()
    except Exception as e:
        return (
            "Overall_Status: INSUFFICIENT\n"
            f"Key_Reason: Gemini API call failed ({type(e).__name__}).\n"
            "Interactions:\n- (none)\n"
            "Per_Drug_Notes:\n- (none)\n"
            "Action: Check Gemini quota/billing + retry.\n"
        )

    overall1 = normalize_status(parse_overall_status(ans1))
    if overall1 and overall1 != "INSUFFICIENT":
        return ans1

    # PASS 2: general fallback
    prompt2 = f"""
You are a clinical decision-support AI.

Prescription drugs: {", ".join(drugs)}

Use general medical knowledge of drug interactions even if evidence is missing.
Only answer INSUFFICIENT if you genuinely cannot assess.

Return ONLY this format:

Overall_Status: SAFE / CAUTION / NOT SAFE / INSUFFICIENT
Key_Reason: 1-2 lines
Interactions:
- drugA + drugB: SAFE/CAUTION/NOT SAFE/INSUFFICIENT (1 short reason)
Per_Drug_Notes:
- drug: SAFE/CAUTION/NOT SAFE/INSUFFICIENT (1 short reason)
Action: 1 short line for doctor
""".strip()

    try:
        r2 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt2)
        ans2 = (r2.text or "").strip()
        return ans2 if ans2 else ans1
    except Exception:
        return ans1

# ---------------- INPUT PARSING ----------------
def parse_drug_list(text: str) -> List[str]:
    raw = (text or "").lower().strip()
    parts = re.split(r"[,+;/]| and |\+|\n", raw)

    drugs: List[str] = []
    for p in parts:
        d = p.strip()
        if not d:
            continue
        d = SYNONYMS.get(d, d)
        drugs.append(d)

    seen = set()
    out: List[str] = []
    for d in drugs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out

# ---------------- PUBLIC API FUNCTION ----------------
def run_validation_multi(drug_text: str, patient: dict, patient_id: str = "", doctor_id: str = "") -> str:
    """
    FINAL BEHAVIOR:
    ✅ CSV check (structured) — but DOES NOT stop
    ✅ Vector DB retrieval from ALL collections (DrugBank + real_drug + extras)
    ✅ Always call Gemini for final validation
    ✅ If Gemini returns SAFE/CAUTION/NOT SAFE => store into CSV + Chroma (llm_ddi_knowledge)
    """
    drugs = parse_drug_list(drug_text)

    if not drugs:
        return (
            "Overall_Status: INSUFFICIENT\n"
            "Key_Reason: No medicine names detected.\n"
            "Interactions:\n- (none)\n"
            "Per_Drug_Notes:\n- (none)\n"
            "Action: Enter at least one medicine name.\n"
        )

    # 1) CSV structured evidence (no early return)
    interactions = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            hit = find_interaction(drugs[i], drugs[j])
            if hit:
                interactions.append(hit)

    structured_hint = ""
    if interactions:
        structured_hint = "STRUCTURED_DDI_MATCHES:\n"
        for hit in interactions[:10]:
            structured_hint += f"- {hit.get('drug1')} + {hit.get('drug2')}: {hit.get('effect')}\n"

    # 2) Vector DB from ALL collections
    cols = get_all_chroma_collections()

    per_drug_blocks = []
    for d in drugs:
        ev = retrieve_drug_evidence_all(d, cols) if cols else ""
        if ev.strip():
            per_drug_blocks.append(f"### Drug evidence (ALL collections): {d}\n{ev}")
        else:
            per_drug_blocks.append(f"### Drug evidence (ALL collections): {d}\n(No evidence retrieved)")

    interaction_ev = retrieve_interaction_evidence_all(drugs, cols) if cols else ""

    combined_evidence = ("\n\n".join(per_drug_blocks) + "\n\n" + interaction_ev).strip()
    if structured_hint:
        combined_evidence += "\n\n" + structured_hint

    if not cols:
        combined_evidence += "\n\nNOTE: No Chroma collections found. Use general medical knowledge."
    elif not interaction_ev.strip():
        combined_evidence += "\n\nNOTE: Interaction evidence missing from vector DB; use general knowledge."

    combined_evidence = combined_evidence[:6000]

    # 3) Always call LLM
    answer = ask_gemini_multi(drugs, patient, combined_evidence)

    # 4) Self-learning store (only if usable)
    overall = normalize_status(parse_overall_status(answer))
    if overall in {"SAFE", "CAUTION", "NOT SAFE"}:
        pairs = parse_interaction_lines(answer)
        for a, b, status, effect_text in pairs:
            status = normalize_status(status)
            if status in {"SAFE", "CAUTION", "NOT SAFE"}:
                effect_final = f"{status}: {effect_text}"
                if not ddi_exists(a, b):
                    append_ddi_to_csv(a, b, effect_final, path=DDI_PATH)
                    add_ddi_to_chroma(a, b, effect_final, doctor_id=doctor_id, patient_id=patient_id)
        _reload_ddi()

    return answer


# ---------------- LOCAL TEST ----------------
def main():
    patient = {
        "age": 45,
        "allergies": ["ibuprofen"],
        "conditions": ["diabetes"],
        "current_meds": ["metformin"],
        "history_text": "",
    }

    incoming = input("Enter medicines (comma separated): ").strip()
    out = run_validation_multi(incoming, patient, patient_id="PAT_TEST", doctor_id="DOC_TEST")

    print("\n=== Validation Output ===\n")
    print(out)

    out_path = os.path.join(MODULE2_RAG_BASE, "data", "last_validation.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)

if __name__ == "__main__":
    main()