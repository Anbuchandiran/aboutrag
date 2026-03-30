import os
import re
import itertools
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

try:
    from google import genai
except Exception as e:
    raise RuntimeError(
        "google-genai is required. Install it with: pip install google-genai"
    ) from e


# ---------------- CONFIG ----------------
DDI_PATH = "module2_rag/data/drugbank_interactions.csv"
CHROMA_DIR = "module2_rag/db/chroma_drugbank"
COLLECTION = "drugbank_knowledge"
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYNONYMS = {
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
}

SEVERE_WORDS = [
    "contraindicat", "fatal", "life-threatening", "life threatening",
    "major", "severe", "serious", "hemorrhage", "bleeding", "toxicity",
    "arrhythmia", "anaphyl", "coma", "death", "avoid concomitant", "black box"
]
MODERATE_WORDS = [
    "monitor", "moderate", "caution", "adjust", "increase", "decrease",
    "risk", "warning", "impair", "reduce", "elevat", "careful",
    "may enhance", "may increase", "may decrease", "interaction"
]
MILD_WORDS = [
    "minor", "mild", "slight", "limited", "low risk", "unlikely", "minimal"
]


# ---------------- HELPERS ----------------
def norm(x: str) -> str:
    return str(x).strip().lower()


def canonicalize(drug: str) -> str:
    d = norm(drug)
    return SYNONYMS.get(d, d)


def parse_drugs(text: str) -> List[str]:
    parts = re.split(r"\s*(?:,|\+|/|;| and |\n)\s*", text.strip().lower())
    drugs = []
    seen = set()
    for part in parts:
        if not part:
            continue
        d = canonicalize(part)
        if d not in seen:
            seen.add(d)
            drugs.append(d)
    return drugs


def sorted_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([canonicalize(a), canonicalize(b)]))


def pair_text(a: str, b: str) -> str:
    x, y = sorted_pair(a, b)
    return f"{x} [SEP] {y}"


# ---------------- DATA ----------------
def load_ddi() -> pd.DataFrame:
    df = pd.read_csv(
        DDI_PATH,
        header=None,
        names=["drug1", "drug2", "effect"],
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df["drug1"] = df["drug1"].astype(str).map(canonicalize)
    df["drug2"] = df["drug2"].astype(str).map(canonicalize)
    df["effect"] = df["effect"].astype(str).str.strip()
    df = df[(df["drug1"] != "") & (df["drug2"] != "") & (df["effect"] != "")].copy()
    df["pair_text"] = df.apply(lambda r: pair_text(r["drug1"], r["drug2"]), axis=1)
    df["severity"] = df["effect"].map(severity_from_effect)
    return df


def severity_from_effect(effect: str) -> float:
    text = norm(effect)
    severe_hits = sum(1 for w in SEVERE_WORDS if w in text)
    moderate_hits = sum(1 for w in MODERATE_WORDS if w in text)
    mild_hits = sum(1 for w in MILD_WORDS if w in text)

    score = 0.45
    score += 0.18 * severe_hits
    score += 0.08 * moderate_hits
    score -= 0.10 * mild_hits

    if "contraindicat" in text or "fatal" in text or "death" in text:
        score = max(score, 0.92)
    elif "major" in text or "severe" in text:
        score = max(score, 0.80)
    elif "moderate" in text or "monitor" in text or "caution" in text:
        score = max(score, 0.58)
    elif "minor" in text or "mild" in text:
        score = min(score, 0.35)

    return float(np.clip(score, 0.05, 0.98))


def label_from_score(score: float) -> str:
    if score >= 0.67:
        return "NOT SAFE"
    if score >= 0.34:
        return "CAUTION"
    return "SAFE"


# ---------------- REGRESSION MODEL ----------------
class DDIRegressor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.model = Ridge(alpha=1.0)
        self.drug_stats: Dict[str, Dict[str, float]] = {}
        self.global_mean: float = 0.5

    def fit(self, df: pd.DataFrame) -> None:
        self.global_mean = float(df["severity"].mean()) if len(df) else 0.5

        stats = {}
        for drug in pd.unique(pd.concat([df["drug1"], df["drug2"]], ignore_index=True)):
            mask = (df["drug1"] == drug) | (df["drug2"] == drug)
            sev = df.loc[mask, "severity"]
            stats[drug] = {
                "mean": float(sev.mean()) if len(sev) else self.global_mean,
                "max": float(sev.max()) if len(sev) else self.global_mean,
                "count": float(len(sev)),
            }
        self.drug_stats = stats

        X_text = self.vectorizer.fit_transform(df["pair_text"])
        X_num = self._make_numeric_features(df["drug1"].tolist(), df["drug2"].tolist())
        X = hstack([X_text, X_num])
        y = df["severity"].to_numpy(dtype=float)
        self.model.fit(X, y)

    def _drug_feature(self, drug: str) -> Tuple[float, float, float]:
        s = self.drug_stats.get(drug)
        if not s:
            return self.global_mean, self.global_mean, 0.0
        return s["mean"], s["max"], s["count"]

    def _make_numeric_features(self, drug1_list: List[str], drug2_list: List[str]) -> csr_matrix:
        feats = []
        for a, b in zip(drug1_list, drug2_list):
            a_mean, a_max, a_count = self._drug_feature(a)
            b_mean, b_max, b_count = self._drug_feature(b)
            feats.append([
                a_mean, a_max, np.log1p(a_count),
                b_mean, b_max, np.log1p(b_count),
                abs(a_mean - b_mean),
                max(a_max, b_max),
                (a_mean + b_mean) / 2.0,
            ])
        return csr_matrix(np.asarray(feats, dtype=float))

    def predict_score(self, drug_a: str, drug_b: str) -> float:
        a, b = sorted_pair(drug_a, drug_b)
        X_text = self.vectorizer.transform([pair_text(a, b)])
        X_num = self._make_numeric_features([a], [b])
        X = hstack([X_text, X_num])
        pred = float(self.model.predict(X)[0])
        return float(np.clip(pred, 0.01, 0.99))


# ---------------- RETRIEVAL ----------------
def exact_csv_hit(df: pd.DataFrame, drug_a: str, drug_b: str) -> pd.DataFrame:
    a, b = sorted_pair(drug_a, drug_b)
    hit = df[((df["drug1"] == a) & (df["drug2"] == b)) | ((df["drug1"] == b) & (df["drug2"] == a))]
    return hit.head(3)


def related_csv_examples(df: pd.DataFrame, drug_a: str, drug_b: str, k: int = 5) -> List[Dict[str, str]]:
    a, b = sorted_pair(drug_a, drug_b)

    work = df.copy()
    work["overlap"] = (
        (work["drug1"].eq(a) | work["drug2"].eq(a)).astype(int) +
        (work["drug1"].eq(b) | work["drug2"].eq(b)).astype(int)
    )
    work = work[work["overlap"] > 0].copy()

    if work.empty:
        return []

    work["text_match"] = work["pair_text"].str.contains(a, regex=False).astype(int) + work["pair_text"].str.contains(b, regex=False).astype(int)
    work["rank_score"] = work["overlap"] * 10 + work["text_match"] + work["severity"]
    work = work.sort_values(["rank_score", "severity"], ascending=False)

    rows = []
    for _, r in work.head(k).iterrows():
        rows.append({
            "drug1": r["drug1"],
            "drug2": r["drug2"],
            "effect": r["effect"],
            "severity": round(float(r["severity"]), 3),
        })
    return rows


def chroma_evidence_for_pair(drug_a: str, drug_b: str, k: int = 4) -> str:
    if chromadb is None or embedding_functions is None:
        return ""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        col = client.get_collection(name=COLLECTION, embedding_function=embed_fn)
        query = f"{drug_a} {drug_b} interaction contraindication warning toxicity mechanism"
        res = col.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\n".join(docs[:k])[:4000]
    except Exception:
        return ""


# ---------------- LLM ----------------
def get_llm_client():
    if not GEMINI_KEY:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment variables."
        )
    return genai.Client(api_key=GEMINI_KEY)


def build_pair_prompt(drug_a: str, drug_b: str, score: float, label: str,
                      exact_rows: List[Dict[str, str]], related_rows: List[Dict[str, str]],
                      rag_text: str) -> str:
    exact_text = "\n".join(
        f"- {r['drug1']} + {r['drug2']}: {r['effect']}"
        for r in exact_rows
    ) or "- No exact pair found in CSV"

    related_text = "\n".join(
        f"- {r['drug1']} + {r['drug2']} | severity={r['severity']}: {r['effect']}"
        for r in related_rows
    ) or "- No related CSV examples found"

    rag_block = rag_text if rag_text.strip() else "No Chroma evidence found."

    return f"""
You are a clinical writing assistant for drug interaction research.

Task:
Generate a NEW interaction description for this drug pair.
Do not copy any single evidence line exactly. Write a fresh grounded summary.
Do not mention that you are an AI.
Do not invent certainty beyond the evidence.
Keep it concise and medically styled.

Drug pair: {drug_a} + {drug_b}
Predicted regression severity score: {score:.3f}
Predicted risk label: {label}

Exact CSV evidence:
{exact_text}

Related CSV examples:
{related_text}

Retrieved RAG evidence:
{rag_block}

Return ONLY this format:

Predicted_Severity_Score: <number>
Risk_Level: SAFE / CAUTION / NOT SAFE
Generated_Description: <4-6 line grounded description>
Clinical_Advice: <1-2 lines>
""".strip()


def generate_pair_description(client, drug_a: str, drug_b: str, score: float, label: str,
                              exact_rows: List[Dict[str, str]], related_rows: List[Dict[str, str]],
                              rag_text: str) -> str:
    prompt = build_pair_prompt(drug_a, drug_b, score, label, exact_rows, related_rows, rag_text)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()


def build_overall_prompt(drugs: List[str], pair_outputs: List[Dict[str, str]]) -> str:
    details = "\n\n".join(
        f"Pair: {p['pair']}\n{p['text']}" for p in pair_outputs
    )
    return f"""
You are summarizing multi-drug interaction findings.

Input drugs: {", ".join(drugs)}

Pairwise findings:
{details}

Return ONLY this format:
Overall_Risk: SAFE / CAUTION / NOT SAFE
Overall_Summary: <4-6 lines>
Priority_Pairs:
- <pair>: <short reason>
General_Advice: <2 lines>
""".strip()


def generate_overall_summary(client, drugs: List[str], pair_outputs: List[Dict[str, str]]) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_overall_prompt(drugs, pair_outputs),
    )
    return (response.text or "").strip()


# ---------------- MAIN FLOW ----------------
def process_input(drug_text: str) -> str:
    drugs = parse_drugs(drug_text)
    if len(drugs) < 2:
        return "Enter at least two drug names separated by comma or +"

    df = load_ddi()
    reg = DDIRegressor()
    reg.fit(df)
    client = get_llm_client()

    outputs = []
    lines = []
    lines.append("=" * 90)
    lines.append("DRUG INTERACTION DESCRIPTION GENERATOR (REGRESSION + LLM)")
    lines.append("=" * 90)
    lines.append(f"Input drugs: {', '.join(drugs)}")
    lines.append("")

    for drug_a, drug_b in itertools.combinations(drugs, 2):
        score = reg.predict_score(drug_a, drug_b)
        label = label_from_score(score)

        exact_hit_df = exact_csv_hit(df, drug_a, drug_b)
        exact_rows = exact_hit_df[["drug1", "drug2", "effect"]].to_dict("records")
        related_rows = related_csv_examples(df, drug_a, drug_b, k=5)
        rag_text = chroma_evidence_for_pair(drug_a, drug_b, k=4)

        try:
            llm_text = generate_pair_description(
                client=client,
                drug_a=drug_a,
                drug_b=drug_b,
                score=score,
                label=label,
                exact_rows=exact_rows,
                related_rows=related_rows,
                rag_text=rag_text,
            )
        except Exception as e:
            llm_text = (
                f"Predicted_Severity_Score: {score:.3f}\n"
                f"Risk_Level: {label}\n"
                f"Generated_Description: LLM generation failed ({type(e).__name__}).\n"
                f"Clinical_Advice: Check API key, model access, quota, and network."
            )

        outputs.append({"pair": f"{drug_a} + {drug_b}", "text": llm_text, "score": score})

        lines.append(f"--- Pair: {drug_a} + {drug_b} ---")
        lines.append(llm_text)
        lines.append("")

    if len(outputs) > 1:
        try:
            overall = generate_overall_summary(client, drugs, outputs)
        except Exception as e:
            overall = f"Overall summary generation failed ({type(e).__name__})."
        lines.append("=" * 90)
        lines.append("OVERALL MULTI-DRUG SUMMARY")
        lines.append("=" * 90)
        lines.append(overall)
        lines.append("")

    return "\n".join(lines)


def main():
    print("Enter 2 or more drug names")
    print("Example: aspirin + warfarin")
    print("Example: aspirin, warfarin, ibuprofen")
    text = input("\nDrugs: ").strip()
    result = process_input(text)
    print("\n" + result)


if __name__ == "__main__":
    main()
