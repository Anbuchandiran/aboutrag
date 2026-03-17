# module2_rag/src/02b_build_drugnames_chroma.py

from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]   # -> module2_rag/
DATA_DIR = BASE_DIR / "data"

FILES = [
    DATA_DIR / "drugbank_interactions.csv",
    DATA_DIR / "db_drug_interactions.csv",
    DATA_DIR / "drugbank_docs.csv",
    DATA_DIR / "real_drug_dataset.csv",
]

CHROMA_DIR = str(BASE_DIR / "db" / "chroma_drugnames")
COLLECTION = "drug_names"

NAME_COLS = ["name", "drug", "drug_name", "generic_name", "medicine", "medication", "brand"]


def add_from_csv(path: Path, names: set):
    if not path.exists():
        print("❌ Missing:", path)
        return

    df = pd.read_csv(path)
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # interaction style columns
    if "drug1" in cols_lower:
        names.update(df[cols_lower["drug1"]].dropna().astype(str).str.strip().str.lower().tolist())
    if "drug2" in cols_lower:
        names.update(df[cols_lower["drug2"]].dropna().astype(str).str.strip().str.lower().tolist())

    # name-like columns
    for key in NAME_COLS:
        if key in cols_lower:
            names.update(df[cols_lower[key]].dropna().astype(str).str.strip().str.lower().tolist())


def main():
    print("BASE_DIR =", BASE_DIR)
    print("DATA_DIR =", DATA_DIR)
    print("CHROMA_DIR =", CHROMA_DIR)

    names = set()

    for f in FILES:
        try:
            add_from_csv(f, names)
            print("✅ Loaded:", f.name)
        except Exception as e:
            print("❌ Failed:", f.name, "->", repr(e))

    # cleanup
    names = {n for n in names if n and n != "nan" and len(n) > 1}
    names = sorted(names)

    print("Total unique drug names:", len(names))
    if not names:
        print("❌ No names extracted. (Check your CSV columns)")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # rebuild fresh
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"drug_{i}" for i in range(len(names))]
    metas = [{"type": "drug_name"} for _ in names]

    B = 2000
    for i in range(0, len(ids), B):
        col.add(
            ids=ids[i:i + B],
            documents=names[i:i + B],
            metadatas=metas[i:i + B]
        )

    print("✅ Inserted drug names:", len(names))
    print("✅ Chroma path:", CHROMA_DIR)


if __name__ == "__main__":
    main()