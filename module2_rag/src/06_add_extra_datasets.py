import os
import json
import math
import hashlib
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions


# ---------------- CONFIG ----------------
CHROMA_DIR = "module2_rag/db/chroma_drugbank"
COLLECTION = "drugbank_knowledge"

DATA_DIR = "module2_rag/data"

# your extra dataset files (add/remove as needed)
FILES = [
    os.path.join(DATA_DIR, "real_drug_dataset.csv"),
    os.path.join(DATA_DIR, "side_effects.csv"),
    os.path.join(DATA_DIR, "drugs_side_effects_drugs_com.csv"),
    os.path.join(DATA_DIR, "drugs_cleaned_dataset.xls"),
    os.path.join(DATA_DIR, "DDI 2.0.json"),
    os.path.join(DATA_DIR, "db_drug_interactions.csv"),
]

BATCH_SIZE = 4000   # must be <= 5461 (your max). keep 4000 safe.


# ---------------- HELPERS ----------------
def safe_id(text: str) -> str:
    """Stable unique ID from content (avoids duplicates)."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def row_to_doc(row: dict, source: str) -> str:
    """Convert one row to a single text doc for embedding."""
    parts = [f"[SOURCE] {source}"]
    for k, v in row.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        parts.append(f"{k}: {s}")
    return "\n".join(parts)


def load_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    df = df.fillna("")
    docs = []
    for _, r in df.iterrows():
        doc = row_to_doc(r.to_dict(), os.path.basename(path))
        if doc.strip():
            docs.append(doc)
    return docs


def load_excel(path: str) -> list[str]:
    df = pd.read_excel(path)
    df = df.fillna("")
    docs = []
    for _, r in df.iterrows():
        doc = row_to_doc(r.to_dict(), os.path.basename(path))
        if doc.strip():
            docs.append(doc)
    return docs


def load_json(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    # if JSON is list of objects
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                doc = row_to_doc(obj, os.path.basename(path))
                if doc.strip():
                    docs.append(doc)

    # if JSON is dict with nested lists
    elif isinstance(data, dict):
        # flatten: store each key-group as a document
        for k, v in data.items():
            if isinstance(v, list):
                for obj in v:
                    if isinstance(obj, dict):
                        obj["_group"] = k
                        doc = row_to_doc(obj, os.path.basename(path))
                        if doc.strip():
                            docs.append(doc)
            elif isinstance(v, dict):
                v["_group"] = k
                doc = row_to_doc(v, os.path.basename(path))
                if doc.strip():
                    docs.append(doc)
    return docs


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------- MAIN ----------------
def main():
    # Connect Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # get or create collection
    try:
        col = client.get_collection(name=COLLECTION, embedding_function=embed_fn)
    except Exception:
        col = client.create_collection(name=COLLECTION, embedding_function=embed_fn)

    all_docs = []

    # Load each dataset file
    for file in FILES:
        if not os.path.exists(file):
            print(f"⚠️ Skipping missing file: {file}")
            continue

        # empty file protection
        if os.path.getsize(file) == 0:
            print(f"⚠️ Skipping empty file: {file}")
            continue

        print(f"📥 Loading: {file}")

        try:
            if file.lower().endswith(".csv"):
                docs = load_csv(file)
            elif file.lower().endswith(".xls") or file.lower().endswith(".xlsx"):
                docs = load_excel(file)
            elif file.lower().endswith(".json"):
                docs = load_json(file)
            else:
                print(f"⚠️ Unsupported format: {file}")
                continue

            print(f"✅ Loaded {len(docs)} docs from {os.path.basename(file)}")
            all_docs.extend(docs)

        except Exception as e:
            print(f"❌ Failed to load {file}: {repr(e)}")

    if not all_docs:
        print("❌ No documents found to add.")
        return

    # Create IDs
    ids = [safe_id(d) for d in all_docs]

    print(f"\n🚀 Total docs to add: {len(all_docs)}")
    total_batches = math.ceil(len(all_docs) / BATCH_SIZE)
    print(f"📦 Batch size: {BATCH_SIZE} | Total batches: {total_batches}\n")

    # Add in batches
    added = 0
    for idx, (doc_batch, id_batch) in enumerate(zip(chunk_list(all_docs, BATCH_SIZE), chunk_list(ids, BATCH_SIZE)), start=1):
        print(f"➡️ Adding batch {idx}/{total_batches} (docs={len(doc_batch)}) ...")
        col.add(documents=doc_batch, ids=id_batch)
        added += len(doc_batch)

    print(f"\n✅ Done! Added {added} documents into Chroma collection: {COLLECTION}")
    print("Now run your API and test /ask again.")


if __name__ == "__main__":
    main()