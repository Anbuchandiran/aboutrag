import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = "data/drugbank_docs.csv"
CHROMA_DIR = "db/chroma_drugbank"
COLLECTION = "drugbank_knowledge"

# Simple chunking (character-based)
def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += (chunk_size - overlap)
    return chunks

def main():
    df = pd.read_csv(CSV_PATH)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    col = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids, docs, metas = [], [], []
    for idx, row in df.iterrows():
        name = str(row["name"])
        drug_id = str(row["drugbank_id"])
        text = str(row["doc_text"])

        chunks = chunk_text(text)
        for j, ch in enumerate(chunks):
            ids.append(f"{drug_id}_{j}")
            docs.append(ch)
            metas.append({"drug": name, "drugbank_id": drug_id, "chunk": j})

    # Add to chroma (batching is good for big data)
    B = 2000
    for i in range(0, len(ids), B):
        col.add(
            ids=ids[i:i+B],
            documents=docs[i:i+B],
            metadatas=metas[i:i+B]
        )

    print("Inserted chunks:", len(ids))
    print("Chroma path:", CHROMA_DIR)

if __name__ == "__main__":
    main()