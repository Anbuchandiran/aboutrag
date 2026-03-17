import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

XML_PATH = "data/drugbank.xml"
OUT_CSV  = "data/drugbank_docs.csv"

NS = "{http://www.drugbank.ca}"

def text_or_empty(node):
    return (node.text or "").strip() if node is not None else ""

def collect_text_list(parent, tag):
    if parent is None:
        return []
    return [text_or_empty(x) for x in parent.findall(f"{NS}{tag}") if text_or_empty(x)]

def main():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    rows = []
    drugs = root.findall(f"{NS}drug")

    for drug in tqdm(drugs, desc="Parsing drugs"):
        drug_id = text_or_empty(drug.find(f"{NS}drugbank-id[@primary='true']"))
        name = text_or_empty(drug.find(f"{NS}name"))
        description = text_or_empty(drug.find(f"{NS}description"))
        indication  = text_or_empty(drug.find(f"{NS}indication"))
        mechanism   = text_or_empty(drug.find(f"{NS}mechanism-of-action"))
        toxicity    = text_or_empty(drug.find(f"{NS}toxicity"))

        # Drug-drug interactions (names + description)
        interactions_parent = drug.find(f"{NS}drug-interactions")
        interaction_texts = []
        if interactions_parent is not None:
            for ddi in interactions_parent.findall(f"{NS}drug-interaction"):
                iname = text_or_empty(ddi.find(f"{NS}name"))
                idesc = text_or_empty(ddi.find(f"{NS}description"))
                if iname or idesc:
                    interaction_texts.append(f"{iname}: {idesc}".strip(": "))

        # Build one “knowledge document” text per drug
        doc = (
            f"Drug: {name}\n"
            f"DrugBankID: {drug_id}\n\n"
            f"Description: {description}\n\n"
            f"Indication: {indication}\n\n"
            f"Mechanism: {mechanism}\n\n"
            f"Toxicity: {toxicity}\n\n"
            f"Interactions:\n- " + "\n- ".join(interaction_texts[:50])  # limit to keep size sane
        )

        if name and doc.strip():
            rows.append({
                "drugbank_id": drug_id,
                "name": name.lower(),
                "doc_text": doc
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("Saved:", OUT_CSV, "rows:", len(df))

if __name__ == "__main__":
    main()