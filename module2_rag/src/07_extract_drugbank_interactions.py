import os
import csv
import xml.etree.ElementTree as ET

XML_PATH = "module2_rag/data/drugbank.xml"
OUT_CSV  = "module2_rag/data/drugbank_interactions.csv"

NS = {"db": "http://www.drugbank.ca"}

def clean(s: str) -> str:
    return (s or "").strip()

def main():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"Missing: {XML_PATH}")

    print("Parsing XML... this can take a bit.")
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    rows = []
    # DrugBank structure: <drug> ... <name> ... <drug-interactions> <drug-interaction> ...
    for drug in root.findall("db:drug", NS):
        drug_name_el = drug.find("db:name", NS)
        drug_name = clean(drug_name_el.text if drug_name_el is not None else "")

        inters = drug.find("db:drug-interactions", NS)
        if inters is None:
            continue

        for di in inters.findall("db:drug-interaction", NS):
            name_el = di.find("db:name", NS)
            desc_el = di.find("db:description", NS)
            other = clean(name_el.text if name_el is not None else "")
            desc  = clean(desc_el.text if desc_el is not None else "")

            if drug_name and other and desc:
                rows.append([drug_name, other, desc])

    # Remove duplicates (A,B,desc) simple
    uniq = []
    seen = set()
    for a,b,e in rows:
        key = (a.lower(), b.lower(), e)
        if key not in seen:
            seen.add(key)
            uniq.append([a,b,e])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["drug1","drug2","effect"])
        w.writerows(uniq)

    print(f"✅ Wrote {len(uniq)} interactions to: {OUT_CSV}")

if __name__ == "__main__":
    main()