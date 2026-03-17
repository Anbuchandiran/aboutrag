import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

import certifi
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load env
load_dotenv("module2_rag/.env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "haithunamma")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is missing in module2_rag/.env")

client = AsyncIOMotorClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=8000
)

db = client[MONGO_DB]

patients = db["patients"]
doctors = db["doctors"]
visits = db["visits"]
solved_cases = db["solved_cases"]


def now() -> str:
    return datetime.utcnow().isoformat()


# -------------------- NORMALIZE DRUG QUERY --------------------
def normalize_drug_query(q: str) -> str:
    """
    Turns: 'Warfarin + aspirin and Ibuprofen'
    Into : 'aspirin|ibuprofen|warfarin' (sorted unique)
    """
    q = (q or "").lower()
    q = re.sub(r"[\+\.,;/]", " ", q)
    q = re.sub(r"\band\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    parts = [p.strip() for p in q.split(" ") if p.strip()]
    parts = [p for p in parts if len(p) > 2]
    parts = sorted(set(parts))
    return "|".join(parts)


# -------------------- PATIENTS --------------------
async def upsert_patient(
    patient_id: str,
    name: str,
    age: int,
    gender: str,
    phone: str = "",
    chronic_conditions: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    notes: str = "",
):
    update_set = {
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "chronic_conditions": chronic_conditions or [],
        "allergies": allergies or [],
        "notes": notes,
        "updated_at": now(),
    }

    await patients.update_one(
        {"_id": patient_id},
        {"$set": update_set, "$setOnInsert": {"created_at": now()}},
        upsert=True
    )


# -------------------- DOCTORS --------------------
async def upsert_doctor(
    doctor_id: str,
    name: str,
    department: str = "",
    phone: str = "",
    email: str = "",
):
    update_set = {
        "name": name,
        "department": department,
        "phone": phone,
        "email": email,
        "updated_at": now(),
    }

    await doctors.update_one(
        {"_id": doctor_id},
        {"$set": update_set, "$setOnInsert": {"created_at": now()}},
        upsert=True
    )


# -------------------- VISITS --------------------
async def add_visit(
    patient_id: str,
    doctor_id: str,
    complaint: str,
    diagnosis: str = "",
    prescription_text: str = "",
    extracted_entities: Optional[Dict[str, Any]] = None,
    notes: str = "",
):
    await visits.insert_one({
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "complaint": complaint,
        "diagnosis": diagnosis,
        "prescription_text": prescription_text,
        "extracted_entities": extracted_entities or {},
        "notes": notes,
        "created_at": now(),
    })


async def get_patient_history(patient_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    cursor = visits.find({"patient_id": patient_id}).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    for d in docs:
        d["_id"] = str(d["_id"])
        doc = await doctors.find_one({"_id": d.get("doctor_id")})
        if doc:
            d["doctor"] = {
                "doctor_id": doc["_id"],
                "name": doc.get("name", ""),
                "department": doc.get("department", "")
            }
    return docs


async def get_doctor_history(doctor_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    cursor = visits.find({"doctor_id": doctor_id}).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    for d in docs:
        d["_id"] = str(d["_id"])
        p = await patients.find_one({"_id": d.get("patient_id")})
        if p:
            d["patient"] = {
                "patient_id": p["_id"],
                "name": p.get("name", ""),
                "age": p.get("age", None),
                "gender": p.get("gender", "")
            }
    return docs


# -------------------- SOLVED CASES: EXACT MATCH ONLY --------------------
async def mongo_first_search(patient_id: str, normalized_key: str, limit: int = 3):
    cursor = solved_cases.find(
        {
            "patient_id": patient_id,
            "normalized_key": normalized_key,
            "final_answer": {"$not": re.compile("Overall_Status:\\s*INSUFFICIENT", re.I)},
        },
        {
            "final_answer": 1,
            "query_text": 1,
            "created_at": 1,
            "confidence": 1,
            "normalized_key": 1,
            "source": 1,
        },
    ).sort("created_at", -1).limit(limit)

    results = []
    async for r in cursor:
        r["score"] = 1.0
        results.append(r)
    return results


async def store_solved_case(
    patient_id: str,
    doctor_id: str,
    query: str,
    normalized_key: str,
    answer: str,
    confidence: float = 0.7,
    source: str = "rag",
):
    doc = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "query_text": query,
        "normalized_key": normalized_key,
        "final_answer": answer,
        "confidence": confidence,
        "source": source,
        "created_at": now(),
    }
    await solved_cases.insert_one(doc)