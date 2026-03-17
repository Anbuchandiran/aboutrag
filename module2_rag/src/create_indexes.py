import os
import asyncio
import certifi
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv("module2_rag/.env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "haithunamma")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing in module2_rag/.env")

async def main():
    client = AsyncIOMotorClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=8000
    )
    db = client[MONGO_DB]

    solved_cases = db["solved_cases"]
    visits = db["visits"]

    # visits history
    await visits.create_index([("patient_id", 1), ("created_at", -1)])
    await visits.create_index([("doctor_id", 1), ("created_at", -1)])

    # ✅ exact match: patient_id + normalized_key
    await solved_cases.create_index([("patient_id", 1), ("normalized_key", 1)])

    print("✅ Indexes created successfully")

asyncio.run(main())