import os, asyncio
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv("module2_rag/.env")  # run from project root

async def main():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB")]

    patients = db["patients"]
    doctors = db["doctors"]

    await patients.update_one(
        {"_id": "PAT001"},
        {"$set": {"name": "Ravi", "age": 45, "gender": "M", "chronic_conditions": ["diabetes"]}},
        upsert=True
    )

    await doctors.update_one(
        {"_id": "DOC001"},
        {"$set": {"name": "Dr Meena", "department": "General"}},
        upsert=True
    )

    print("✅ Inserted patient + doctor")
    print("Collections:", await db.list_collection_names())

asyncio.run(main())