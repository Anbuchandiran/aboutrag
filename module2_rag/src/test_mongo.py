import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("module2_rag/.env")

uri = os.getenv("MONGO_URI")
print("URI starts with:", uri[:20])

client = MongoClient(uri, serverSelectionTimeoutMS=8000)
print("Ping:", client.admin.command("ping"))
print("OK connected!")