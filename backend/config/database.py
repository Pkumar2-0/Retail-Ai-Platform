from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "retail_ai")
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://127.0.0.1:27017")


def create_client(uri: str) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)


client = None

if MONGO_URI:
    try:
        client = create_client(MONGO_URI)
    except Exception as exc:
        print(f"Warning: failed to connect with MONGO_URI: {exc}")

if client is None:
    try:
        client = create_client(LOCAL_MONGO_URI)
        print(f"Using local MongoDB URI: {LOCAL_MONGO_URI}")
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to MongoDB. Set a valid MONGO_URI or start local MongoDB at mongodb://127.0.0.1:27017."
        ) from exc


database = client[DATABASE_NAME]