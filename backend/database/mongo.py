from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

import certifi
import os

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

client = AsyncIOMotorClient(
    MONGO_URL,
    tls=True,
    tlsCAFile=certifi.where()
)

database = client["retail_ai"]