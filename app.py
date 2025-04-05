from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from weaviate_client import WeaviateClient
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGO_URI")  # Change this if needed
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


class QueryRequest(BaseModel):
    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str
    OPENAI_API_KEY: str
    COLLECTION_NAME: str

class ContentRequest(BaseModel):
    rag_id: str
    content: str

class QueryTextRequest(BaseModel):
    rag_id: str
    query: str

app = FastAPI()
clients = {}


@app.post("/get_rag_id")
async def get_rag_id(request: QueryRequest):
    try:
        rag_id = str(uuid4())
        # client = WeaviateClient(request.WEAVIATE_URL, request.WEAVIATE_API_KEY, request.OPENAI_API_KEY, request.COLLECTION_NAME)
        print("rag done")
        await collection.insert_one({
            "rag_id": rag_id,
            "WEAVIATE_URL": request.WEAVIATE_URL,
            "WEAVIATE_API_KEY": request.WEAVIATE_API_KEY,
            "OPENAI_API_KEY": request.OPENAI_API_KEY,
            "COLLECTION_NAME": request.COLLECTION_NAME
        })
        print("insert done")
        return {"rag_id": rag_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating RAG client: {str(e)}")

@app.post("/add_rag")
async def add_rag(request: ContentRequest):
    try:
        client_data = await collection.find_one({"rag_id": request.rag_id})
        if not client_data:
            raise HTTPException(status_code=404, detail="RAG ID not found")
        client = WeaviateClient(client_data["WEAVIATE_URL"], client_data["WEAVIATE_API_KEY"],
                                client_data["OPENAI_API_KEY"], client_data["COLLECTION_NAME"])

        client.add_text(request.content)
        return {"rag_id": client_data["rag_id"], "message": "Content added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding content: {str(e)}")

@app.post("/ask_rag")
async def ask_rag(request: QueryTextRequest):
    try:
        client_data = await collection.find_one({"rag_id": request.rag_id})
        if not client_data:
            raise HTTPException(status_code=404, detail="RAG ID not found")
        client = WeaviateClient(client_data["WEAVIATE_URL"], client_data["WEAVIATE_API_KEY"],
                                client_data["OPENAI_API_KEY"], client_data["COLLECTION_NAME"])

        response = client.query_text("give answer", request.query)
        return {"rag_id": client_data["rag_id"], "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG: {str(e)}")
