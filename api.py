import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from weaviate_client import WeaviateClient
from llm.openai_llm import OpenAIChatbot
from motor.motor_asyncio import AsyncIOMotorClient
import re
# Initialize FastAPI app
app = FastAPI()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI")  # Change this if needed
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set.")

mongo_client = AsyncIOMotorClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Request Models
class QueryRequest(BaseModel):
    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str
    OPENAI_API_KEY: str
    COLLECTION_NAME: str

class ContentRequest(BaseModel):
    rag_id: str
    content: list

class QueryTextRequest(BaseModel):
    rag_id: str
    query: str
    prompt: str


class TextRequest(BaseModel):
    text: str
    word_limit: int = 1000


def split_into_chunks(text: str, word_limit: int):
    words = text.split()
    chunks = [" ".join(words[i:i + word_limit]) for i in range(0, len(words), word_limit)]
    return chunks


@app.post("/split-text")
def split_text(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    chunks = split_into_chunks(request.text, 200)
    return {"formatted_text": chunks}

@app.post("/get_rag_id")
async def get_rag_id(request: QueryRequest):
    try:
        rag_id = str(uuid4())
        # The rag_id is generated with uuid4, so the client does not
        # need to pass it.
        # It does not make any sense to create WeaviateClient instance here
        # and then just discard it.
        # client = WeaviateClient(request.WEAVIATE_URL, request.WEAVIATE_API_KEY, request.OPENAI_API_KEY, request.COLLECTION_NAME)

        # Store in MongoDB
        await collection.insert_one({
            "rag_id": rag_id,
            "WEAVIATE_URL": request.WEAVIATE_URL,
            "WEAVIATE_API_KEY": request.WEAVIATE_API_KEY,
            "OPENAI_API_KEY": request.OPENAI_API_KEY,
            "COLLECTION_NAME": request.COLLECTION_NAME
        })

        return {"rag_id": rag_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating RAG client: {str(e)}")

@app.post("/add_rag")
async def add_rag(request: ContentRequest):
    try:
        # Retrieve client details from MongoDB
        client_data = await collection.find_one({"rag_id": request.rag_id})
        if not client_data:
            raise HTTPException(status_code=404, detail="RAG ID not found")

        # Create a new Weaviate client
        client = WeaviateClient(client_data["WEAVIATE_URL"], client_data["WEAVIATE_API_KEY"], client_data["OPENAI_API_KEY"], client_data["COLLECTION_NAME"])
        for con in request.content:
            client.add_text(con)

        return {"message": "Content added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding content: {str(e)}")

@app.post("/ask_rag")
async def ask_rag(request: QueryTextRequest):
    try:
        # Retrieve client details from MongoDB
        client_data = await collection.find_one({"rag_id": request.rag_id})
        if not client_data:
            raise HTTPException(status_code=404, detail="RAG ID not found")

        # Create a new Weaviate client
        client = WeaviateClient(client_data["WEAVIATE_URL"], client_data["WEAVIATE_API_KEY"], client_data["OPENAI_API_KEY"], client_data["COLLECTION_NAME"])
        response = client.query_text(request.prompt, request.query)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying RAG: {str(e)}")