from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from litellm import acompletion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class RequestData(BaseModel):
    model_provider: str
    model: str
    content: str

async def get_response(model_provider: str, model: str, content: str):
    messages = [{"content": content, "role": "user"}]
    response = await acompletion(model=f"{model_provider}/{model}", messages=messages)
    return response

@app.post("/generate-response/")
async def generate_response(data: RequestData):
    response = await get_response(data.model_provider, data.model, data.content)
    return {"response": response}

