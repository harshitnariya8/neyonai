import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from llm.openai_llm import OpenAIChatbot
import time

class QdrantClientWrapper:
    def __init__(self, qdrant_url, qdrant_api_key, openai_api_key, collection_name, vector_size):
        """Initialize the Qdrant and OpenAI clients."""
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key

        # Ensure collection exists
        if not self.client.get_collection(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def get_embeddings(self, text):
        """Get text embeddings using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def add_text(self, text, id):
        """Add text to the Qdrant collection."""
        vector = self.get_embeddings(text)
        point = PointStruct(id=id, vector=vector, payload={"text": text})
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def query_text(self, prompt, query_text, limit=10):
        """Query the collection for similar text."""
        bot = OpenAIChatbot(api_key=self.openai_api_key)
        query_vector = self.get_embeddings(query_text)
        time.sleep(1)  # Ensure async indexing completes
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
        )
        chunks = []
        metadata_list = ""
        for result in search_result:
            metadata_list += result.payload['text']
            chunks.append(result.payload)
        answer = bot.get_response(
            system_message=f"You are a knowledge assistant. Your task is to answer based on the given context: {metadata_list} Prompt: {prompt}",
            user_message=query_text
        )
        return {"response": answer, "chunks": chunks}
