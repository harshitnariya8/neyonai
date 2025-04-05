import weaviate
import weaviate.classes as wvc
from openai import OpenAI
import time
from llm.openai_llm import OpenAIChatbot

class WeaviateClient:
    def __init__(self, weaviate_url, weaviate_api_key, openai_api_key, collection_name):
        """Initialize the Weaviate and OpenAI client."""
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=wvc.init.Auth.api_key(weaviate_api_key),
        )
        if not self.client.is_ready():
            raise ConnectionError("Weaviate client is not ready.")

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key

        # Ensure collection exists
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                self.collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )

        self.collection = self.client.collections.get(self.collection_name)

    def get_embeddings(self, text):
        """Get text embeddings using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def add_text(self, text):
        """Add text to the Weaviate collection."""
        vector = self.get_embeddings(text)
        obj = wvc.data.DataObject(
            properties={"text": text},
            vector=vector
        )
        self.collection.data.insert_many([obj])

    def query_text(self, prompt, query_text, limit=10):
        """Query the collection for similar text."""
        bot = OpenAIChatbot(api_key=self.openai_api_key)
        query_vector = self.get_embeddings(query_text)
        time.sleep(1)  # Ensure async indexing completes
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )
        chunks = []
        metadata_list = ""
        for obj in response.objects:
            metadata_list += obj.properties['text']
            chunks.append(obj.properties)
        answer = bot.get_response(
            system_message=f"you are a knowledge assistant your task is to give answer based on given context: {metadata_list} prompt: {prompt}",
            user_message=query_text
        )
        return {"response": answer, "chunks": response}