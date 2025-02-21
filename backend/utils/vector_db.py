import os
from typing import List

import chromadb
from langfuse.openai import AsyncAzureOpenAI

class MovieVectorDB:
    def __init__(self, collection_name: str):
        
        self.chroma_client = chromadb.HttpClient(
            host=os.environ.get('CHROMA_HOST'), 
            port=os.environ.get('CHROMA_PORT')
            )
        self.vector_db = self.get_collection(collection_name)


    def get_collection(self, collection_name: str):
        return self.chroma_client.get_collection(collection_name)
    
    @staticmethod
    def format_docs(docs: List[str]) -> str:
        return "\n\n".join(f"Source {i+1}: {doc}" for i, doc in enumerate(docs))

    async def query_collection(
            self, 
            text: str, 
            azure_oai_client: AsyncAzureOpenAI, 
            n_results: int = 3):
        
        embedding = await azure_oai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        query_result = self.vector_db.query(
            query_embeddings=embedding.data[0].embedding,
            n_results=n_results
        )

        retrieved_context = self.format_docs(query_result['documents'][0])
        
        return retrieved_context