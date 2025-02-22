import os
from typing import List
import uuid

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import yaml


class MovieVectorDB:
    def __init__(self, collection_name: str):
        self.config = self.load_config()
        self.chroma_client = chromadb.HttpClient(
            host=os.environ.get('CHROMA_HOST'), 
            port=os.environ.get('CHROMA_PORT')
            )
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
                api_base=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                api_type="azure",
                api_version=self.config['API_VERSION'],
                model_name=self.config['EMBEDDING_MODEL']
            )
        self.vector_db = self.chroma_client.get_collection(
            name = collection_name,
            embedding_function = self.embedding_function)


    @staticmethod
    def load_config():
        file_path = os.path.join('backend', 'config', 'llm.yaml')
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    
    def create_collection(self, 
                          collection_name: str, 
                          documents: List[str]):
        
        collection = self.chroma_client.create_collection(
            name = collection_name,
            embedding_function = self.embedding_function)

        ids = [str(uuid.uuid4()) for _ in documents]
        
        collection.add(
            ids=ids,
            documents=documents)
        
        return f"Created {collection_name} collection successfully!"
    
    @staticmethod
    def format_docs(docs: List[str]) -> str:
        return "\n\n".join(f"Source {i+1}: {doc}" for i, doc in enumerate(docs))

    async def query_collection(
            self, 
            question: str, 
            n_results: int = 3):
        
        query_result = self.vector_db.query(
            query_texts=question,
            n_results=n_results
        )

        retrieved_context = self.format_docs(query_result['documents'][0])
        
        return retrieved_context