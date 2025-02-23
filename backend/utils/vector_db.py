import os
from typing import List, Dict, Any
import uuid

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import yaml


class MovieVectorDB:
    def __init__(self, collection_name: str):
        """
        Initialize the MovieVectorDB with a collection name.

        Args:
            collection_name (str): The name of the collection.
        """
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
        try:
            self.vector_db = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            raise Warning("Collection does not exist!")

    @staticmethod
    def load_config() -> Dict[str, Any]:
        """
        Load the configuration from a YAML file.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        file_path = os.path.join('backend', 'config', 'llm.yaml')
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def create_collection(self, collection_name: str, documents: List[str]) -> str:
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): The name of the collection.
            documents (List[str]): The list of documents to add to the collection.

        Returns:
            str: Success message.
        """
        collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        ids = [str(uuid.uuid4()) for _ in documents]

        collection.add(
            ids=ids,
            documents=documents
        )

        return f"Created {collection_name} collection successfully!"

    @staticmethod
    def format_docs(docs: List[str]) -> str:
        """
        Format a list of documents into a single string.

        Args:
            docs (List[str]): The list of documents.

        Returns:
            str: The formatted string.
        """
        return "\n\n".join(f"Source {i+1}: {doc}" for i, doc in enumerate(docs))

    async def query_collection(self, question: str, n_results: int = 3) -> str:
        """
        Query the collection with a question and retrieve the context.

        Args:
            question (str): The question to query.
            n_results (int, optional): The number of results to retrieve. Defaults to 3.

        Returns:
            str: The retrieved context.
        """
        query_result = self.vector_db.query(
            query_texts=question,
            n_results=n_results
        )

        retrieved_context = self.format_docs(query_result['documents'][0])

        return retrieved_context