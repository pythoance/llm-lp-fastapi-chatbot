from pydantic import BaseModel
from typing import List

class ChatbotRequest(BaseModel):
    """
    Request model for chatbot interactions.
    
    Attributes:
        language (str): The language of the question.
        question (str): The question to be asked.
        session_id (str): The session identifier.
    """
    language: str
    question: str
    session_id: str

class CreateDBRequest(BaseModel):
    """
    Request model for creating a database entry.
    
    Attributes:
        collection_name (str): The name of the collection.
        documents (List[str]): A list of documents as strings to be added to the collection.
    """
    collection_name: str
    documents: List[str]