from pydantic import BaseModel
from typing import List

class ChatbotRequest(BaseModel):
    language: str
    question: str
    session_id: str

class CreateDBRequest(BaseModel):
    collection_name: str
    documents: List[str]