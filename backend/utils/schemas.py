from pydantic import BaseModel

class ChatbotRequest(BaseModel):
    language: str
    question: str
    temperature: float
    session_id: str