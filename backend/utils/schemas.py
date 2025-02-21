from pydantic import BaseModel

class ChatbotRequest(BaseModel):
    language: str
    question: str
    session_id: str