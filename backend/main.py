from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from .utils.llm import LLM
from .utils.schemas import ChatbotRequest, CreateDBRequest
from .utils.vector_db import MovieVectorDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLM()
    app.state.vector_db = MovieVectorDB(
        collection_name='langchain')
    yield

app = FastAPI(
    lifespan=lifespan
    )

# Redirect root to FASTAPI docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post("/create_vector_db")
async def create_vector_db(
    create_db_request: CreateDBRequest,
    req: Request):
    return app.state.vector_db.create_collection(
        collection_name=create_db_request.collection_name,
        documents=create_db_request.documents
    )

@app.post("/chatbot-response-stream")
async def stream(
    chatbot_request: ChatbotRequest,
    req: Request
    ):
    
    context = await req.app.state.vector_db.query_collection(
        question = chatbot_request.question
    )
    
    streaming_response = await req.app.state.llm.get_chatbot_response(
        question=chatbot_request.question,
        context=context,
        response_language=chatbot_request.language,
        session_id=chatbot_request.session_id
    )

    return StreamingResponse(streaming_response, media_type="text/event-stream")




