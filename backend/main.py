from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from .utils.llm import LLM
from .utils.schemas import ChatbotRequest, CreateDBRequest
from .utils.vector_db import MovieVectorDB


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Context manager for the lifespan of the FastAPI application.
    Initializes the LLM and MovieVectorDB instances.
    """
    app.state.llm = LLM()
    ###TODO Needs to be seperated so that app starts even without existing db
    app.state.vector_db = MovieVectorDB(collection_name='langchain')
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """
    Redirects the root URL to the FastAPI documentation.
    """
    return RedirectResponse(url="/docs")


@app.post("/create_vector_db")
async def create_vector_db(
    create_db_request: CreateDBRequest,
    req: Request
) -> dict:
    """
    Endpoint to create a vector database collection.
    
    Args:
        create_db_request (CreateDBRequest): The request body containing collection name and documents.
        req (Request): The request object passing on the app state objects.
    
    Returns:
        dict: The result of the collection creation.
    """
    return app.state.vector_db.create_collection(
        collection_name=create_db_request.collection_name,
        documents=create_db_request.documents
    )

@app.post("/chatbot-response-stream")
async def stream(
    chatbot_request: ChatbotRequest,
    req: Request
) -> StreamingResponse:
    """
    Endpoint to stream chatbot responses.
    
    Args:
        chatbot_request (ChatbotRequest): The request body containing question, language, and session ID.
        req (Request): The request object passing on the app state objects.
    
    Returns:
        StreamingResponse: The streaming response with chatbot answers.
    """

    context = await req.app.state.vector_db.query_collection(
        question=chatbot_request.question
    )
    
    streaming_response = await req.app.state.llm.get_chatbot_response(
        question=chatbot_request.question,
        context=context,
        response_language=chatbot_request.language,
        session_id=chatbot_request.session_id
    )

    return StreamingResponse(streaming_response, media_type="text/event-stream")
