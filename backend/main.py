from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse
import os
from langfuse.openai import AsyncAzureOpenAI
import uuid
import chromadb
from contextlib import asynccontextmanager
from .utils.schemas import ChatbotRequest



app = FastAPI()


def get_system_prompt(language: str) -> str:
    return f"""
    You are a helpful assistant for question on famous movies.
    You will formulate all your answers in {language}.
    Base your answer only on sources of context below. 
    If you don't know the answer, just say that you don't know. 
    Do not answer any question that are not related to movies."""


def format_docs(docs):
    return "\n\n".join(f"Source {i+1}: {doc}" for i, doc in enumerate(docs))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.azure_oai_client = AsyncAzureOpenAI(
        azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
        api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
        api_version="2024-06-01"
        )
    app.state.chroma_client = chromadb.HttpClient(
        host=os.environ.get('CHROMA_HOST'), 
        port=os.environ.get('CHROMA_PORT'))
    yield

app = FastAPI(
    lifespan=lifespan
    )

# Redirect root to Streamlit UI
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


async def stream_processor(response, context):
    async for chunk in response:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:

                yield delta.content
    yield f"\n\nCONTEXT: {context}" 



@app.post("/chatbot-response-stream")
async def stream(
    chatbot_request: ChatbotRequest,
    req: Request
    ):
    
    embedding = await req.app.state.azure_oai_client.embeddings.create(
        input=chatbot_request.question,
        model="text-embedding-ada-002")
        
    vector_db = req.app.state.chroma_client.get_collection('test_movie_collection')
    query_result = vector_db.query(
        query_embeddings=embedding.data[0].embedding,
        n_results = 3
        )
    docs = query_result['documents'][0]
    context = format_docs(docs)
    
    messages = [
        {"role": "system", "content": get_system_prompt(chatbot_request.language)},
        {"role": "user", "content": f"QUESTION: {chatbot_request.question}"},
        {"role": "system", "content": f"CONTEXT: {context}"}
    ]

    azure_open_ai_response =  await  req.app.state.azure_oai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=chatbot_request.temperature,
        stream=True,
        session_id=chatbot_request.session_id
    )

    return StreamingResponse(stream_processor(azure_open_ai_response, context), media_type="text/event-stream")




