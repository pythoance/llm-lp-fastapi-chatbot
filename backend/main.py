import subprocess
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
import os
from langfuse.openai import AsyncAzureOpenAI
import uuid
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

app = FastAPI()

# Initialize clients
client = AsyncAzureOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    api_version="2024-06-01"
)

embedding_model = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version="2024-06-01",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
retriever = Chroma(
        persist_directory = 'local_movie_db', 
        embedding_function=embedding_model).as_retriever(
            search_kwargs={"k": 4})

def get_system_prompt(language: str) -> str:
    return f"""
    You are a helpful assistant for question on famous movies. 
    You will formulate all its answers in {language}.
    Base your answer only on pieces of context below. 
    If you don't know the answer, just say that you don't know. 
    Do not answer any question that are not related to movies."""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def start_streamlit():
    subprocess.Popen(["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=localhost"])

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_streamlit()
    yield

app = FastAPI(lifespan=lifespan)

# Redirect root to Streamlit UI
@app.get("/")
def redirect_to_streamlit():
    return RedirectResponse(url="http://localhost:8501")

# Generate Stream
async def stream_processor(response, context):
    async for chunk in response:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    yield f"\n\nCONTEXT: {context}" 


# API Endpoint
@app.post("/stream")
async def stream(request: Request):
    payload = await request.json()
    language = payload.get("language", "English")
    question = payload.get("question", "")
    temperature = payload.get("temperature", 0.5)
    session_id = payload.get("session_id", str(uuid.uuid4()))

    
    context = format_docs(retriever.invoke(question))

    messages = [
        {"role": "system", "content": get_system_prompt(language)},
        {"role": "user", "content": f"QUESTION: {question}"},
        {"role": "system", "content": f"CONTEXT: {context}"}
    ]

    print(messages)
    azure_open_ai_response =  await client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=temperature,
        stream=True,
        session_id=session_id
    )

    return StreamingResponse(stream_processor(azure_open_ai_response, context), media_type="text/event-stream")

# # New API endpoint to handle response stream
# @app.post("/api/stream")
# async def stream_response():
#     payload = await request.json()
#     language = payload.get("language", "English")
#     question = payload.get("question", "")
#     context = payload.get("context", "")
#     temperature = payload.get("temperature", 0.5)
#     session_id = payload.get("session_id", str(uuid.uuid4()))
    
#     messages = [
#         {"role": "system", "content": get_system_prompt(language)},
#         {"role": "user", "content": f"QUESTION: {question}"},
#         {"role": "system", "content": f"CONTEXT: {context}"}
#     ]
    
#     # Call the client with streaming enabled

    
#     async def event_generator():
#         # Yield each streamed chunk as text
#         for chunk in stream:
#             # Assuming each chunk is a dict with the key "message" that holds a "content" field
#             content = chunk.choices[0].message.content
#             yield f"data: {content}\n\n"
    
#     return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
