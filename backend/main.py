import subprocess
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse
import os
from langfuse.openai import AsyncAzureOpenAI
import uuid
import chromadb
from contextlib import asynccontextmanager



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

def start_streamlit():
    global streamlit_process

    
    backend_dir = os.path.dirname(__file__)  # Directory of main.py
    project_root = os.path.dirname(backend_dir)  # Go up one level
    frontend_path = os.path.join(project_root, "frontend")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"],
        cwd=frontend_path
    )
        


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_streamlit()
    yield
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()

app = FastAPI(lifespan=lifespan)

# Redirect root to Streamlit UI
@app.get("/")
async def serve_streamlit():
    streamlit_url = os.environ.get("STREAMLIT_URL", "http://localhost:8501")  # Default to localhost
    content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streamlit App</title>
    </head>
    <body>
        <iframe src="{streamlit_url}" width="100%" height="800" frameborder="0"></iframe>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

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

    azure_oai_client = AsyncAzureOpenAI(
        azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
        api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
        api_version="2024-06-01"
        )
    

    embedding = await azure_oai_client.embeddings.create(
        input=question,
        model="text-embedding-ada-002")
    
    
    chroma_client = chromadb.HttpClient(host='98.71.147.60', port=8000)
    vector_db = chroma_client.get_collection('test_movie_collection')
    query_result = vector_db.query(query_embeddings=embedding.data[0].embedding)
    docs = query_result['documents'][0]
    context = format_docs(docs)
    
    messages = [
        {"role": "system", "content": get_system_prompt(language)},
        {"role": "user", "content": f"QUESTION: {question}"},
        {"role": "system", "content": f"CONTEXT: {context}"}
    ]

    azure_open_ai_response =  await azure_oai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=temperature,
        stream=True,
        session_id=session_id
    )

    return StreamingResponse(stream_processor(azure_open_ai_response, context), media_type="text/event-stream")




