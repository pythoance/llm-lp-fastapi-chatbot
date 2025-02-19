import subprocess
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
import os
from langfuse.openai import AsyncAzureOpenAI
import uuid
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


app = FastAPI()

# Initialize clients
client = AsyncAzureOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    api_version="2024-06-01"
)

# embedding_model = AzureOpenAIEmbeddings(
#         model="text-embedding-ada-002",
#         api_key=os.getenv('AZURE_OPENAI_API_KEY'),
#         api_version="2024-06-01",
#         azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
#     )
# retriever = Chroma(
#         persist_directory = 'local_movie_db', 
#         embedding_function=embedding_model).as_retriever(
#             search_kwargs={"k": 2})

def get_system_prompt(language: str) -> str:
    return f"""
    You are a helpful assistant for question on famous movies.""" 
    # You will formulate all its answers in {language}.
    # Base your answer only on sources of context below. 
    # If you don't know the answer, just say that you don't know. 
    # Do not answer any question that are not related to movies."""


def format_docs(docs):
    return "\n\n".join(f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs))

def start_streamlit():
    global streamlit_process
    streamlit_process = subprocess.Popen(["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=localhost"])

from contextlib import asynccontextmanager

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

    
    # context = format_docs(retriever.invoke(question))
    context = 'test context'

    messages = [
        {"role": "system", "content": get_system_prompt(language)},
        {"role": "user", "content": f"QUESTION: {question}"},
        {"role": "system", "content": f"CONTEXT: {context}"}
    ]

    azure_open_ai_response =  await client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=temperature,
        stream=True,
        session_id=session_id
    )

    return StreamingResponse(stream_processor(azure_open_ai_response, context), media_type="text/event-stream")



if __name__ == "__main__":
    uvicorn.run(app, port=8000)
