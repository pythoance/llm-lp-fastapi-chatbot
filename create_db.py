from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb
from openai import AzureOpenAI
import os

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-06-01",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
)

retriever = WikipediaRetriever(
    top_k_results = 1,
    lang = 'en',
)

movie_docs =[]
movies = ["Inception", "Django Unchained", "Shutter Island", "The Dark Knight"]

for movie in movies:
  movie_docs += retriever.invoke(movie)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

movie_docs_split = text_splitter.split_documents(movie_docs)

chroma_client = chromadb.HttpClient(host='98.71.147.60', port=8000)



movie_vector_db = Chroma.from_documents(collection_name='test_movie_collection',
                                        documents=movie_docs_split, 
                                        embedding=embeddings, 
                                        client=chroma_client,)
