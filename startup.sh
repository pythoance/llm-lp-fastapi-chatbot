#!/bin/bash
# Start FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
streamlit run frontend/app.py --server.port 8501 --server.address=0.0.0.0