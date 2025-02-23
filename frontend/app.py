import requests
import uuid

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
import streamlit as st
from streamlit_feedback import streamlit_feedback

st.title("Movie Chatbot with RAG")

def reset_session():
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.trace_id = None


with st.sidebar:
    language = st.selectbox(
        'Response Language', 
        options=['Romanian', 'English'],
        index=0)
    if 'language' not in st.session_state or language != st.session_state.language:
        st.session_state.language = language
        reset_session()
    
if st.button('Start New Chat'):
    reset_session()

if "chat_history" not in st.session_state:
    reset_session()

for message in st.session_state.chat_history:
    if message["role"] != "system":
         with st.chat_message(message["role"]):
            st.markdown(message["content"])

if question := st.chat_input("What is up?"):
    st.session_state.chat_history.append({"role": "user", "content": f"QUESTION: {question}"})
    with st.chat_message("user"):
        st.markdown(question)
            
    with st.chat_message("assistant"):
        response_text = ""
        payload = {
            "language": st.session_state.language,
            "question": question,
            "session_id": st.session_state.session_id
        }
        
        backend_url = "http://localhost:8000/chatbot-response-stream"
        r = requests.post(backend_url, json=payload, stream=True)
       
        context_started = False
        for chunk in r.iter_lines(decode_unicode=True):
            if chunk:
                if "CONTEXT:" in chunk:
                    context = chunk.replace("CONTEXT: ", "")
                    context_started = True
                elif context_started:
                    context += f"\n\n{chunk}"
                else:
                    response_text += chunk.replace("data: ", "").strip()
                    st.write(chunk.replace("data: ", "").strip())

        context = context.replace("$", "\\$")

        st.session_state.chat_history.append({"role": "system", "content": f"CONTEXT: {context}"})       
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        with st.expander("See Sources"):
            st.write(context)

