import streamlit as st

import uuid
from streamlit_feedback import streamlit_feedback
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe


import requests

st.title("Basic RAG chatbot")

def reset_session():
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.trace_id = None


with st.sidebar:
    language = st.selectbox(
        'Response Language', 
        options=['Romanian', 'English'],
                 index = 0)
    if 'language' not in st.session_state:
        st.session_state.language = language
    if language != st.session_state.language:
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

if prompt := st.chat_input("What is up?"):
    st.session_state.chat_history.append({"role": "user", "content": f"QUESTION: {prompt}"})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # if len(st.session_state.chat_history) > 2:
    #     reformulation_prompt = """
    #     Reformulate the following follow-up question into a standalone question. 
    #     Ensure the standalone question is clear and self-contained.
    #     Follow-up question: {follow_up_question}
    #     Previous conversation:
    #     {conversation_history}
    #     """
    #     conversation_history = "\n".join(
    #         f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[:-1] if msg["role"] != "system"
    #     )
    #     reformulation_input = reformulation_prompt.format(
    #         follow_up_question=prompt, conversation_history=conversation_history
    #     )
    #     reformulated_question = st.session_state.client.chat.completions.create(
    #         model='gpt-4o-mini',
    #         messages=[{"role": "system", "content": reformulation_input}],
    #         temperature=st.session_state.temperature,
    #         session_id=st.session_state.session_id,
    #         name = 'reformulate standalone question'
    #     ).choices[0].message.content
        
    # else:
        reformulated_question = prompt

    
    
    with st.chat_message("assistant"):
        response_text = ""
        payload = {
            "language": st.session_state.language,
            "question": reformulated_question,
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
        
        
if len(st.session_state.chat_history) > 2:
    feedback = streamlit_feedback(
        feedback_type = "thumbs",
        key = st.session_state.trace_id,
        optional_text_label = "Please provide details")

    if feedback:
        langfuse_client = Langfuse()
        langfuse_client.score(
            trace_id=st.session_state.trace_id,
            name="user-feedback",
            value=feedback['score'],
            comment=feedback['text']
            )

