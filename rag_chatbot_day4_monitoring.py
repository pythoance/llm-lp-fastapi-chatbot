from langchain_openai import AzureOpenAIEmbeddings
import streamlit as st
import os
from langchain_chroma import Chroma
from langfuse.openai import AzureOpenAI
import uuid
from streamlit_feedback import streamlit_feedback
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
import json
from pydantic import BaseModel

st.title("Basic RAG chatbot")


if "client" not in st.session_state:
    st.session_state.client = AzureOpenAI(
        azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT'),
        api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
        api_version="2024-06-01" )
    st.session_state.embedding_model = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version="2024-06-01",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    st.session_state.retriever = Chroma(
        persist_directory = 'local_movie_db', 
        embedding_function=st.session_state.embedding_model).as_retriever(
            search_kwargs={"k": 4})
    

def get_system_prompt(language):
    return f"""
    You are a helpful assistant for question on famous movies. 
    You will formulate all its answers in {language}.
    Base you answer only on pieces of information received as context below. 
    If you don't know the answer, just say that you don't know. 
    Do not answer any question that are not related to movies."""

import numpy as np
def evaluate_faithfullness(query, context, response):
    FAITHFULLNESS_PROMPT = f"""Your task is to judge the faithfulness or groundedness of statements based on context information.
First, please extract statements from a provided response to a query.
Second, calculate a faithfulness score for each statement made in the predicted answer.
The score is 1 if the statement can be inferred from the provided context or 0 if it cannot be inferred.

EXAMPLE
"inputs": {{
    "query": "What is the capital of Italy?", 
    "context": ["Rome is the capital of Italy."],
    "response": "Rome is the capital of Italy with more than 4 million inhabitants."
}},
"outputs": {{
    "statements": ["Rome is the capital of Italy.", "Rome has more than 4 million inhabitants."],
    "statement_scores": [1, 0]
}}

RESPONSE FORMAT
Format your answer as a valid JSON in the following format:
{{
    "statements": <insert a list of statements extracted from the response>,
    "statement_scores": <insert a list of scores (0 or 1) for each statement>
}}

INPUT:
Query: {query}
Context: {context}
Ground truth: {response}"""


    response = json.loads(st.session_state.client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[{"role": "system", "content": FAITHFULLNESS_PROMPT}],
        temperature=st.session_state.temperature,
        session_id=st.session_state.session_id,
        name = 'evaluate context faithfullness',
        response_format={ "type": "json_object" }
    ).choices[0].message.content)

    langfuse_client = Langfuse()
    langfuse_client.score(
        trace_id=st.session_state.trace_id,
        name="faithfulness_evaluation",
            value=np.mean(response['statement_scores']),
            comment=str(response)
            )



def reset_session():
    system_prompt = get_system_prompt(st.session_state.language)
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]
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

    st.session_state.temperature = st.slider(
        'Temperature', 
        min_value=0.0, 
        max_value=1.0, 
        value = 0.5)
    
if st.button('Start New Chat'):
    reset_session()

if "chat_history" not in st.session_state:
    reset_session()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


for message in st.session_state.chat_history:
    if message["role"] != "system":
         with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.chat_history.append({"role": "user", "content": f"QUESTION: {prompt}"})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    
    if len(st.session_state.chat_history) > 2:
        reformulation_prompt = """
        Reformulate the following follow-up question into a standalone question. 
        Ensure the standalone question is clear and self-contained.
        Follow-up question: {follow_up_question}
        Previous conversation:
        {conversation_history}
        """
        conversation_history = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[:-1] if msg["role"] != "system"
        )
        reformulation_input = reformulation_prompt.format(
            follow_up_question=prompt, conversation_history=conversation_history
        )
        reformulated_question = st.session_state.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "system", "content": reformulation_input}],
            temperature=st.session_state.temperature,
            session_id=st.session_state.session_id,
            name = 'reformulate standalone question'
        ).choices[0].message.content
        
    else:
        reformulated_question = prompt

    context = format_docs(st.session_state.retriever.invoke(reformulated_question))
    st.session_state.chat_history.append({"role": "system", "content": f"CONTEXT: {context}"})
    
    with st.chat_message("assistant"):
        
        @observe(capture_output=False)
        def get_response_stream(reformulated_question, context):
            trace_id = langfuse_context.get_current_trace_id()
            
            stream = st.session_state.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": get_system_prompt(st.session_state.language)},
                    {"role": "user", "content": f"QUESTION: {reformulated_question}"},
                    {"role": "system", "content": f"CONTEXT: {context}"}],
                temperature=st.session_state.temperature,
                stream=True,
                session_id=st.session_state.session_id)
           
            response = st.write_stream(stream)
            langfuse_context.update_current_observation(
                output = response
            )
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            return response, trace_id

        response, st.session_state.trace_id = get_response_stream(reformulated_question, context)
        
        evaluate_faithfullness(reformulated_question, context, response)

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
        
