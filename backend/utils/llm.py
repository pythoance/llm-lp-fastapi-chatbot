import os

from langfuse.openai import AsyncAzureOpenAI
import yaml

class LLM:
    def __init__(self):
        self.config = self.load_config()
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            api_version=self.config['API_VERSION']
            )
        
    @staticmethod
    def load_config():
        file_path = os.path.join('backend', 'config', 'llm.yaml')
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
        
    @staticmethod
    async def stream_processor(response, context):
        async for chunk in response:
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:

                    yield delta.content
        yield f"\n\nCONTEXT: {context}" 
    
    @staticmethod
    def get_system_prompt(language: str) -> str:
        return f"""
        You are a helpful assistant for question on famous movies.
        You will formulate all your answers in {language}.
        Base your answer only on sources of context below. 
        If you don't know the answer, just say that you don't know. 
        Do not answer any question that are not related to movies."""
    
    
    async def get_chatbot_response(
            self, 
            question: str, 
            context: str,
            response_language: str,
            session_id: str,
            stream: bool = True):
         
        messages = [
            {"role": "system", "content": self.get_system_prompt(response_language)},
            {"role": "user", "content": f"QUESTION: {question}"},
            {"role": "system", "content": f"CONTEXT: {context}"}]
        
        streaming_response = await self.client.chat.completions.create(
            model=self.config['LLM_DEPLOYMENT_NAME'],
            messages=messages,
            temperature=self.config['TEMPERATURE'],
            stream=stream,
            session_id=session_id,
            max_completion_tokens=self.config['MAX_COMPLETION_TOKENS']
            )
        
        return self.stream_processor(streaming_response, context)