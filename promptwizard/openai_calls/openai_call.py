import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type,
)
from typing import List
from openai import OpenAI
import httpx

client = OpenAI()

def create_chat_completion(model: str, messages :List[dict], max_tokens: int, temperature: float, number_of_prompts: int, logit_bias: dict=None, functions: dict=None, function_call: str=None, timeout: int=10, n_retries: int=5):

    if model == 'gpt-4-turbo':
            model = 'gpt-4-1106-preview'

    client = OpenAI(timeout=httpx.Timeout(timeout, read=5.0, write=10.0, connect=3.0))
            
    @retry(wait=wait_random_exponential(min=1, max=timeout), stop=stop_after_attempt(n_retries), retry=retry_if_not_exception_type(openai.BadRequestError))
    def create_chat_completion_retry():
            
        try:
            if (logit_bias==None and functions==None):

                respond = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n = number_of_prompts,
                    temperature=temperature,
                )

            elif functions!=None:
                    
                respond = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n = number_of_prompts,
                    temperature=temperature,
                    functions=functions,
                    function_call=function_call,
                )

            else:

                respond = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n = number_of_prompts,
                    temperature=temperature,
                    logit_bias=logit_bias,
                )
        
        except openai.APITimeoutError as e:
            print(f"API Timeout Error: {e}")
        except openai.APIError as err:
            print(f"API Error: {err}")
        except openai.OpenAIError as e:
            print(f"Error in request: {e}")
            raise
        except Exception as err:
            print(f"An unexpected error occurred: {e}")
                
        return respond
    return create_chat_completion_retry()

def create_embedding(model: str, input: str, timeout: int=10, n_retries: int=5):

    client = OpenAI(timeout=httpx.Timeout(timeout, read=5.0, write=10.0, connect=3.0))

    @retry(wait=wait_random_exponential(min=1, max=timeout), stop=stop_after_attempt(n_retries), retry=retry_if_not_exception_type(openai.BadRequestError))
    def create_embedding_retries():

        try:
            embedding = client.embeddings.create(
                model=model,
                input=input
            )

        except openai.OpenAIError as e:
            print(f"Error in request: {e}")
            raise

        return embedding
    return create_embedding_retries()

def create_completion(model: str, messages: str, max_tokens: int, temperature: float, number_of_prompts: int, timeout: int=10, n_retries: int=5):

    client = OpenAI(timeout=httpx.Timeout(timeout, read=5.0, write=10.0, connect=3.0))

    @retry(wait=wait_random_exponential(min=1, max=timeout), stop=stop_after_attempt(n_retries), retry=retry_if_not_exception_type(openai.BadRequestError))
    def create_completion_retry():
    
        try:

            respond = client.completions.create(
                model=model,
                prompt=messages,
                max_tokens=max_tokens,
                n = number_of_prompts,
                temperature=temperature,
                logprobs=5,
            )
        
        except openai.OpenAIError as e:
            print(f"Error in request: {e}")
            raise
                
        return respond
    return create_completion_retry()

def create_assistant(model: str, instructions: str, tools: List[dict]):

    if model == 'gpt-4-turbo':
            model = 'gpt-4-1106-preview'

    try:
        assistant = client.beta.assistants.create(
            instructions=instructions,
            model=model,
            tools=tools
        )

    except openai.OpenAIError as e:
            print(f"Error in request: {e}")
            raise
    
    return assistant

def create_thread():

    thread = client.beta.threads.create()

    return thread

def create_message(thread_id, role: str, content: str):
    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )

    except openai.OpenAIError as e:
            print(f"Error in request: {e}")
            raise
    
    return message

def create_run(thread_id, assistant_id):
    try:
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

    except openai.OpenAIError as e:
            print(f"Error in request: {e}")
            raise
    
    return run

def retrieve(thread_id, run_id):
    try:
        retribed_run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

    except openai.OpenAIError as e:
        print(f"Error in request: {e}")
        raise
    
    return retribed_run

def thread_messages(thread_id):
    
    try:

        thread_messages = client.beta.threads.messages.list(thread_id)
    
    except openai.OpenAIError as e:
        print(f"Error in request: {e}")
        raise
    
    return thread_messages

def delete_assistant(assistant_id):

    client.beta.assistants.delete(assistant_id)