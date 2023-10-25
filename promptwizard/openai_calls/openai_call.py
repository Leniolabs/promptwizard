import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type,
)



def create_chat_completion(model, messages, max_tokens, temperature, number_of_prompts, logit_bias=None, functions=None, function_call=None, timeout=10, n_retries=5):
    @retry(wait=wait_random_exponential(min=1, max=timeout), stop=stop_after_attempt(n_retries), retry=retry_if_not_exception_type(openai.InvalidRequestError))
    def create_chat_completion_retry():
    
        try:
            if (logit_bias==None and functions==None):

                respond = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n = number_of_prompts,
                    temperature=temperature,
                    request_timeout=timeout,
                )

            elif functions!=None:
                    
                respond = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n = number_of_prompts,
                    temperature=temperature,
                    functions=functions,
                    function_call=function_call,
                    request_timeout=timeout,
                )

            else:

                respond = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n = number_of_prompts,
                    temperature=temperature,
                    logit_bias=logit_bias,
                    request_timeout=timeout,
                )
        
        except openai.error.OpenAIError as e:
            print(f"Error in request: {e}")
            raise
                
        return respond
    return create_chat_completion_retry()
def create_embedding(model, input, timeout, n_retries):
    @retry(wait=wait_random_exponential(min=1, max=timeout), stop=stop_after_attempt(n_retries), retry=retry_if_not_exception_type(openai.InvalidRequestError))
    def create_embedding_retries():

        try:
            embedding = openai.Embedding.create(
                model=model,
                input=input
            )

        except openai.error.OpenAIError as e:
            print(f"Error in request: {e}")
            raise

        return embedding
    return create_embedding_retries()