import openai

def create_chat_completion(model, messages, max_tokens, temperature, number_of_prompts, logit_bias=None, functions=None, function_call=None):
    if (logit_bias==None and functions==None):
        respond = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n = number_of_prompts,
            temperature=temperature,
        )
    elif functions!=None:
        respond = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n = number_of_prompts,
            temperature=temperature,
            functions=functions,
            function_call=function_call
        )
    else:
        respond = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n = number_of_prompts,
            temperature=temperature,
            logit_bias=logit_bias
        )
    return respond