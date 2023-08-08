import openai
import tiktoken
from .cost import input, output

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        if isinstance(message, dict):
            for key, value in message.items():
                if key == "content":
                    num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def generate_candidate_prompts(system_gen_system_prompt, test_cases, description, model_generation, model_generation_temperature, number_of_prompts, prompt_features=None):
        if prompt_features is not None:
            messages = [
                {"role": "system", "content": system_gen_system_prompt + ' ' + prompt_features},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative."}
                ]
            outputs = openai.ChatCompletion.create(
            model=model_generation,
            messages= messages,
            temperature=model_generation_temperature,
            n=number_of_prompts)
            messages = []
            for choice in outputs["choices"]:
                messages.append(choice["message"])
            #print(num_tokens_from_messages(messages, model_generation))
        else:
            messages=[
                {"role": "system", "content": system_gen_system_prompt},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative."}
                ]
            outputs = openai.ChatCompletion.create(
            model=model_generation,
            messages=messages,
            temperature=model_generation_temperature,
            n=number_of_prompts)
            messages = []
            for choice in outputs["choices"]:
                messages.append(choice["message"])
            #print(num_tokens_from_messages(messages, model_generation))
        tokens_input = outputs["usage"]["prompt_tokens"]
        tokens_output = outputs["usage"]["completion_tokens"]
        cost_input = input.cost(tokens_input, model_generation)
        cost_output = output.cost(tokens_output, model_generation)
        cost = cost_input + cost_output

        prompts = []
        for i in outputs.choices:
            prompts.append(i.message.content)
        return prompts, cost