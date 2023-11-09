from promptwizard.cost import input, output
from ..openai_calls import openai_call
from typing import List, Dict

def generate_candidate_prompts(system_gen_system_prompt: str, test_cases: List[Dict], description: str, model_generation: str='gpt-4', model_generation_temperature: float=1.2, model_generation_max_tokens: int=500, number_of_prompts: int=4, prompt_features: str=None, timeout: int=10, n_retries: int=5):
        
        # Check if additional prompt features are provided
        if prompt_features is not None:
            # Construct messages for generating prompts with prompt features
            model=model_generation
            messages = [
                {"role": "system", "content": system_gen_system_prompt + ' ' + prompt_features},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative. Amplify your vocabulary as much as you can. All your prompts must be different from each other."}
                ]
            temperature=model_generation_temperature
            n=number_of_prompts
            max_tokens=model_generation_max_tokens
            outputs = openai_call.create_chat_completion(model, messages, max_tokens, temperature, n, timeout=timeout, n_retries=n_retries)
        else:
            # Construct messages for generating prompts without prompt features
            model=model_generation
            messages=[
                {"role": "system", "content": system_gen_system_prompt},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative. Amplify your vocabulary as much as you can. All your prompts must be different from each other."}
                ]
            temperature=model_generation_temperature
            n=number_of_prompts
            max_tokens=model_generation_max_tokens
            # Generate prompts using OpenAI's ChatCompletion API
            outputs = openai_call.create_chat_completion(model, messages, max_tokens, temperature, n, timeout=timeout, n_retries=n_retries)

        # Extract tokens input and tokens output usage
        tokens_input = outputs.usage.prompt_tokens
        tokens_output = outputs.usage.completion_tokens

        # Calculate the cost based on token usage
        cost_input = input.cost(tokens_input, model_generation)
        cost_output = output.cost(tokens_output, model_generation)
        cost = cost_input + cost_output

        # Extract the generated prompts
        prompts = [choice.message.content for choice in outputs.choices]
        return prompts, cost, tokens_input, tokens_output