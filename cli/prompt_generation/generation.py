import openai
from .cost import input, output

def generate_candidate_prompts(system_gen_system_prompt, test_cases, description, model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts, prompt_features=None):
        if prompt_features is not None:
            messages = [
                {"role": "system", "content": system_gen_system_prompt + ' ' + prompt_features},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative. Amplify your vocabulary as much as you can. All your prompts must be different from each other."}
                ]
            outputs = openai.ChatCompletion.create(
            model=model_generation,
            messages= messages,
            temperature=model_generation_temperature,
            n=number_of_prompts,
            max_tokens=model_generation_max_tokens)
            messages = []
            for choice in outputs["choices"]:
                messages.append(choice["message"])
        else:
            messages=[
                {"role": "system", "content": system_gen_system_prompt},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative. Amplify your vocabulary as much as you can. All your prompts must be different from each other."}
                ]
            outputs = openai.ChatCompletion.create(
            model=model_generation,
            messages=messages,
            temperature=model_generation_temperature,
            n=number_of_prompts,
            max_tokens=model_generation_max_tokens)
            messages = []
            for choice in outputs["choices"]:
                messages.append(choice["message"])
        tokens_input = outputs["usage"]["prompt_tokens"]
        tokens_output = outputs["usage"]["completion_tokens"]
        cost_input = input.cost(tokens_input, model_generation)
        cost_output = output.cost(tokens_output, model_generation)
        cost = cost_input + cost_output

        prompts = []
        for i in outputs.choices:
            prompts.append(i.message.content)
        return prompts, cost, tokens_input, tokens_output