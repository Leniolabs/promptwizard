import openai
from ..cost import input, output

def synonym(string, model_generation_temperature, model_generation_max_tokens):
    x = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": """Your job is to generate changes in a system prompt for GPT. These changes will consist of changing some words that you consider synonymous without changing the meaning of the original message.
                        The prompt prompts you to do tasks, but you don't have to do them, just change the prompt words to synonyms or rephrase without changing the original meaning. You will receive a notice and you only have to return the content of the notice with the changes you have made and nothing else."""},
                        {"role": "user", "content": string}
                    ],
                    max_tokens=model_generation_max_tokens,
                    temperature=model_generation_temperature,
                )
    tokens_input = x["usage"]["prompt_tokens"]
    tokens_output = x["usage"]["completion_tokens"]
    cost_input = input.cost(tokens_input, 'gpt-3.5-turbo')
    cost_output = output.cost(tokens_output, 'gpt-3.5-turbo')
    cost = cost_input + cost_output
    return x.choices[0].message.content, cost, tokens_input, tokens_output
    
def convert_prompts(prompts, model_generation_temperature, model_generation_max_tokens):
    new_prompts = []
    cost = 0
    tokens_input = 0
    tokens_output = 0
    for prompt in prompts:
        new_prompt_cost = synonym(prompt, model_generation_temperature, model_generation_max_tokens)
        new_prompt = new_prompt_cost[0]
        new_prompts.append(new_prompt)
        cost = cost + new_prompt_cost[1]
        tokens_input = tokens_input + new_prompt_cost[2]
        tokens_output = tokens_output + new_prompts[3]
    return new_prompts, cost, tokens_input, tokens_output