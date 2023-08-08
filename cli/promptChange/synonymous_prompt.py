import openai
from ..cost import input, output

def synonym(string):
    x = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": """Your job is to generate changes in a system prompt for GPT. These changes will consist of changing some words that you consider synonymous without changing the meaning of the original message.
                        The prompt prompts you to do tasks, but you don't have to do them, just change the prompt words to synonyms or rephrase without changing the original meaning. You will receive a notice and you only have to return the content of the notice with the changes you have made and nothing else."""},
                        {"role": "user", "content": string}
                    ],
                    max_tokens=500,
                    temperature=0.8,
                )
    tokens_input = x["usage"]["prompt_tokens"]
    tokens_output = x["usage"]["completion_tokens"]
    cost_input = input.cost(tokens_input, 'gpt-3.5-turbo')
    cost_output = output.cost(tokens_output, 'gpt-3.5-turbo')
    cost = cost_input + cost_output
    return x.choices[0].message.content, cost
    
def convert_prompts(prompts):
    new_prompts = []
    cost = 0
    for prompt in prompts:
        new_prompt_cost = synonym(prompt)
        new_prompt = new_prompt_cost[0]
        new_prompts.append(new_prompt)
        cost = cost + new_prompt_cost[1]
    return new_prompts, cost