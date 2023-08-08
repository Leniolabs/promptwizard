import openai
from ..cost import input, output

def grammatical_errors_prompt(string):
    x = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": """Your job is to generate changes to a "system prompt" for GPT. Said changes will consist of changing some words that you consider for others with grammatical errors without changing the meaning of the original prompt. 
                         The prompt tells you to do tasks but you don't have to do the tasks, just change words from the prompt for others with misspellings. You will receive a prompt and you only have to return the content of the prompt with the changes with grammatical errors that you have made and nothing else."""},
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
        new_prompt_cost = grammatical_errors_prompt(prompt)[0]
        new_prompt = new_prompt_cost[0]
        new_prompts.append(new_prompt)
        cost = cost + new_prompt_cost[1]
    return new_prompts, cost