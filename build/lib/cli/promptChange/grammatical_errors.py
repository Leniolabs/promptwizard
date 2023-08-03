import openai

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
                ).choices[0].message.content
    return x
    
def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = grammatical_errors_prompt(prompt)
        new_prompts.append(new_prompt)
    return new_prompts