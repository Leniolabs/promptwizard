import openai

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
                ).choices[0].message.content
    return x
    
def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = synonym(prompt)
        new_prompts.append(new_prompt)
    return new_prompts