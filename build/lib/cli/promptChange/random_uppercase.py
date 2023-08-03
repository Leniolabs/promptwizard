import random

def convert_to_uppercase_randomly(string):
    string = string.lower()
    for i in range(len(string)):
        if random.random() < 0.5:
            string = string[:i] + string[i].upper() + string[i+1:]

    return string

def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = convert_to_uppercase_randomly(prompt)
        new_prompts.append(new_prompt)
    return new_prompts