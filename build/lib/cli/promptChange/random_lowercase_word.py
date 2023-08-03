import random

def convert_words_to_lowercase(string):
    string = string.upper()
    words = string.split()
    transformed_string = ""

    for word in words:
        if random.random() < 0.5:
            word = word.lower()
        transformed_string += word + " "
    return transformed_string.strip()

def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = convert_words_to_lowercase(prompt)
        new_prompts.append(new_prompt)
    return new_prompts