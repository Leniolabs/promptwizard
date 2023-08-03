import random

def convert_words_to_uppercase(string):
    string = string.lower()
    words = string.split()
    transformed_string = ""

    for word in words:
        if random.random() < 0.5:
            word = word.upper()
        transformed_string += word + " "
    return transformed_string.strip()

def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = convert_words_to_uppercase(prompt)
        new_prompts.append(new_prompt)
    return new_prompts