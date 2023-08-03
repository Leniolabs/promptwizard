def convert_to_uppercase(string):
    return string.upper()

def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = convert_to_uppercase(prompt)
        new_prompts.append(new_prompt)
    return new_prompts