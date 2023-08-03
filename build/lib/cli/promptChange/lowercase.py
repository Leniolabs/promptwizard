def convert_to_lowercase(string):
    return string.lower()

def convert_prompts(prompts):
    new_prompts = []
    for prompt in prompts:
        new_prompt = convert_to_lowercase(prompt)
        new_prompts.append(new_prompt)
    return new_prompts