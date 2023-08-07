import openai

def generate_candidate_prompts(system_gen_system_prompt, test_cases, description, model_generation, model_generation_temperature, number_of_prompts, prompt_features=None):
        if prompt_features is not None:
            outputs = openai.ChatCompletion.create(
            model=model_generation,
            messages=[
                {"role": "system", "content": system_gen_system_prompt + ' ' + prompt_features},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative."}
                ],
            temperature=model_generation_temperature,
            n=number_of_prompts)
        else:
             outputs = openai.ChatCompletion.create(
            model=model_generation,
            messages=[
                {"role": "system", "content": system_gen_system_prompt},
                {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative."}
                ],
            temperature=model_generation_temperature,
            n=number_of_prompts)

        prompts = []
        for i in outputs.choices:
            prompts.append(i.message.content)
        return prompts