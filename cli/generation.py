import openai

def generate_candidate_prompts(system_gen_system_prompt, test_cases, description, candidate_model, candidate_model_temperature, number_of_prompts):
        outputs = openai.ChatCompletion.create(
        model=candidate_model, # change this to gpt-3.5-turbo if you don't have GPT-4 access
        messages=[
            {"role": "system", "content": system_gen_system_prompt},
            {"role": "user", "content": f"Here are some test cases:`{test_cases}`\n\nHere is the description of the use-case: `{description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative."}
            ],
        temperature=candidate_model_temperature,
        n=number_of_prompts)

        prompts = []

        for i in outputs.choices:
            prompts.append(i.message.content)
        return prompts