from prettytable import PrettyTable
import time
import openai

class Selection:
    def __init__(self, description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model):
        self.description = description
        self.test_cases = test_cases
        self.number_of_prompts = number_of_prompts
        self.generation_model = generation_model
        self.generation_model_temperature = generation_model_temperature
        self.generation_model_max_tokens = generation_model_max_tokens
        self.candidate_model = candidate_model

    def generate_candidate_prompts(self):
        outputs = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.
                 
Remember that the prompt should only allow the AI to answer the name of the country and nothing else.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified. I repeat, do not include the test cases.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""},
                {"role": "user", "content": f"Here are some test cases:`{self.test_cases}`\n\nHere is the description of the use-case: `{self.description.strip()}`\n\nRespond with your prompt, and nothing else. Be creative."}
                ],
            temperature=0.9,
            n=self.number_of_prompts)

        prompts = []

        for i in outputs.choices:
            prompts.append(i.message.content)
        return prompts

    def test_candidate_prompts(self, prompts):
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in prompts}

        # Initialize the table
        table = PrettyTable()
        table.field_names = ["Prompt", "Expected"] + [f"Prompt {i+1}-{j+1}" for j, prompt in enumerate(prompts) for i in range(prompts.count(prompt))]


        # Wrap the text in the "Prompt" column
        table.max_width["Prompt"] = 100


        for test_case in self.test_cases:
            row = [test_case['prompt'], test_case['answer']]
            for prompt in prompts:
                x = openai.ChatCompletion.create(
                    model=self.generation_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"{test_case['prompt']}"}
                    ],
                    max_tokens=self.generation_model_max_tokens,
                    temperature=self.generation_model_temperature,
                ).choices[0].message.content

                status = "✅" if x == test_case['answer'] else "❌"
                row.append(status)

                # Update model results
                if x == test_case['answer']:
                    prompt_results[prompt]['correct'] += 1
                prompt_results[prompt]['total'] += 1

            table.add_row(row)

        print(table)

        # Calculate and print the percentage of correct answers and average time for each model
        best_prompt = None
        best_percentage = 0
        for i, prompt in enumerate(prompts):
            correct = prompt_results[prompt]['correct']
            total = prompt_results[prompt]['total']
            percentage = (correct / total) * 100
            print(f"Prompt {i+1} got {percentage:.2f}% correct.")
            if percentage > best_percentage:
                best_percentage = percentage
                best_prompt = prompt
        
        print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
        return table
    
    def generate_optimal_prompt(self):
        candidate_prompts = self.generate_candidate_prompts()
        return self.test_candidate_prompts(candidate_prompts)