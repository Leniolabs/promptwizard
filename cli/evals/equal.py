import time
import openai

class Equal:
    def __init__(self, description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model, candidate_model_temperature, prompts):
        self.description = description
        self.test_cases = test_cases
        self.number_of_prompts = number_of_prompts
        self.generation_model = generation_model
        self.generation_model_temperature = generation_model_temperature
        self.generation_model_max_tokens = generation_model_max_tokens
        self.candidate_model = candidate_model
        self.candidate_model_temperature = candidate_model_temperature
        self.prompts = prompts
        self.system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.
                 
Remember that the prompt should only allow the AI to answer the answer and nothing else. No explanation is necessary.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified. I repeat, do not include the test cases.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""

    def test_candidate_prompts(self):
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in self.prompts}
        for test_case in self.test_cases:
            for prompt in self.prompts:
                x = openai.ChatCompletion.create(
                    model=self.generation_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"{test_case['prompt']}"}
                    ],
                    max_tokens=self.generation_model_max_tokens,
                    temperature=self.generation_model_temperature,
                ).choices[0].message.content
                # Update model results
                if x == test_case['answer']:
                    prompt_results[prompt]['correct'] += 1
                prompt_results[prompt]['total'] += 1

        # Calculate and print the percentage of correct answers and average time for each model
        best_prompt = None
        best_percentage = 0
        data_list = []
        for i, prompt in enumerate(self.prompts):
            correct = prompt_results[prompt]['correct']
            total = prompt_results[prompt]['total']
            percentage = (correct / total) * 100
            data_list.append({"prompt": prompt, "rating": percentage})
            print(f"Prompt {i+1} got {percentage:.2f}% correct.")
            if percentage > best_percentage:
                best_percentage = percentage
                best_prompt = prompt
        
        print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
        return data_list
    
    def evaluate_optimal_prompt(self):
        return self.test_candidate_prompts()