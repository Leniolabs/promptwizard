import openai
from ..cost import input, output

class Includes:
    def __init__(self, description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts):
        self.description = description
        self.test_cases = test_cases
        self.number_of_prompts = number_of_prompts
        self.model_test = model_test
        self.model_generation = model_generation
        self.model_test_temperature = model_test_temperature
        self.model_test_max_tokens = model_test_max_tokens
        self.model_generation_temperature = model_generation_temperature
        self.system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Specify in the prompts that you generate that they give a step-by-step response.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        self.prompts = prompts

    def test_candidate_prompts(self):
        cost = 0
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in self.prompts}
        results = [{"description": self.description, "method": "Equal"}]
        for prompt in self.prompts:
            prompt_and_results = [{"prompt": prompt}]
            for test_case in self.test_cases:
                response = openai.ChatCompletion.create(
                    model=self.model_test,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"{test_case[0]}"}
                    ],
                    max_tokens=self.model_test_max_tokens,
                    temperature=self.model_test_temperature,
                )
                tokens_input = response["usage"]["prompt_tokens"]
                tokens_output = response["usage"]["completion_tokens"]
                cost_input = input.cost(tokens_input, self.model_test)
                cost_output = output.cost(tokens_output, self.model_test)
                cost = cost_input + cost_output
                # Update model results
                if test_case[1].lower() in response.choices[0].message.content.lower():
                    prompt_results[prompt]['correct'] += 1
                prompt_results[prompt]['total'] += 1
                prompt_and_results.append({"test": test_case[0], "answer": response.choices[0].message.content, "ideal": test_case[1], "result": test_case[1].lower() in response.choices[0].message.content.lower()})
            results.append(prompt_and_results)
            prompt_and_results = []

        # Calculate and print the percentage of correct answers and average time for each model
        best_prompt = self.prompts[0]
        best_percentage = 0
        data_list = []
        for i, prompt in enumerate(self.prompts):
            correct = prompt_results[prompt]['correct']
            total = prompt_results[prompt]['total']
            percentage = (correct / total) * 100
            data_list.append({"prompt": prompt, "rating": percentage})
            print(f"Prompt {i+1} got {percentage:.2f}% correct.")
            if percentage >= best_percentage:
                best_percentage = percentage
                best_prompt = prompt
        sorted_data = sorted(data_list, key=lambda x: x['rating'], reverse=True)
        best_prompts = [sorted_data[0], sorted_data[1]]
        print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
        sorted_data.append(results)
        return sorted_data, best_prompts, cost
    
    def evaluate_optimal_prompt(self):
        return self.test_candidate_prompts()