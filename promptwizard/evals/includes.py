from ..openai_calls import openai_call
from ..cost import input, output
import concurrent.futures

class Includes:
    def __init__(self, test_cases, prompts, model_test="gpt-3.5-turbo", model_test_temperature=0.6, model_test_max_tokens=1000, best_prompts=2, timeout=10, n_retries=5):

        """
        Initialize a Includes instance.

        Args:
            test_cases (list): List of test cases to evaluate.
            number_of_prompts (int): Number of prompts to generate and/or test.
            model_test (str): The language model used for testing.
            model_test_temperature (float): The temperature parameter for the testing model.
            model_test_max_tokens (int): The maximum number of tokens allowed for the testing model.
            model_generation (str): The language model used for generating prompts.
            model_generation_temperature (float): The temperature parameter for the generation model.
            prompts (list): List of prompts to evaluate.
            best_prompts (int): Number of best prompts to consider.

        Note:
            The 'system_gen_system_prompt' attribute is predefined within the class constructor.
        """

        self.test_cases = test_cases
        self.model_test = model_test
        self.model_test_temperature = model_test_temperature
        self.model_test_max_tokens = model_test_max_tokens
        self.system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Specify in the prompts that you generate that they give a step-by-step response.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        self.prompts = prompts
        self.best_prompts = best_prompts
        self.timeout = timeout
        self.n_retries = n_retries

    def process_prompt(self, prompt, test_case, model, model_max_tokens, model_temperature):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{test_case['input']}"}
        ]
        response = openai_call.create_chat_completion(model, messages, model_max_tokens, model_temperature, 1, timeout=self.timeout, n_retries=self.n_retries)
        partial_tokens_input = response["usage"]["prompt_tokens"]
        partial_tokens_output = response["usage"]["completion_tokens"]
        result_content = response.choices[0].message.content
        
        return partial_tokens_input, partial_tokens_output, result_content, test_case['output']

    def test_candidate_prompts(self):

        """
        Test a list of candidate prompts with test cases and evaluate their performance.

        Returns:
            tuple: A tuple containing the following elements:
                - List of results and statistics.
                - List of best-performing prompts.
                - Cost of generating and testing prompts.
                - Tokens used for input.
                - Tokens used for output.
        """

        cost = 0
        tokens_input = 0
        tokens_output = 0
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in self.prompts}
        results = [{"method": "Includes"}]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for prompt in self.prompts:
                prompt_and_results = [{"prompt": prompt}]
                for test_case in self.test_cases:
                    future = executor.submit(
                        self.process_prompt,
                        prompt,
                        test_case,
                        self.model_test,
                        self.model_test_max_tokens,
                        self.model_test_temperature
                    )
                    futures.append(future)
                    partial_tokens_input, partial_tokens_output, result_content, ideal_output = future.result()
                    tokens_input += partial_tokens_input
                    tokens_output += partial_tokens_output

                    if ideal_output.lower() in result_content.lower():
                        prompt_results[prompt]['correct'] += 1
                    prompt_results[prompt]['total'] += 1

                    prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": ideal_output.lower() in result_content.lower()})
                
                results.append(prompt_and_results)
                prompt_and_results = []

        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost = cost + cost_input + cost_output

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
        best_prompts = sorted_data[:self.best_prompts]
        print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
        sorted_data.append(results)
        return sorted_data, best_prompts, cost, tokens_input, tokens_output
    
    def evaluate_optimal_prompt(self):

        """
        Evaluate the optimal prompt by testing candidate prompts and selecting the best ones.

        Returns:
            tuple: A tuple containing the result data, best prompts, cost, input tokens used, and output tokens used.
        """
        
        return self.test_candidate_prompts()