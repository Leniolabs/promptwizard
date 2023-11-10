from ..openai_calls import openai_call
from ..cost import input, output
import concurrent.futures
from typing import List, Dict
import math

class LogProbs:
    def __init__(self, test_cases: List[Dict], prompts: List[str], model_test: str="gpt-3.5-instruct", model_test_temperature: float=0.6, model_test_max_tokens: int=1000, best_prompts: int=2, timeout: int=10, n_retries: int=5):

        """
        Initialize a LogProbs instance.

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

        Specify in your prompts that the response has to be with the token with the highest logprobs and the response should noy have \n at the beginning.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        self.prompts = prompts
        self.best_prompts = best_prompts
        self.timeout = timeout
        self.n_retries = n_retries

    def process_prompt(self, prompt, test_case, model, model_max_tokens, model_temperature):
        messages = prompt + ' ' + test_case['input']
        response = openai_call.create_completion(model, messages, model_max_tokens, model_temperature, 1, timeout=self.timeout, n_retries=self.n_retries)
        partial_tokens_input = response.usage.prompt_tokens
        partial_tokens_output = response.usage.completion_tokens
        top_logprobs = response.choices[0].logprobs.top_logprobs
        
        return partial_tokens_input, partial_tokens_output, top_logprobs

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
        prompt_results = {prompt: {'total': 0} for prompt in self.prompts}
        results = [{"method": "LogProbs"}]

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
                    partial_tokens_input, partial_tokens_output, logprobs = future.result()
                    tokens_input += partial_tokens_input
                    tokens_output += partial_tokens_output
                    
                    def remove_empty_keys_first_dictionary(arrays):
                        if arrays:
                            first_dictionary = arrays[0]
                            keys_to_remove = [key for key in first_dictionary.keys() if '\n' in key or key == '' or (key == ' ') or (key == '|endoftext|')]
                            for key in keys_to_remove:
                                del first_dictionary[key]
                            return first_dictionary
                        
                    def remove_whitespace_from_keys(dictionary):
                        modified_dictionary = {}

                        for key, value in dictionary.items():
                            if key.startswith(' '):
                                new_key = key.lstrip(' ')

                                if new_key in modified_dictionary:
                                    modified_dictionary[new_key] = math.log(math.exp(modified_dictionary[new_key])*math.exp(value))
                                else:
                                    modified_dictionary[new_key] = value
                            elif key.lstrip(' ') in modified_dictionary:
                                modified_dictionary[key.lstrip(' ')] += value
                            else:
                                modified_dictionary[key] = value
                        return modified_dictionary
                    
                    first_dictionary1 = remove_empty_keys_first_dictionary(logprobs)
                    first_dictionary = remove_whitespace_from_keys(first_dictionary1)

                    def generate_combinations(dictionaries, index=0, current_combination="", current_percentage=1, memo={}):
                        if index == len(dictionaries):
                            return [{current_combination: current_percentage}]

                        if (index, current_combination) in memo:
                            return memo[(index, current_combination)]

                        results = []
                        for key, value in dictionaries[index].items():
                            new_combination = current_combination + key
                            new_percentage = current_percentage * math.exp(value)
                            results += generate_combinations(dictionaries, index + 1, new_combination, new_percentage, memo)

                        memo[(index, current_combination)] = results
                        return results
                    
                    combinations_tokens = generate_combinations(logprobs[1:])
                    combined_dict = {}
                    for dict in combinations_tokens:
                        combined_dict.update(dict)
                    
                    def multiply_dictionaries(dict1, dict2):
                        result = {}

                        for key1, value1 in dict1.items():
                            for key2, value2 in dict2.items():
                                new_key = key1 + key2
                                new_value = math.exp(value1) * math.exp(value2)
                                result[new_key] = new_value

                        return result

                    if len(first_dictionary)>0:
                        combination = multiply_dictionaries(first_dictionary, combined_dict)

                    combinations = {**combined_dict, **combination}

                    if test_case['output'] in combinations:
                        probability = combinations[test_case['output']]
                    
                    if not(test_case['output'] in combinations):
                        probability = 0

                    prompt_results[prompt]['total'] = prompt_results[prompt]['total'] + probability
                    prompt_and_results.append({"test": test_case['input'], "probability": probability, "answer": test_case["output"]})
                
                results.append(prompt_and_results)
                prompt_and_results = []

        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost = cost + cost_input + cost_output

        
        # Sort the prompts by score.
        data_list = []
        for i, prompt in enumerate(self.prompts):
            score = prompt_results[prompt]['total']/len(self.test_cases)
            data_list.append({"prompt": prompt, "rating": score})

        sorted_data = sorted(data_list, key=lambda x: x['rating'], reverse=True)
        best_prompts = sorted_data[:self.best_prompts]
        sorted_data.append(results)
        return sorted_data, best_prompts, cost, tokens_input, tokens_output
    
    def evaluate_optimal_prompt(self):

        """
        Evaluate the optimal prompt by testing candidate prompts and selecting the best ones.

        Returns:
            tuple: A tuple containing the result data, best prompts, cost, input tokens used, and output tokens used.
        """
        
        return self.test_candidate_prompts()