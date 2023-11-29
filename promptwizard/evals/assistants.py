from ..openai_calls import openai_call
from concurrent.futures import ThreadPoolExecutor
from typing import List
from openai import OpenAI
import httpx

class Assistants:
    def __init__(self, test_cases: List[dict], prompts: List[str], model_test: str="gpt-3.5-turbo", model_test_temperature: float=0.6, model_test_max_tokens: int=1000, best_prompts: int=2, timeout: int=10, n_retries: int=5):

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
        self.n_retries = n_retries

        self.client = OpenAI(timeout=httpx.Timeout(timeout, read=5.0, write=10.0, connect=3.0))

    def test_candidate_prompts(self):
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in self.prompts}
        results = [{"method": "Assistants"}]

        def evaluate_prompt(prompt):
            prompt_and_results = [{"prompt": prompt}]
            assistant = openai_call.create_assistant(self.model_test, prompt, [{"type": "code_interpreter"}])
            thread = openai_call.create_thread()
            
            for test_case in self.test_cases:
                message = openai_call.create_message(thread.id, "user", test_case['input'])
                run = openai_call.create_run(thread.id, assistant.id)
                
                finished = False
                while not finished:
                    if openai_call.retrieve(thread.id, run.id).status == "completed":
                        finished = True

            thread_messages = openai_call.thread_messages(thread.id)
            assistant_messages = []
            current_message = None

            for message in reversed(thread_messages.data):
                if message.role == "assistant":
                    if current_message is None:
                        current_message = {"text": message.content[0].text.value}
                    else:
                        current_message["text"] += " " + message.content[0].text.value
                elif message.role == "user":
                    if current_message is not None:
                        assistant_messages.append(current_message)
                        current_message = None

            if current_message is not None:
                assistant_messages.append(current_message)

            j = 0
            while j < len(self.test_cases):
                if str(self.test_cases[j]['output']).lower() in assistant_messages[j]['text']:
                    prompt_results[prompt]['correct'] += 1
                prompt_results[prompt]['total'] += 1

                prompt_and_results.append({"test": self.test_cases[j]['input'], "answer": assistant_messages[j]['text'], "ideal": self.test_cases[j]['output'], "result": str(self.test_cases[j]['output']).lower() in assistant_messages[j]['text']})
                j = j + 1
            
            results.append(prompt_and_results)
            prompt_and_results = []
            openai_call.delete_assistant(assistant.id)

        with ThreadPoolExecutor(max_workers=len(self.prompts)) as executor:
            executor.map(evaluate_prompt, self.prompts)

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
        return sorted_data, best_prompts
    
    def evaluate_optimal_prompt(self):

        """
        Evaluate the optimal prompt by testing candidate prompts and selecting the best ones.

        Returns:
            tuple: A tuple containing the result data, best prompts, cost, input tokens used, and output tokens used.
        """
        
        return self.test_candidate_prompts()