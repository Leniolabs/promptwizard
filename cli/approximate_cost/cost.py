import tiktoken
from ..cost import input, output

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations):
    tokens_input = 0
    tokens_output = 0
    if prompts_value == []:
        if method == 'includes.Includes':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Specify in the prompts that you generate that they give a step-by-step response.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        elif method == 'equal.Equal':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.
                 
Remember that the prompt should only allow the AI to answer the answer and nothing else. No explanation is necessary.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified. I repeat, do not include the test cases.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        elif method == 'classification.Classification':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        The prompts you will be generating will be for classifiers, with 'true' and 'false' being the only possible outputs.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        elif method == 'elovalue.Elo':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

The prompts you will be generating will be for freeform tasks, such as generating a landing page headline, an intro paragraph, solving a math problem, etc.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        elif method == 'function_calling.functionCalling':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        if prompt_change != 'None':
            message = [
                    {"role": "system", "content": system_gen_system_prompt + ' ' + prompt_change},
                    {"role": "user", "content": f"Here are some test cases:`{test_cases}\n\nRespond with your prompt, and nothing else. Be creative."}
                    ]
        if prompt_change == 'None':
            message = [
                    {"role": "system", "content": system_gen_system_prompt},
                    {"role": "user", "content": f"Here are some test cases:`{test_cases}\n\nRespond with your prompt, and nothing else. Be creative."}
                    ]
        tokens_output = tokens_output + number_of_prompts*model_generation_max_tokens
        tokens_input = num_tokens_from_messages(message, model=model_generation)
    cost_input = input.cost(tokens_input, model_generation)
    cost_output = output.cost(tokens_output, model_generation)
    cost = cost_input + cost_output
    return cost

def approximate_cost_change(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations):
    tokens_input = 0
    tokens_output = 0
    if prompt_change == 'synonymous_prompt':
        message=[
                        {"role": "system", "content": """Your job is to generate changes in a system prompt for GPT. These changes will consist of changing some words that you consider synonymous without changing the meaning of the original message.
                        The prompt prompts you to do tasks, but you don't have to do them, just change the prompt words to synonyms or rephrase without changing the original meaning. You will receive a notice and you only have to return the content of the notice with the changes you have made and nothing else."""}
                    ]
        tokens_output = tokens_output + number_of_prompts*model_generation_max_tokens
        tokens_input = tokens_input + num_tokens_from_string(message, 'gpt-3.5-turbo') + 1 + model_generation_max_tokens
    if prompt_change == 'grammatical_errors':
        message=[
                        {"role": "system", "content": """Your job is to generate changes to a "system prompt" for GPT. Said changes will consist of changing some words that you consider for others with grammatical errors without changing the meaning of the original prompt. 
                         The prompt tells you to do tasks but you don't have to do the tasks, just change words from the prompt for others with misspellings. You will receive a prompt and you only have to return the content of the prompt with the changes with grammatical errors that you have made and nothing else."""}
                        ]
        tokens_output = tokens_output + number_of_prompts*model_generation_max_tokens
        tokens_input = tokens_input + num_tokens_from_string(message, 'gpt-3.5-turbo') + 1 + model_generation_max_tokens
    cost_input = input.cost(tokens_input, 'gpt-3.5-turbo')
    cost_output = output.cost(tokens_output, 'gpt-3.5-turbo')
    cost = cost_input + cost_output
    return cost

def approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations, functions):
    tokens_input = 0
    tokens_output = 0
    if method == 'classification.Classification' or method == 'equal.Equal' or method == 'includes.Includes':
        for test_case in test_cases:

            for i in range(number_of_prompts):
                tokens_input = tokens_input + 3 + model_generation_max_tokens + num_tokens_from_string(test_case[0], model_test) 
                tokens_output = tokens_output + model_test_max_tokens
    if method == 'function_calling.functionCalling':
        tokens_functions = 0
        for function in functions:
            tokens_functions = tokens_functions + 150
        for test_case in test_cases:
            for i in range(number_of_prompts):
                tokens_input = tokens_input + 3 + model_generation_max_tokens + num_tokens_from_string(test_case[0], model_test) + tokens_functions
                tokens_output = tokens_output + model_test_max_tokens

    if method == 'elovalue.Elo':
        for test_case in test_cases:
            tokens_input = tokens_input + num_tokens_from_string("""Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.

            You will be provided with the task description, the test prompt, and two generations - one for each system prompt.

            Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.

            Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.

            Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.

            Respond with your ranking, and nothing else. Be fair and unbiased in your judgement.""", 'gpt-3.5-turbo') + 2*model_test_max_tokens + 2*num_tokens_from_string(test_case, 'gpt-3.5-turbo') + model_generation_max_tokens
            tokens_output = tokens_output + model_generation_max_tokens + len(test_cases) * len(prompts_value) * (len(prompts_value) - 1) // 2
    cost_input = input.cost(tokens_input, model_test)
    cost_output = output.cost(tokens_output, model_test)
    cost = cost_input + cost_output
    return cost

def approximate_cost_iterations(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_iteration, model_iteration_max_tokens, iterations, functions):
    cost = 0
    if method == 'equal.Equal' or method == 'includes.Includes' or method == 'classification.Classification' or method == 'function_calling.functionCalling':
        while iterations > 0:
            cost = approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts - 2, 'None', model_iteration, model_iteration_max_tokens, iterations) + approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts - 2, 'None', model_iteration, model_iteration_max_tokens, iterations, functions)
            iterations = iterations - 1
    else:
        while iterations > 0:
            cost = approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, 'None', model_iteration, model_iteration_max_tokens, iterations) + approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, 'None', model_iteration, model_iteration_max_tokens, iterations, functions)
            iterations = iterations - 1
    return cost

def approximate_cost(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, model_generation, model_generation_max_tokens, iterations, functions, prompt_change=None, description=None, model_iteration=None, model_iteration_max_tokens=None):
    return approximate_cost_change(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations) + approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations) + approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations, functions) + approximate_cost_iterations(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_iteration, model_iteration_max_tokens, iterations, functions)