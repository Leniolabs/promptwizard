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
    tokens_input = 0 # Initialize a variable to count input tokens
    tokens_output = 0 # Initialize a variable to count output tokens

    # Check the method type to determine the system-generated prompt
    if prompts_value == []:

        if method == 'Includes':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Specify in the prompts that you generate that they give a step-by-step response.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
            
        if method == 'Equals':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.
                 
Remember that the prompt should only allow the AI to answer the answer and nothing else. No explanation is necessary.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified. I repeat, do not include the test cases.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""

        if method == 'Classification':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        The prompts you will be generating will be for classifiers, with 'true' and 'false' being the only possible outputs.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
            
        if method == 'Elo':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

The prompts you will be generating will be for freeform tasks, such as generating a landing page headline, an intro paragraph, solving a math problem, etc.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""

        if method == 'function_calling':
            system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""

        # Create a message list based on the prompt change requirement
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

def approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations, functions):
    tokens_input = 0 # Initialize a variable to count input tokens
    tokens_output = 0 # Initialize a variable to count output tokens

    # Check the evaluation method to determine token counts
    if method == 'Classification' or method == 'Equals' or method == 'Includes':
        for test_case in test_cases:
            for i in range(number_of_prompts):
                # Calculate input token count for classification, equals, or includes methods
                tokens_input = tokens_input + 3 + model_generation_max_tokens + num_tokens_from_string(test_case[0], model_test) 
                tokens_output = tokens_output + model_test_max_tokens
    if method == 'function_calling':
        tokens_functions = 150*len(functions) # Assuming a constant token count for function descriptions
        for test_case in test_cases:
            for i in range(number_of_prompts):
                # Calculate input token count for function_calling method
                tokens_input = tokens_input + 3 + model_generation_max_tokens + num_tokens_from_string(test_case[0], model_test) + tokens_functions
                tokens_output = tokens_output + model_test_max_tokens

    if method == 'Elo':
        for test_case in test_cases:
            # Calculate input token count for Elo method
            tokens_input = tokens_input + num_tokens_from_string("""Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.

            You will be provided with the task description, the test prompt, and two generations - one for each system prompt.

            Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.

            Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.

            Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.

            Respond with your ranking, and nothing else. Be fair and unbiased in your judgement.""", 'gpt-3.5-turbo') + 2*model_test_max_tokens + 2*num_tokens_from_string(test_case, 'gpt-3.5-turbo') + model_generation_max_tokens
            tokens_output = tokens_output + model_generation_max_tokens + len(test_cases) * len(prompts_value) * (len(prompts_value) - 1) // 2
    
    # Calculate the cost of input tokens using the model's cost function
    cost_input = input.cost(tokens_input, model_test)

    # Calculate the cost of output tokens using the model's cost function
    cost_output = output.cost(tokens_output, model_test)

    # Calculate the total cost as the sum of input and output costs
    cost = cost_input + cost_output

    return cost # Return the calculated cost

def approximate_cost_iterations(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_iteration, model_iteration_max_tokens, iterations, functions, best_prompts):
    cost = 0 # Initialize a variable to store the total cost

    # Check the evaluation method to determine the cost calculation strategy
    if method == 'Equals' or method == 'Includes' or method == 'Classification' or method == 'function_calling':
        while iterations > 0:
            # Calculate the cost for the current iteration and add it to the accumulated cost
            cost = approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts -  best_prompts, 'None', model_iteration, model_iteration_max_tokens, iterations) + approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts - 2, 'None', model_iteration, model_iteration_max_tokens, iterations, functions) + cost
            iterations = iterations - 1 # Decrement the iteration count
    else:
        while iterations > 0:
            # Calculate the cost for the current iteration and add it to the accumulated cost
            cost = approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, 'None', model_iteration, model_iteration_max_tokens, iterations) + approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, 'None', model_iteration, model_iteration_max_tokens, iterations, functions) + cost
            iterations = iterations - 1 # Decrement the iteration count

    return cost # Return the total cost after all iterations

def approximate_cost(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, model_generation, model_generation_max_tokens, iterations, functions, prompt_change, description, model_iteration, model_iteration_max_tokens, best_prompts):
    return approximate_cost_generation(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations) + approximate_cost_test(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_generation, model_generation_max_tokens, iterations, functions) + approximate_cost_iterations(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_change, model_iteration, model_iteration_max_tokens, iterations, functions, best_prompts)