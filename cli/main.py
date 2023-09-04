import argparse
import os
import json
import matplotlib.pyplot as plt
from cli.prompt_generation import generation, iteration
from cli.evals import elovalue, classification, equals, includes, function_calling
from cli.approximate_cost import cost
import yaml
import textwrap
import numpy as np
from collections import defaultdict
import dotenv
from cli.validation_yaml import validation
from pathlib import Path
from cli.constants import constants
import openai

def valid_yaml(file_name):
    
    # initialize validation
    valid = True

    # Open the YAML file and load its content
    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)

    # Check if 'method' is defined in 'test' section
    if 'method' not in content['test']:
        valid = False
        print("Validation error:")
        print({'test': {'method': ['Must be defined']}}) # Notify about the missing method
    
    # Handle prompts related information
    if 'generation' in content['prompts']:
        if 'number' in content['prompts']['generation']:
            number_of_prompts = content['prompts']['generation']['number']
        if not 'number' in content['prompts']['generation']:
            number_of_prompts = 4
    if 'list' in content['prompts']:
        number_of_prompts = len(content['prompts']['list'])

    # Handle prompts iteration and best_prompts validation
    if 'iterations' in content['prompts']:
        if 'best_prompts' in content['prompts']['iterations']:
            best_prompts = content['prompts']['iterations']['best_prompts']
        if not 'best_prompts' in content['prompts']['iterations']:
            best_prompts = 2
        if best_prompts < 2 or best_prompts > number_of_prompts:
            valid = False
            print("Validation error:")
            print({'prompts': {'iterations': {'best_prompts':{'test': {'method': ['best_prompts has to be greater than or equal to 2 and strictly less than number_of_prompts.']}}}}})
            return valid
    # Allowed test method names
    allowed_names = ['function_calling', 'Classification', 'Equals', 'Includes', 'Elo']

    # Check if the selected 'method' is valid
    if 'method' in content['test']:
        if content['test']['method'] == 'function_calling':
            config_schema = validation.ValidationFunctionCalling()

        if content['test']['method'] == 'Classification' or content['test']['method'] == 'Equals' or content['test']['method'] == 'Includes':
            config_schema = validation.ValidationClaEqIn()

        if content['test']['method'] == 'Elo':
            config_schema = validation.ValidationElo()
        
        # Check if the selected method is in the allowed names
        if content['test']['method'] not in allowed_names:
            valid = False
            error_message = f"Must be one of the following: {', '.join(allowed_names)}"
            print("Validation error:")
            print({'test': {'method': [error_message]}})

        # Validate content using the appropriate schema
        if content['test']['method'] in allowed_names:
            errors = config_schema.validate(content)
            if not errors:
                validated_object = config_schema.load(content)
                print("Successful validation. Validated object:")
                print(validated_object)
            if errors:
                print("Validation errors:")
                print(errors)
            if errors != {}:
                valid = False

    return valid

def read_yaml(file_name):
    """
    It loads and reads the content of a given YAML file and returns its content as a Python dictionary or list.
    :type file_name: path of the yaml file.
    :return: yaml file as an object.
    """
    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)
    return content

def approximate_cost(file):
    # Extract the absolute path of the YAML file from the input 'file' object
    yaml_file_path = os.path.abspath(file)
    # Read the content of the YAML file using the 'read_yaml' function
    yaml_content = read_yaml(yaml_file_path)

    # Extract essential information from the YAML content
    method = yaml_content['test']['method']
    if method == 'Elo':
        description = yaml_content['test']['description']
    test_cases = yaml_content.get('test', {}).get('cases', [])
    if method == 'Classification' or method == 'Includes' or method == 'Equals':
        input_output_pairs = [(case['inout'], case['output']) for case in test_cases]
        test_cases = input_output_pairs
    if method == 'function_calling':
        result_list = [[case['inout'], case['output1'], case['output2']] for case in test_cases]
        test_cases = result_list
    model_test = yaml_content['test']['model']['name']
    model_test_max_tokens = int(yaml_content['test']['model']['max_tokens'])
    if 'list' in yaml_content['prompts']:
        prompts_value = yaml_content['prompts']['list']
        number_of_prompts = len(prompts_value)
    if not 'list' in yaml_content['prompts']:
        prompts_value = []
    if 'generation' in yaml_content['prompts']:
        if 'number' in yaml_content['prompts']['generation']:
            number_of_prompts = int(yaml_content['prompts']['generation']['number'])
        if not 'number' in yaml_content['prompts']['generation']:
            number_of_prompts = 4
        if 'constraints' in yaml_content['prompts']['generation']:
            prompt_constrainst = yaml_content['prompts']['generation']['constraints']
        if not 'constraints' in yaml_content['prompts']['generation']:
            prompt_constrainst = 'None'
        model_generation = yaml_content['prompts']['generation']['model']['name']
        model_generation_max_tokens = int(yaml_content['prompts']['generation']['model']['max_tokens'])
        if method != 'Elo':
            description = yaml_content['prompts']['generation']['description']
    if not 'generation' in yaml_content['prompts']:
        model_generation = 'gpt-4'
        model_generation_max_tokens = 0
        prompt_constrainst = 'None'
        description = None
    if 'functions' in yaml_content['test']:
        functions = yaml_content['test']['functions']
    if 'iterations' in yaml_content['prompts']:
        iterations = int(yaml_content['prompts']['iterations']['number'])
        if 'model' in yaml_content['prompts']['iterations']:
            model_iteration = yaml_content['prompts']['iterations']['model']['name']
            model_iteration_max_tokens = yaml_content['prompts']['iterations']['model']['max_tokens']
        if not 'model' in yaml_content['prompts']['iterations']:
            model_iteration = 'None'
            model_iteration_max_tokens = 0
        if 'best_prompts' in yaml_content['prompts']['iterations']:
            best_prompts = yaml_content['prompts']['iterations']['best_prompts']
        if not 'best_prompts' in yaml_content['prompts']['iterations']:
            best_prompts = 2
    if not 'iterations' in yaml_content['prompts']:
        iterations = 0
        model_iteration = 'None'
        model_iteration_max_tokens = 0
        best_prompts = 2

    # Calculate approximate cost based on the extracted information and the 'cost' module
    if method == 'function_calling':
        approximate_cost = cost.approximate_cost(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, model_generation, model_generation_max_tokens, iterations, functions, prompt_constrainst, description, model_iteration, model_iteration_max_tokens, best_prompts)
    if method == 'Elo' or method == 'Classification' or method == 'Equals' or method == 'Includes':
        approximate_cost = cost.approximate_cost(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, model_generation, model_generation_max_tokens, iterations, None, prompt_constrainst, description, model_iteration, model_iteration_max_tokens, best_prompts)
    return approximate_cost

def run_evaluation(file, approximate_cost):

    # Extract the absolute path of the YAML file from the input 'file' object
    yaml_file_path = os.path.abspath(file)
    # Read the content of the YAML file using the 'read_yaml' function
    yaml_content = read_yaml(yaml_file_path)

    # Extract essential information from the YAML content
    method = yaml_content['test']['method']
    if method == 'Elo':
        description = yaml_content['test']['description']
    test_cases = yaml_content.get('test', {}).get('cases', [])
    
    model_test = yaml_content['test']['model']['name']
    model_test_max_tokens = int(yaml_content['test']['model']['max_tokens'])
    model_test_temperature = int(yaml_content['test']['model']['temperature'])
    if 'list' in yaml_content['prompts']:
        prompts_value = yaml_content['prompts']['list']
        number_of_prompts = len(prompts_value)
    if not 'list' in yaml_content['prompts']:
        prompts_value = []
    if 'generation' in yaml_content['prompts']:
        if 'number' in yaml_content['prompts']['generation']:
            number_of_prompts = int(yaml_content['prompts']['generation']['number'])
        if not 'number' in yaml_content['prompts']['generation']:
            number_of_prompts = 4
        if 'constraints' in yaml_content['prompts']['generation']:
            prompt_constrainst = yaml_content['prompts']['generation']['constraints']
        if not 'constraints' in yaml_content['prompts']['generation']:
            prompt_constrainst = 'None'
        model_generation = yaml_content['prompts']['generation']['model']['name']
        model_generation_max_tokens = int(yaml_content['prompts']['generation']['model']['max_tokens'])
        model_generation_temperature = int(yaml_content['prompts']['generation']['model']['temperature'])
        if method != 'Elo':
            description = yaml_content['prompts']['generation']['description']
    if not 'generation' in yaml_content['prompts']:
        model_generation = 'gpt-4'
        model_generation_max_tokens = 0
        prompt_constrainst = None
        if method != 'Elo':
            description = None
        model_generation_temperature = 0
    if 'functions' in yaml_content['test']:
        functions = yaml_content['test']['functions']
        function_call = yaml_content['test']['function_call']
    if 'iterations' in yaml_content['prompts']:
        iterations = int(yaml_content['prompts']['iterations']['number'])
        if 'model' in yaml_content['prompts']['iterations']:
            model_iteration = yaml_content['prompts']['iterations']['model']['name']
            model_iteration_max_tokens = yaml_content['prompts']['iterations']['model']['max_tokens']
            model_iteration_temperature = yaml_content['prompts']['iterations']['model']['temperature']
        if not 'model' in yaml_content['prompts']['iterations']:
            model_iteration = 'None'
            model_iteration_max_tokens = 0
        if 'best_prompts' in yaml_content['prompts']['iterations']:
            best_prompts = yaml_content['prompts']['iterations']['best_prompts']
        if not 'best_prompts' in yaml_content['prompts']['iterations']:
            best_prompts = 2
    if not 'iterations' in yaml_content['prompts']:
        iterations = 0
        best_prompts = 2

        
    cost = 0
    tokens_input_gpt4 = 0
    tokens_output_gpt4 = 0
    tokens_input_gpt35 = 0
    tokens_output_gpt35 = 0

    # Determine the class corresponding to the selected method
    if method == 'Elo':
        class_method = elovalue.Elo
    if method == 'Classification':
        class_method = classification.Classification
    if method == 'Equals':
        class_method = equals.Equals
    if method == 'Includes':
        class_method = includes.Includes
    if method == 'function_calling':
        class_method = function_calling.functionCalling

    # Initialize an object of the class obtained from the 'method'
    if method != 'function_calling' and method != 'Elo':
        object_class = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, best_prompts)
    if method == 'function_calling':
        object_class = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, functions, function_call, best_prompts)
    if method == 'Elo':
        object_class = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, best_prompts)

    # Checks if prompts exist, and generates them if necessary
    if prompts_value == []:
        if prompt_constrainst != 'None':
            prompts_generation_cost = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, description, object_class.model_generation, object_class.model_generation_temperature, model_generation_max_tokens, object_class.number_of_prompts, prompt_constrainst)
            prompts_value = prompts_generation_cost[0]
            cost = cost + prompts_generation_cost[1]
            if model_generation == 'gpt-3.5-turbo':
                tokens_input_gpt35 = tokens_input_gpt35 + prompts_generation_cost[2]
                tokens_output_gpt35 = tokens_output_gpt35 + prompts_generation_cost[3]
            elif model_generation == 'gpt-4':
                tokens_input_gpt4 = tokens_input_gpt4 + prompts_generation_cost[2]
                tokens_output_gpt4 = tokens_output_gpt4 + prompts_generation_cost[3]
        else:
            prompts_generation_cost = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, description, object_class.model_generation, object_class.model_generation_temperature, model_generation_max_tokens, object_class.number_of_prompts)
            prompts_value = prompts_generation_cost[0]
            cost = cost + prompts_generation_cost[1]
            if model_generation == 'gpt-3.5-turbo':
                tokens_input_gpt35 = tokens_input_gpt35 + prompts_generation_cost[2]
                tokens_output_gpt35 = tokens_output_gpt35 + prompts_generation_cost[3]
            elif model_generation == 'gpt-4':
                tokens_input_gpt4 = tokens_input_gpt4 + prompts_generation_cost[2]
                tokens_output_gpt4 = tokens_output_gpt4 + prompts_generation_cost[3]
    
    # Initialize an object for evaluation
    if method == 'function_calling':
        evaluable_object = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, functions, function_call, best_prompts)
    if method == 'Elo':
        evaluable_object = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, best_prompts)
    if method != 'function_calling' and method != 'Elo':
        evaluable_object = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, best_prompts)

    # Evaluate the prompts and gather results
    results = evaluable_object.evaluate_optimal_prompt()
    cost = cost + results[2]
    if model_test == 'gpt-4':
        tokens_input_gpt4 = tokens_input_gpt4 + results[3]
        tokens_output_gpt4 = tokens_output_gpt4 + results[4]
    if model_test == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + results[3]
        tokens_output_gpt35 = tokens_output_gpt35 + results[4]
    yaml_folder = os.path.dirname(file)
    if method == 'Elo':
        # Group "elo" values by prompt using a dictionary
        elos_by_prompt = defaultdict(list)
        for item in results[0][number_of_prompts + 1]:
            prompt = item["prompt"]
            elo = item["elo"]
            elos_by_prompt[prompt].append(elo)

        # Create a scatter plot
        for prompt, elos in elos_by_prompt.items():
            prompt_truncated = textwrap.shorten(prompt, width=20, placeholder="...")
            x = np.arange(1, len(elos) + 1)
            y = np.array(elos)
            x_smooth = np.linspace(x.min(), x.max(), 200)
            y_smooth = np.interp(x_smooth, x, y)
            plt.plot(x_smooth, y_smooth, linewidth=1.5, markersize=6, label=prompt_truncated)
        plt.xlabel('Comparisons')
        plt.ylabel('Elo')
        plt.title('Scatter Plot: Elo by Prompt')
        plt.legend()
        plt.show()
        output_plot_path = os.path.join(yaml_folder, "scatter_plot.png")
        plt.savefig(output_plot_path)
        print(f"Scatter plot saved in: {output_plot_path}")

    # Full path of the output.json file in the same folder as the YAML
    output_json_path = os.path.join(yaml_folder, "output.json")
    # Convert the result to JSON format and save it to the output.json file
    with open(output_json_path, "w") as json_file:
        json.dump(results[0], json_file, indent=4)
    print(f"Result saved in: {output_json_path}")

    # Make iterations if it's necessary
    old_prompts = results[1]
    number_of_iteration = 1
    if method != 'Elo':
        tokens_input_gen = 0
        tokens_output_gen = 0
        tokens_input_test = 0
        tokens_output_test = 0
        while iterations > 0:
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            if method == 'function_calling':
                new_results_prompts_cost = iteration.iterations(test_cases, number_of_prompts - best_prompts, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, old_prompts, method, functions, function_call, best_prompts)
            if method == 'Elo':
                new_results_prompts_cost = iteration.iterations(test_cases, number_of_prompts - best_prompts, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, old_prompts, method, None, None, description, best_prompts)

            if method != 'function_calling' and method != 'Elo':
                new_results_prompts_cost = iteration.iterations(test_cases, number_of_prompts - best_prompts, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, old_prompts, method, None, None, best_prompts)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen = tokens_input_gen + new_results_prompts_cost[2]
            tokens_output_gen = tokens_output_gen + new_results_prompts_cost[4]
            tokens_input_test = tokens_input_test + new_results_prompts_cost[3]
            tokens_output_test = tokens_output_test + new_results_prompts_cost[5]
            with open(json_file_path, 'w') as file:
                json.dump(new_results[0], file, indent=4)
            iterations = iterations - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = [sorted_data[0], sorted_data[1]]
    if method == 'Elo':
        tokens_input_gen = 0
        tokens_output_gen = 0
        tokens_input_test = 0
        tokens_output_test = 0
        while iterations > 0:
            prompt_contents = [item['prompt'] for item in old_prompts]
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            new_results_prompts_cost = iteration.iterations(test_cases, number_of_prompts - best_prompts, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, prompt_contents, method, None, None, description, best_prompts)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen = tokens_input_gen + new_results_prompts_cost[2]
            tokens_output_gen = tokens_output_gen + new_results_prompts_cost[4]
            tokens_input_test = tokens_input_test + new_results_prompts_cost[3]
            tokens_output_test = tokens_output_test + new_results_prompts_cost[5]
            with open(json_file_path, 'w') as file:
                json.dump(new_results[0], file, indent=4)

            elos_by_prompt = defaultdict(list)
            for item in new_results[0][number_of_prompts + 1]:
                prompt = item["prompt"]
                elo = item["elo"]
                elos_by_prompt[prompt].append(elo)

            # Create a scatter plot
            for prompt, elos in elos_by_prompt.items():
                prompt_truncated = textwrap.shorten(prompt, width=20, placeholder="...")
                x = np.arange(1, len(elos) + 1)
                y = np.array(elos)
                x_smooth = np.linspace(x.min(), x.max(), 200)
                y_smooth = np.interp(x_smooth, x, y)
                plt.plot(x_smooth, y_smooth, linewidth=1.5, markersize=6, label=prompt_truncated)
            plt.xlabel('Comparisons')
            plt.ylabel('Elo')
            plt.title('Scatter Plot: Elo by Prompt')
            plt.legend()
            plt.show()
            scatter = f'scatter_plot_{number_of_iteration}.png'
            output_plot_path = os.path.join(yaml_folder, scatter)
            plt.savefig(output_plot_path)
            print(f"Scatter plot saved in: {output_plot_path}")

            iterations = iterations - 1
            number_of_iteration = number_of_iteration + 1
            old_prompts = new_results[1]
    # Ends the calculation of consumed tokens and stores this information

    if model_generation == 'gpt-4':
        tokens_input_gpt4 = tokens_input_gpt4 + tokens_input_gen
        tokens_output_gpt4 = + tokens_output_gpt4 + tokens_output_gen
    if model_generation == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + tokens_input_gen
        tokens_output_gpt35 = + tokens_output_gpt35 + tokens_output_gen
    if model_test == 'gpt-4':
        tokens_input_gpt4 = tokens_input_gpt4 + tokens_input_test
        tokens_output_gpt4 = + tokens_output_gpt4 + tokens_output_test
    if model_test == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + tokens_input_test
        tokens_output_gpt35 = + tokens_output_gpt35 + tokens_output_test
    
    filename = f'output_best_prompts_and_results.json'
    json_file_path = os.path.join(yaml_folder, filename)
    tokens_and_cost = {
        "approximate_cost": approximate_cost,
        "real_cost": cost,
        "tokens_input_gpt-3.5-turbo": tokens_input_gpt35,
        "tokens_output_gpt-3.5-turbo": tokens_output_gpt35,
        "tokens_input_gpt-4": tokens_input_gpt4,
        "tokens_output_gpt-4": tokens_output_gpt4
    }
    old_prompts.append(tokens_and_cost)
    with open(json_file_path, 'w') as file:
        json.dump(old_prompts, file, indent=4)

    print(f"The cost of your evaluation was: {cost} dollars.")
    

def main():
    
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Path of the YAML file to read.\n The following is the structure that your YAML files should have.\n"

"""test:\n"""

    """cases: Here, you have to put the test cases you are going to use to evaluate your prompts. If you are going to use the
        Elo method to evaluate them, it should be just a list of strings. If you are going to use the methods classification, 
        equal or includes, it should be a list of tuples with two elements, where the first element is the test case and the 
        second element is the correct response to the test. Remember that if you decide to use classification, only a boolean
        value is allowed as a response. the form of your test cases has to be, in case of selecting the Elo method:\n
            -'Test1'\n
            -'Test2'...\n
        If you choose the methods Classification, Equals, Includes they must be of the form:\n
            -inout: 'Test1'\n
            output: 'Answer1'\n
            -inout: 'Test2'\n
            output: 'Answer2'\n
        And in case the method is function_calling:\n
            -inout: 'Test1'\n
            output1: 'name_function'\n
            output2: 'variable'\n
            -inout: 'Test2'\n
            output1: 'name_function'\n
            output2: 'variable'\n"""

    """description: Here is the description of the type of task that summarizes the test cases. You only have to use this field if 
        you are going to use the 'Elo' method.\n"""
    """method: Here, you select the evaluation method for your prompts. You must choose between 'Elo',
        'Classification', 'Equals', 'includes' and 'function_calling'.\n"""

    """model:\n"""
        """name: The name of the GPT model you will use to evaluate the prompts.\n"""
        """temperature: The temperature of the GPT model you will use to evaluate the prompts.\n"""
        """max_tokens: The maximum number of tokens you will allow the GPT model to use to generate the response to the test.\n"""

    """functions: This field must only be filled out in case the 'function_calling' method is intended to be used.
    If another method is used, it must not be filled out. The structure is a JSON object. Let's break down the different components:\n

            - Function Name (name): This is the identifier used to refer to this function within the context of your code.\n

            - Function Description (description): A brief description of what the function does.\n

            - Function Parameters (parameters): This section defines the input parameters that the function accepts.\n

                - Type (type): The type of the parameter being defined.\n

                - Properties (properties): This is an object containing properties that the input parameter object should have.\n

                    - File Type (file_type): This is a property of the parameter object.\n

                    - Enum (enum): An enumeration of allowed values for the 'file_type' property. (optional)\n

                    - Description (description): A description of what the 'file_type' property represents.\n

                - Required (required): An array listing the properties that are required within the parameter object. (optional)\n"""

    """function_call: This field must only be filled out in case the 'function_calling' method is intended to be 
        used. If another method is used, it must not be filled out.\n"""

"""prompts: You have two options, either provide your list of prompts or generate them following the instructions below.\n"""

    """list: A list of prompts you want to evaluate. If you want to generate them with the prompt generator, don't put this key in 
        your YAML file. Please provide a minimum number of 4 prompts. Your prompts must be listed as follows:\n
            - 'Prompt1'\n
            - 'Prompt2'...\n"""

    """generation:\n"""

        """number: The number of prompts you are going to evaluate. You need to provide this key value only if you are going to generate the prompts.
            Indicate the quantity of prompts you want to generate. Please provide a minimum number of 4 prompts. If you do not 
            define this key by default, 4 prompts will be created.\n"""
        """constraints: If you are going to generate prompts, this optional feature allows you to add special characteristics to the 
            prompts that will be generated. For example, if you want prompts with a maximum length of 50 characters, simply complete with 
            'Generate prompts with a maximum length of 50 characters'. If you don't want to use it, you don't need to have this key 
            defined.\n"""
        """description: Here is the description of the type of task that summarizes the test cases.\n"""

        """model:\n"""

            """name: The name of the GPT model you will use to generate the prompts.\n"""
            """temperature: The temperature of the GPT model you will use to generate the prompts.\n"""
            """max_tokens: The maximum number of tokens you will allow the GPT model to use to generate your prompts.\n"""

    """iterations:\n (optional)"""
        """number: The number of iterations you want to perform on the best prompts obtained in your initial testing to arrive at 
            prompts with better final results. If you don't want to try alternatives combining your best prompts just put 0.\n"""
        """best_prompts: The number of prompts you want to iterate over. the value must be between 2 and the number of prompts you 
            provide (or generate) minus one. If you do not define this value but do want to iterate, the default value will be 2.\n"""

        """model:\n"""

            """name: The name of the GPT model you will use to generate the prompts.\n"""
            """temperature: The temperature of the GPT model you will use to generate the prompts.\n"""
            """max_tokens: The maximum number of tokens you will allow the GPT model to use to generate your prompts.""")
    
    parser.add_argument("optional_string", help="Optional string parameter.", nargs='?', default=None)
    parser.add_argument("--env_path", help="Path to the .env file.", default=None)
    args = parser.parse_args()

    try:
        # Define a function to check if a given path is valid
        def valid_path(path):
            try:
                Path(path)
                return True
            except (ValueError, TypeError):
                return False
        # Check if the provided environment path is valid
        if valid_path(args.env_path):
            # Load environment variables from the specified path
            dotenv.load_dotenv(dotenv_path=args.env_path)
        else:
            # Load environment variables from the default '.env' file in the current working directory
            dotenv.load_dotenv(dotenv_path=os.getcwd()+'/.env')
        if os.getenv(constants.OPENAI_API_KEY):
            openai.api_key = os.getenv(constants.OPENAI_API_KEY)
        else:
            raise Exception("No API key provided into enviroment variables, please configure OPENAI_API_KEY")
        
    except FileNotFoundError:
        # Handle the case where the .env file is not found
        print("Error: The .env file was not found in your directory.")
        exit(1)

    # Check if the OPENAI_API_TYPE environment variable is set to "azure"
    if os.getenv("OPENAI_API_TYPE") == "azure":

        # Check if OPENAI_API_BASE is not defined when using the "azure" API type
        if os.getenv("OPENAI_API_BASE") is None:
            print("Error: OPENAI_API_BASE is required when OPENAI_API_TYPE is 'azure'")
            exit(1)

        # Check if OPENAI_API_VERSION is not defined when using the "azure" API type
        if os.getenv("OPENAI_API_VERSION") is None:
            print("Error: OPENAI_API_VERSION is required when OPENAI_API_TYPE is 'azure'")
            exit(1)
    if os.getenv("OPENAI_API_BASE") != None and os.getenv("OPENAI_API_TYPE") != None and os.getenv("OPENAI_API_VERSION") != None:
        openai_env = {
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        constants.OPENAI_API_KEY: os.getenv(constants.OPENAI_API_KEY),
        "OPENAI_API_TYPE": os.getenv("OPENAI_API_TYPE"),
        "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION")
        }
    if not (os.getenv("OPENAI_API_BASE") != None and os.getenv("OPENAI_API_TYPE") != None and os.getenv("OPENAI_API_VERSION") != None):
        openai_env = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        }

    # Check if the optional string argument is set to "don't run"
    if args.optional_string == "don't run":
        # Check if the provided YAML file is valid
        if (valid_yaml(args.yaml_file)):
            # Calculate the approximate cost of the evaluation based on the YAML file
            approximate = approximate_cost(args.yaml_file)
            print(f"The cost of your evaluation will be approximately {approximate} dollars.")
    else:
        # Check if the provided YAML file is valid
        if (valid_yaml(args.yaml_file)):
            # Calculate the approximate cost of the evaluation based on the YAML file
            approximate = approximate_cost(args.yaml_file)
            print(f"The cost of your evaluation will be approximately {approximate} dollars.")
            run_evaluation(args.yaml_file, approximate)

if __name__ == "__main__":
    main()