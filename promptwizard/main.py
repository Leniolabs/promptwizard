import argparse
import os
import json
import matplotlib.pyplot as plt
from promptwizard.prompt_generation import generation, iteration
from promptwizard.evals import elovalue, classification, equals, includes, function_calling, code_generation, json_validation, semantic_similarity, logprobs
from promptwizard.approximate_cost import cost
import yaml
import textwrap
import numpy as np
from collections import defaultdict
import dotenv
from promptwizard.validation_yaml import validation
from pathlib import Path
from promptwizard.constants import constants
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
        if best_prompts < 2 or best_prompts >= number_of_prompts:
            valid = False
            print("Validation error:")
            print({'prompts': {'iterations': {'best_prompts':['best_prompts has to be greater than or equal to 2 and strictly less than number_of_prompts.']}}})
            return valid
    # Allowed test method names
    allowed_names = ['Function Calling', 'Classification', 'Equals', 'Includes', 'Elo', 'Code Generation', 'JSON Validation', 'Semantic Similarity', 'LogProbs']

    # Check if the selected 'method' is valid
    if 'method' in content['test']:
        if content['test']['method'] == 'Function Calling':
            config_schema = validation.ValidationFunctionCalling()

        if content['test']['method'] == 'Classification' or content['test']['method'] == 'Equals' or content['test']['method'] == 'Includes':
            config_schema = validation.ValidationClaEqIn()

        if content['test']['method'] == 'Elo':
            config_schema = validation.ValidationElo()

        if content['test']['method'] == 'Code Generation':
            config_schema = validation.ValidationCode()

        if content['test']['method'] == 'JSON Validation':
            config_schema = validation.ValidationJSON()

        if content['test']['method'] == 'Semantic Similarity':
            config_schema = validation.ValidationEmbeddings()

        if content['test']['method'] == 'LogProbs':
            config_schema = validation.ValidationLogProbs()
        
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
    if method == 'Classification' or method == 'Includes' or method == 'Equals' or method == 'JSON Validation' or method == 'Semantic Similarity' or method == 'LogProbs':
        input_output_pairs = [(case['input'], case['output']) for case in test_cases]
        test_cases = input_output_pairs
    if method == 'Function Calling':
        result_list = [[case['input'], case['output1'], case['output2']] for case in test_cases]
        test_cases = result_list
    if method == 'Code Generation':
        result_list = [[case['input'], case['arguments'], case['output']] for case in test_cases]
        test_cases = result_list
    if method == 'Semantic Similarity':
        if 'embeddings' in yaml_content['test']:
            model_embedding = yaml_content['test']['embeddings']['model_name']
        if not 'embeddings' in yaml_content['test']:
            model_embedding = "text-embedding-ada-002"
    if not method == 'Semantic Similarity':
        model_embedding = None
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
            model_iteration = model_generation
            model_iteration_max_tokens = model_generation_max_tokens
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
    if method == 'Function Calling':
        approximate_cost = cost.approximate_cost(test_cases, method, prompts_value, model_test, model_test_max_tokens, number_of_prompts, model_generation, model_generation_max_tokens, iterations, functions, prompt_constrainst, model_iteration, model_iteration_max_tokens, best_prompts, model_embedding)
    if method == 'Elo' or method == 'Classification' or method == 'Equals' or method == 'Includes' or method == 'Code Generation' or method == 'JSON Validation' or method =='LogProbs':
        approximate_cost = cost.approximate_cost(test_cases, method, prompts_value, model_test, model_test_max_tokens, number_of_prompts, model_generation, model_generation_max_tokens, iterations, None, prompt_constrainst, model_iteration, model_iteration_max_tokens, best_prompts, model_embedding)
    
    if method == 'Semantic Similarity':
        approximate_cost = cost.approximate_cost(test_cases, method, prompts_value, model_test, model_test_max_tokens, number_of_prompts, model_generation, model_generation_max_tokens, iterations, None, prompt_constrainst, model_iteration, model_iteration_max_tokens, best_prompts, model_embedding)

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

    if method == 'Semantic Similarity':
        if 'embeddings' in yaml_content['test']:
            model_embedding = yaml_content['test']['embeddings']['model_name']
        if not 'embeddings' in yaml_content['test']:
            model_embedding = "text-embedding-ada-002"
            print("model_embedding will be 'text-embedding-ada-002'.")
    
    model_test = yaml_content['test']['model']['name']
    model_test_max_tokens = int(yaml_content['test']['model']['max_tokens'])
    model_test_temperature = int(yaml_content['test']['model']['temperature'])
    if 'best_prompts' in yaml_content['prompts']:
        best_prompts = yaml_content['prompts']['best_prompts']
    if not 'best_prompts' in yaml_content['prompts']:
        best_prompts = 2
        print("best_prompts will be 2.")
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
        if 'function_call' in yaml_content['test']:
            function_call = yaml_content['test']['function_call']
        else:
            function_call = 'auto'
    if 'iterations' in yaml_content['prompts']:
        iterations = int(yaml_content['prompts']['iterations']['number'])
        if 'model' in yaml_content['prompts']['iterations']:
            model_iteration = yaml_content['prompts']['iterations']['model']['name']
            model_iteration_max_tokens = yaml_content['prompts']['iterations']['model']['max_tokens']
            model_iteration_temperature = yaml_content['prompts']['iterations']['model']['temperature']
        if not 'model' in yaml_content['prompts']['iterations']:
            if 'generation' in yaml_content['prompts']:
                model_iteration = model_generation
                model_iteration_max_tokens = model_generation_max_tokens
                model_iteration_temperature = model_generation_temperature
                print("You will use the information from the model that you used for your first generation to generate your iteration prompts.")
            else:
                model_iteration = 'gpt-4'
                model_iteration_max_tokens = 300
                model_iteration_temperature = 0.6
                print("You will generate prompts with gpt-4, max_tokens = 300 and temperature = 0.6.")
        if 'best_percentage' in yaml_content['prompts']['iterations']:
            best_percentage = yaml_content['prompts']['iterations']['best_percentage']
        if not ('best_percentage' in yaml_content['prompts']['iterations']) and (method != 'Elo' or method != 'Semantic Similarity'):
            best_percentage = 100
            if method != 'Elo' and method != 'Semantic Similarity':
                print("The percentage to be overcome by your best prompts to stop the iteration will be 100%.")
    if not 'iterations' in yaml_content['prompts']:
        iterations = 0
        best_percentage = 100

    if 'timeout' in yaml_content:
        timeout = yaml_content['timeout']
    if not 'timeout' in yaml_content:
        timeout = 10
    if 'n_retries' in yaml_content:
        n_retries = yaml_content['n_retries']
    if not 'n_retries' in yaml_content:
        n_retries = 5

        
    cost = 0
    tokens_input_gpt4 = 0
    tokens_output_gpt4 = 0
    tokens_input_gpt35 = 0
    tokens_output_gpt35 = 0
    tokens_embeddings = 0

    # Determine the class corresponding to the selected method
    if method == 'Elo':
        class_method = elovalue.Elo

    if method == 'Classification':
        class_method = classification.Classification

    if method == 'Equals':
        class_method = equals.Equals

    if method == 'Includes':
        class_method = includes.Includes

    if method == 'Function Calling':
        class_method = function_calling.functionCalling

    if method == 'Code Generation':
        class_method = code_generation.codeGeneration

    if method == 'JSON Validation':
        class_method = json_validation.jsonValidation

    if method == 'Semantic Similarity':
        class_method = semantic_similarity.semanticSimilarity

    if method == 'LogProbs':
        class_method = logprobs.LogProbs

    # Initialize an object of the class obtained from the 'method'
    if method != 'Function Calling' and method != 'Elo' and method != 'Semantic Similarity':
        object_class = class_method(test_cases, None, model_test, model_test_temperature, model_test_max_tokens, best_prompts, timeout, n_retries)
    
    if method == 'Semantic Similarity':
        object_class = class_method(test_cases, None, model_test, model_test_temperature, model_test_max_tokens, model_embedding, best_prompts, timeout, n_retries)

    if method == 'Function Calling':
        object_class = class_method(test_cases, None, functions, model_test, model_test_temperature, model_test_max_tokens, best_prompts, function_call, timeout, n_retries)
    if method == 'Elo':
        object_class = class_method(test_cases, None, description, model_test, model_test_temperature, model_test_max_tokens, best_prompts, timeout, n_retries)

    # Checks if prompts exist, and generates them if necessary
    if prompts_value == []:
        if prompt_constrainst != 'None':
            prompts_generation_cost = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, description, model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts, prompt_constrainst, timeout, n_retries)
            prompts_value = prompts_generation_cost[0]
            cost = cost + prompts_generation_cost[1]
            if model_generation == 'gpt-3.5-turbo':
                tokens_input_gpt35 = tokens_input_gpt35 + prompts_generation_cost[2]
                tokens_output_gpt35 = tokens_output_gpt35 + prompts_generation_cost[3]
            elif model_generation == 'gpt-4':
                tokens_input_gpt4 = tokens_input_gpt4 + prompts_generation_cost[2]
                tokens_output_gpt4 = tokens_output_gpt4 + prompts_generation_cost[3]
        else:
            prompts_generation_cost = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, description, model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts, prompt_constrainst, timeout, n_retries)
            prompts_value = prompts_generation_cost[0]
            cost = cost + prompts_generation_cost[1]
            if model_generation == 'gpt-3.5-turbo':
                tokens_input_gpt35 = tokens_input_gpt35 + prompts_generation_cost[2]
                tokens_output_gpt35 = tokens_output_gpt35 + prompts_generation_cost[3]
            elif model_generation == 'gpt-4':
                tokens_input_gpt4 = tokens_input_gpt4 + prompts_generation_cost[2]
                tokens_output_gpt4 = tokens_output_gpt4 + prompts_generation_cost[3]

    
    # Initialize an object for evaluation
    if method == 'Function Calling':
        evaluable_object = class_method(test_cases, prompts_value, functions, model_test, model_test_temperature, model_test_max_tokens, best_prompts, function_call, timeout, n_retries)
    if method == 'Elo':
        evaluable_object = class_method(test_cases, prompts_value, description, model_test, model_test_temperature, model_test_max_tokens, best_prompts, timeout, n_retries)

    if method == 'Semantic Similarity':
        evaluable_object = class_method(test_cases, prompts_value, model_test, model_test_temperature, model_test_max_tokens, model_embedding, best_prompts, timeout, n_retries)

    if method != 'Function Calling' and method != 'Elo' and method != 'Semantic Similarity':
        evaluable_object = class_method(test_cases, prompts_value, model_test, model_test_temperature, model_test_max_tokens, best_prompts, timeout, n_retries)

    # Evaluate the prompts and gather results
    results = evaluable_object.evaluate_optimal_prompt()
    cost = cost + results[2]
    if model_test == 'gpt-4':
        tokens_input_gpt4 = tokens_input_gpt4 + results[3]
        tokens_output_gpt4 = tokens_output_gpt4 + results[4]
    if model_test == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + results[3]
        tokens_output_gpt35 = tokens_output_gpt35 + results[4]
    if model_test == 'gpt-3.5-turbo-instruct':
        tokens_input_gpt35 = tokens_input_gpt35 + results[3]
        tokens_output_gpt35 = tokens_output_gpt35 + results[4]
    yaml_folder = os.path.dirname(file)

    if method == 'Semantic Similarity':
        tokens_embeddings = tokens_embeddings + results[5]
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
    if method != 'Elo' and method != 'Semantic Similarity':
        tokens_input_gen_iter = 0
        tokens_output_gen_iter = 0
        tokens_input_test_iter = 0
        tokens_output_test_iter = 0
        all_have_rating_percentage = True
        # Iterate through the list of elements and check the 'rating' value.
        for element in old_prompts:
            if element["rating"] < best_percentage:
                all_have_rating_percentage = False
                break  # If an element with a different rating is found, stop the iteration.
        while (iterations > 0 and (not all_have_rating_percentage)):
            print(f"Prompts from iteration number {number_of_iteration}")
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            if method == 'Function Calling':
                new_results_prompts_cost = iteration.iterations(test_cases, method, old_prompts, number_of_prompts - best_prompts, functions, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, function_call, None, best_prompts, timeout, n_retries)

            if method != 'Function Calling':
                new_results_prompts_cost = iteration.iterations(test_cases, method, old_prompts, number_of_prompts - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, None, best_prompts, timeout, n_retries)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen_iter = tokens_input_gen_iter + new_results_prompts_cost[2]
            tokens_output_gen_iter = tokens_output_gen_iter + new_results_prompts_cost[4]
            tokens_input_test_iter = tokens_input_test_iter + new_results_prompts_cost[3]
            tokens_output_test_iter = tokens_output_test_iter + new_results_prompts_cost[5]
            with open(json_file_path, 'w') as file:
                json.dump(new_results[0], file, indent=4)
                print(f"Result saved in: {filename}")
            iterations = iterations - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = sorted_data[:best_prompts]
            all_have_rating_percentage = True
            for element in old_prompts:
                if element["rating"] < best_percentage:
                    all_have_rating_percentage = False
                    break  # If an element with a different rating is found, stop the iteration.

    if method == 'Semantic Similarity':
        tokens_input_gen_iter = 0
        tokens_output_gen_iter = 0
        tokens_input_test_iter = 0
        tokens_output_test_iter = 0
        tokens_embeddings_iter = 0
        while (iterations > 0):
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []

            new_results_prompts_cost = iteration.iterations(test_cases, method, old_prompts, number_of_prompts - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, None, best_prompts, model_embedding, timeout, n_retries)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen_iter = tokens_input_gen_iter + new_results_prompts_cost[2]
            tokens_output_gen_iter = tokens_output_gen_iter + new_results_prompts_cost[4]
            tokens_input_test_iter = tokens_input_test_iter + new_results_prompts_cost[3]
            tokens_output_test_iter = tokens_output_test_iter + new_results_prompts_cost[5]
            tokens_embeddings_iter = tokens_embeddings_iter + new_results_prompts_cost[6]
            with open(json_file_path, 'w') as file:
                json.dump(new_results[0], file, indent=4)
                print(f"Result saved in: {filename}")
            iterations = iterations - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = sorted_data[:best_prompts]
        tokens_embeddings = tokens_embeddings + tokens_embeddings_iter

    if method == 'Elo':
        tokens_input_gen_iter = 0
        tokens_output_gen_iter = 0
        tokens_input_test_iter = 0
        tokens_output_test_iter = 0
        while iterations > 0:
            print(f"Prompts from iteration number {number_of_iteration}")
            prompt_contents = [item['prompt'] for item in old_prompts]
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            new_results_prompts_cost = iteration.iterations(test_cases, method, prompt_contents, number_of_prompts - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, description, best_prompts, timeout, n_retries)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen_iter = tokens_input_gen_iter + new_results_prompts_cost[2]
            tokens_output_gen_iter = tokens_output_gen_iter + new_results_prompts_cost[4]
            tokens_input_test_iter = tokens_input_test_iter + new_results_prompts_cost[3]
            tokens_output_test_iter = tokens_output_test_iter + new_results_prompts_cost[5]
            with open(json_file_path, 'w') as file:
                json.dump(new_results[0], file, indent=4)
                print(f"Result saved in: {filename}")

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
        tokens_input_gpt4 = tokens_input_gpt4 + tokens_input_gen_iter
        tokens_output_gpt4 = + tokens_output_gpt4 + tokens_output_gen_iter
    if model_generation == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + tokens_input_gen_iter
        tokens_output_gpt35 = + tokens_output_gpt35 + tokens_output_gen_iter
    if model_test == 'gpt-4':
        tokens_input_gpt4 = tokens_input_gpt4 + tokens_input_test_iter
        tokens_output_gpt4 = + tokens_output_gpt4 + tokens_output_test_iter
    if model_test == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + tokens_input_test_iter
        tokens_output_gpt35 = + tokens_output_gpt35 + tokens_output_test_iter
    if model_test == 'gpt-3.5-turbo-instruct':
        tokens_input_gpt35 = tokens_input_gpt35 + tokens_input_test_iter
        tokens_output_gpt35 = + tokens_output_gpt35 + tokens_output_test_iter
    
    filename = f'output_best_prompts_and_results.json'
    json_file_path = os.path.join(yaml_folder, filename)
    if method == 'Semantic Similarity':
        tokens_and_cost = {
        "approximate_cost": approximate_cost,
        "real_cost": cost,
        "tokens_input_gpt-3.5-turbo": tokens_input_gpt35,
        "tokens_output_gpt-3.5-turbo": tokens_output_gpt35,
        "tokens_input_gpt-4": tokens_input_gpt4,
        "tokens_output_gpt-4": tokens_output_gpt4,
        "tokens_embeddings": tokens_embeddings
        }
    if method != 'Semantic Similarity':
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
    parser.add_argument("yaml_file", help="Path of the YAML file to read.")
    
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

    # Check if the provided YAML file is valid
    if (valid_yaml(args.yaml_file)):
        # Calculate the approximate cost of the evaluation based on the YAML file
        approximate = approximate_cost(args.yaml_file)
        print(f"The cost of your evaluation will be approximately {approximate} dollars.")
        user_input = input("Continue? (Y/N): ").strip().lower()
        if user_input == "y":
            run_evaluation(args.yaml_file, approximate)
        else:
            print("Execution aborted.")

if __name__ == "__main__":
    main()