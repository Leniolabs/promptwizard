import argparse
import os
import openai
import json
import matplotlib.pyplot as plt
from . import generation, iteration
from .evals import elovalue, classification, equal, includes, function_calling
from .approximate_cost import cost
import yaml
import textwrap
import numpy as np
from collections import defaultdict
import dotenv
from .validation_yaml import validation

from marshmallow import Schema, fields, ValidationError, validates_schema

class EnvSchema(Schema):
    OPENAI_API_BASE = fields.Str(allow_none=True)
    OPENAI_API_KEY = fields.Str(required=True)
    OPENAI_API_TYPE = fields.Str(allow_none=True)
    OPENAI_API_VERSION = fields.Str(allow_none=True)

    @validates_schema
    def validate_related_fields(self, data, **kwargs):
        api_type = data.get('OPENAI_API_TYPE')
        api_base = data.get('OPENAI_API_BASE')
        api_version = data.get('OPENAI_API_VERSION')

        if api_type == 'azure':
            if api_base is None:
                raise ValidationError("OPENAI_API_BASE is required when OPENAI_API_TYPE is 'azure'")
            if api_version is None:
                raise ValidationError("OPENAI_API_VERSION is required when OPENAI_API_TYPE is 'azure'")


def valid_yaml(file_name):
    
    valid = True

    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)

    if 'method' not in content['test']:
        valid = False
        print("Validation error:")
        print({'test': {'method': ['Must be defined']}})
    
    if 'generation' in content['prompts']:
        if 'number' in content['prompts']['generation']:
            number_of_prompts = content['prompts']['generation']['number']
        if not 'number' in content['prompts']['generation']:
            number_of_prompts = 4
    if 'list' in content['prompts']:
        number_of_prompts = len(content['prompts']['list'])

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

    allowed_names = ['function_calling.functionCalling', 'classification.Classification', 'equal.Equal', 'includes.Includes', 'elovalue.Elo']
    if 'method' in content['test']:
        if content['test']['method'] == 'function_calling.functionCalling':
            config_schema = validation.ConfigSchema3()

        if content['test']['method'] == 'classification.Classification' or content['test']['method'] == 'equal.Equal' or content['test']['method'] == 'includes.Includes':
            config_schema = validation.ConfigSchema2()

        if content['test']['method'] == 'elovalue.Elo':
            config_schema = validation.ConfigSchema1()
        
        if content['test']['method'] not in allowed_names:
            valid = False
            error_message = f"Must be one of the following: {', '.join(allowed_names)}"
            print("Validation error:")
            print({'test': {'method': [error_message]}})

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
    # Extract the 'yaml_file' attribute from the input 'file' object
    yaml_file_path = os.path.abspath(file)
    # Read the content of the YAML file
    yaml_content = read_yaml(yaml_file_path)

    # Extract the 'description', 'test_cases', 'number_of_prompts', 'candidate_model', 'generation_model',
    # 'generation_model_temperature', 'generation_model_max_tokens', and 'method' from the YAML content
    method = yaml_content['test']['method']
    if method == 'elovalue.Elo':
        description = yaml_content['test']['description']
    test_cases = yaml_content.get('test', {}).get('cases', [])
    if method == 'classification.Classification' or method == 'includes.Includes' or method == 'equal.Equal':
        input_output_pairs = [(case['inout'], case['output']) for case in test_cases]
        test_cases = input_output_pairs
    if method == 'function_calling.functionCalling':
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
    if not 'iterations' in yaml_content['prompts']:
        iterations = 0
        model_iteration = 'None'
        model_iteration_max_tokens = 0

    
    if method == 'function_calling.functionCalling':
        approximate_cost = cost.approximate_cost(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, model_generation, model_generation_max_tokens, iterations, functions, prompt_constrainst, description, model_iteration, model_iteration_max_tokens)
    if method == 'elovalue.Elo' or method == 'classification.Classification' or method == 'equal.Equal' or method == 'includes.Includes':
        approximate_cost = cost.approximate_cost(test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, model_generation, model_generation_max_tokens, iterations, None, prompt_constrainst, description, model_iteration, model_iteration_max_tokens)
    return approximate_cost

def run_evaluation(file, approximate_cost):

    # Extract the 'yaml_file' attribute from the input 'file' object
    yaml_file_path = os.path.abspath(file)
    # Read the content of the YAML file
    yaml_content = read_yaml(yaml_file_path)

    # Extract the 'description', 'test_cases', 'number_of_prompts', 'candidate_model', 'generation_model',
    # 'generation_model_temperature', 'generation_model_max_tokens', and 'method' from the YAML content
    method = yaml_content['test']['method']
    if method == 'elovalue.Elo':
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
        description = yaml_content['prompts']['generation']['description']
    if not 'generation' in yaml_content['prompts']:
        model_generation = 'gpt-4'
        model_generation_max_tokens = 0
        prompt_constrainst = None
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

    if method == 'elovalue.Elo':
        class_method = elovalue.Elo
    if method == 'classification.Classification':
        class_method = classification.Classification
    if method == 'equal.Equal':
        class_method = equal.Equal
    if method == 'includes.Includes':
        class_method = includes.Includes
    if method == 'function_calling.functionCalling':
        class_method = function_calling.functionCalling

    # Initialize an object of the class obtained from the 'method'
    if method != 'function_calling.functionCalling' and method != 'elovalue.Elo':
        object_class = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, best_prompts)
    if method == 'function_calling.functionCalling':
        object_class = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, functions, function_call, best_prompts)
    if method == 'elovalue.Elo':
        object_class = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, best_prompts)
    # Checks if the prompts to evaluate already exist and if not, creates them
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
    
    if method == 'function_calling.functionCalling':
        evaluable_object = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, functions, function_call, best_prompts)
    if method == 'elovalue.Elo':
        evaluable_object = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, best_prompts)
    if method != 'function_calling.functionCalling' and method != 'elovalue.Elo':

        evaluable_object = class_method(test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, best_prompts)

    # Evaluate the prompts
    results = evaluable_object.evaluate_optimal_prompt()
    cost = cost + results[2]
    if model_test == 'gpt-4':
        tokens_input_gpt4 = tokens_input_gpt4 + results[3]
        tokens_output_gpt4 = tokens_output_gpt4 + results[4]
    if model_test == 'gpt-3.5-turbo':
        tokens_input_gpt35 = tokens_input_gpt35 + results[3]
        tokens_output_gpt35 = tokens_output_gpt35 + results[4]
    yaml_folder = os.path.dirname(file)
    if method == 'elovalue.Elo':
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
    old_prompts = results[1]
    number_of_iteration = 1
    if method != 'elovalue.Elo':
        tokens_input_gen = 0
        tokens_output_gen = 0
        tokens_input_test = 0
        tokens_output_test = 0
        while iterations > 0:
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            if method == 'function_calling.functionCalling':
                new_results_prompts_cost = iteration.iterations(test_cases, number_of_prompts - best_prompts, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, old_prompts, method, functions, function_call, best_prompts)
            if method == 'elovalue.Elo':
                new_results_prompts_cost = iteration.iterations(test_cases, number_of_prompts - best_prompts, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, old_prompts, method, None, None, description, best_prompts)

            if method != 'function_calling.functionCalling' and method != 'elovalue.Elo':
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
    if method == 'elovalue.Elo':
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

"test:\n\n"

    "cases: Here, you have to put the test cases you are going to use to evaluate your prompts. If you are going to use the"
        "Elo method to evaluate them, it should be just a list of strings. If you are going to use the methods classification, "
        "equal or includes, it should be a list of lists with two elements, where the first element is the test case and the "
        "second element is the correct response to the test. Remember that if you decide to use classification, only a boolean"
        "value is allowed as a response. And if you choose function_calling.functionCalling method the test cases should be a list"
        "of lists with three elements where the first one is a test case, the second one is the correct function and the third one "
        "the correct variable to call the function.\n"

    "description: Here is the description of the type of task that summarizes the test cases.\n"
    "method: Here, you select the evaluation method for your prompts. The options are: elovalue.Elo, classification.Classification, "
        "equal.Equal, includes.Includes and function_calling.functionCalling.\n"

    "model:\n"
        "name: The name of the GPT model you will use to evaluate the prompts. Options: 'gpt-3.5-turbo', 'gpt-4'.\n"
        "temperature: The temperature of the GPT model you will use to evaluate the prompts. The value must be between 0 and 2.\n"
        "max_tokens: The maximum number of tokens you will allow the GPT model to use to generate the response to the test.\n"

    "functions: This field must only be filled out in case the 'function_calling.functionCalling' method is intended to be used."
    "If another method is used, it must not be filled out. The structure is a JSON object. Let's break down the different components:\n\n"

            "- Function Name (name): This is the identifier used to refer to this function within the context of your code.\n"

            "- Function Description (description): A brief description of what the function does.\n"

            "- Function Parameters (parameters): This section defines the input parameters that the function accepts.\n\n"

                "- Type (type): The type of the parameter being defined.\n"

                "- Properties (properties): This is an object containing properties that the input parameter object should have.\n\n"

                    "- File Type (file_type): This is a property of the parameter object.\n"

                    "- Enum (enum): An enumeration of allowed values for the 'file_type' property. (optional)\n"

                    "- Description (description): A description of what the 'file_type' property represents.\n"

                "- Required (required): An array listing the properties that are required within the parameter object. (optional)\n"

    "function_call: This field must only be filled out in case the 'function_calling.functionCalling' method is intended to be "
            "used. If another method is used, it must not be filled out.\n"

"prompts:\n\n"

    "content: A list of prompts you want to evaluate. If you want to generate them with the prompt generator, leave the list empty. "
        "Please provide a minimum number of 4 prompts\n"
    "number: The number of prompts you are going to evaluate. If you are going to provide the prompts yourself, please specify "
        "the corresponding quantity of prompts you inserted previously. If you are going to generate the prompts, indicate the "
        "quantity of prompts you want to generate. Please provide a minimum number of 4 prompts.\n"
    "change: An optional feature that allows you to make changes to your prompts, whether you provide them or generate them. "
        "If you don't want to use it, simply put None. Otherwise, the options are: uppercase, lowercase, random_uppercase, "
        "random_lowercase, random_lowercase_word, random_uppercase_word, synonymous_prompt, grammatical_errors.\n"
    "features: If you are going to generate prompts, this optional feature allows you to add special characteristics to the "
        "prompts that will be generated. For example, if you want prompts with a maximum length of 50 characters, simply complete with "
        "'Generate prompts with a maximum length of 50 characters.'\n"

"generation:\n\n"

    "model:\n\n"

        "name: The name of the GPT model you will use to generate the prompts. Options: 'gpt-3.5-turbo', 'gpt-4'.\n"
        "temperature: The temperature of the GPT model you will use to generate the prompts. The value must be between 0 and 2.\n"
        "max_tokens: The maximum number of tokens you will allow the GPT model to use to generate your prompts.\n"

"iterations:\n\n"

    "number: The number of iterations you want to perform on the best prompts obtained in your initial testing to arrive at "
        "prompts with better final results. If you don't want to try alternatives combining your best prompts just put 0.")
    parser.add_argument("optional_string", help="Optional string parameter.", nargs='?', default=None)
    args = parser.parse_args()
    try:
        dotenv.load_dotenv()
    except FileNotFoundError:
        print("Error: The .env file was not found in your directory.")
        exit(1)

    openai_env = {
    "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_TYPE": os.getenv("OPENAI_API_TYPE"),
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION")
    }
    env_schema = EnvSchema()

    try:
        validated_env = env_schema.load(openai_env)
    except ValidationError as error:
        print("Error: Invalid environment variables:", error.messages)
        exit(1)

    openai.api_base = validated_env["OPENAI_API_BASE"]
    openai.api_key = validated_env["OPENAI_API_KEY"]
    openai.api_type = validated_env["OPENAI_API_TYPE"]
    openai.api_version = validated_env["OPENAI_API_VERSION"]

    print("openai.api_base =", openai.api_base)
    print("openai.api_key =", openai.api_key)
    print("openai.api_type =", openai.api_type)
    print("openai.api_version =", openai.api_version)

    if args.optional_string == "don't run":
        if (valid_yaml(args.yaml_file)):
            approximate = approximate_cost(args.yaml_file)
            print(f"The cost of your evaluation will be approximately {approximate} dollars.")
    else:
        if (valid_yaml(args.yaml_file)):
            approximate = approximate_cost(args.yaml_file)
            print(f"The cost of your evaluation will be approximately {approximate} dollars.")
            run_evaluation(args.yaml_file, approximate)

if __name__ == "__main__":
    main()