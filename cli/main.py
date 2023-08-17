import argparse
import os
import openai
import json
import matplotlib.pyplot as plt
from . import generation, iteration
from .evals import elovalue, classification, equal, includes, function_calling
from .promptChange import uppercase, lowercase, random_uppercase, random_lowercase, random_lowercase_word, random_uppercase_word, synonymous_prompt, grammatical_errors
from .approximate_cost import cost
import yaml
import textwrap
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from .validation_yaml import validation
load_dotenv()
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

def valid_yaml(file_name):
    
    valid = True

    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)

    if 'method' not in content['test']:
        valid = False
        print("Validation error:")
        print({'test': {'method': ['Must be defined']}})

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

# It loads and reads the content of a given YAML file and returns its content as a Python dictionary or list.
def read_yaml(file_name):

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
    description = yaml_content['test']['description']
    test_cases = yaml_content['test']['cases']
    method = yaml_content['test']['method']
    model_test = yaml_content['test']['model']['name']
    model_test_max_tokens = int(yaml_content['test']['model']['max_tokens'])
    prompts_value = yaml_content['prompts']['content']
    number_of_prompts = int(yaml_content['prompts']['number'])
    prompt_features = yaml_content['prompts']['features']
    prompt_change = yaml_content['prompts']['change']
    if 'generation' in yaml_content:
        model_generation = yaml_content['generation']['model']['name']
        model_generation_max_tokens = int(yaml_content['generation']['model']['max_tokens'])
    if 'functions' in yaml_content['test']:
        functions = yaml_content['test']['functions']
    iterations = int(yaml_content['iterations']['number'])
    if method == 'function_calling.functionCalling':
        approximate_cost = cost.approximate_cost(description, test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_features, prompt_change, model_generation, model_generation_max_tokens, iterations, functions)
    if method == 'elovalue.Elo' or method == 'classification.Classification' or method == 'equal.Equal' or method == 'includes.Includes':
        approximate_cost = cost.approximate_cost(description, test_cases, method, model_test, model_test_max_tokens, prompts_value, number_of_prompts, prompt_features, prompt_change, model_generation, model_generation_max_tokens, iterations, functions=None)
    return approximate_cost

def run_evaluation(file, approximate_cost):

    # Extract the 'yaml_file' attribute from the input 'file' object
    yaml_file_path = os.path.abspath(file)
    # Read the content of the YAML file
    yaml_content = read_yaml(yaml_file_path)

    # Extract the 'description', 'test_cases', 'number_of_prompts', 'candidate_model', 'generation_model',
    # 'generation_model_temperature', 'generation_model_max_tokens', and 'method' from the YAML content
    description = yaml_content['test']['description']
    test_cases = yaml_content['test']['cases']
    method = yaml_content['test']['method']
    model_test = yaml_content['test']['model']['name']
    model_test_temperature = int(yaml_content['test']['model']['temperature'])
    model_test_max_tokens = int(yaml_content['test']['model']['max_tokens'])
    prompts_value = yaml_content['prompts']['content']
    number_of_prompts = int(yaml_content['prompts']['number'])
    prompt_features = yaml_content['prompts']['features']
    prompt_change = yaml_content['prompts']['change']
    if 'generation' in yaml_content:
        model_generation = yaml_content['generation']['model']['name']
        model_generation_temperature = int(yaml_content['generation']['model']['temperature'])
        model_generation_max_tokens = int(yaml_content['generation']['model']['max_tokens'])
    if 'functions' in yaml_content['test']:
        functions = yaml_content['test']['functions']
        function_call = yaml_content['test']['function_call']
    iterations = int(yaml_content['iterations']['number'])
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
    if method != 'function_calling.functionCalling':
        object_class = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None)
    if method == 'function_calling.functionCalling':
        object_class = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None, functions, function_call)
    # Checks if the prompts to evaluate already exist and if not, creates them
    if prompts_value == []:
        if prompt_features != 'None':
            prompts_generation_cost = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.model_generation, object_class.model_generation_temperature, model_generation_max_tokens, object_class.number_of_prompts, prompt_features)
            prompts_value = prompts_generation_cost[0]
            cost = cost + prompts_generation_cost[1]
            if model_generation == 'gpt-3.5-turbo':
                tokens_input_gpt35 = tokens_input_gpt35 + prompts_generation_cost[2]
                tokens_output_gpt35 = tokens_output_gpt35 + prompts_generation_cost[3]
            elif model_generation == 'gpt-4':
                tokens_input_gpt4 = tokens_input_gpt4 + prompts_generation_cost[2]
                tokens_output_gpt4 = tokens_output_gpt4 + prompts_generation_cost[3]
        else:
            prompts_generation_cost = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.model_generation, object_class.model_generation_temperature, model_generation_max_tokens, object_class.number_of_prompts)
            prompts_value = prompts_generation_cost[0]
            cost = cost + prompts_generation_cost[1]
            if model_generation == 'gpt-3.5-turbo':
                tokens_input_gpt35 = tokens_input_gpt35 + prompts_generation_cost[2]
                tokens_output_gpt35 = tokens_output_gpt35 + prompts_generation_cost[3]
            elif model_generation == 'gpt-4':
                tokens_input_gpt4 = tokens_input_gpt4 + prompts_generation_cost[2]
                tokens_output_gpt4 = tokens_output_gpt4 + prompts_generation_cost[3]
    
    if prompt_change != 'None':
        if prompt_change == 'uppercase':
            prompts_value = uppercase.convert_prompts(prompts_value)
        elif prompt_change == 'lowercase':
            prompts_value = lowercase.convert_prompts(prompts_value)
        elif prompt_change == 'random_uppercase':
            prompts_value = random_uppercase.convert_prompts(prompts_value)
        elif prompt_change == 'random_lowercase':
            prompts_value = random_lowercase.convert_prompts(prompts_value)
        elif prompt_change == 'random_lowercase_word':
            prompts_value = random_lowercase_word.convert_prompts(prompts_value)
        elif prompt_change == 'random_uppercase_word':
            prompts_value = random_uppercase_word.convert_prompts(prompts_value)
        elif prompt_change == 'synonymous_prompt':
            prompts_value_cost = synonymous_prompt.convert_prompts(prompts_value, model_generation_temperature, model_generation_max_tokens)
            prompts_value = prompts_value_cost[0]
            cost = cost + prompts_value_cost[1]
            tokens_input_gpt35 = tokens_input_gpt35 + prompts_value_cost[2]
            tokens_output_gpt35 = tokens_output_gpt35 + prompts_value_cost[3]
        elif prompt_change == 'grammatical_errors':
            prompts_value_cost = grammatical_errors.convert_prompts(prompts_value, model_generation_temperature, model_generation_max_tokens)
            prompts_value = prompts_value_cost[0]
            cost = cost + prompts_value_cost[1]
            tokens_input_gpt35 = tokens_input_gpt35 + prompts_value_cost[2]
            tokens_output_gpt35 = tokens_output_gpt35 + prompts_value_cost[3]
    if method == 'function_calling.functionCalling':
        evaluable_object = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value, functions, function_call)
    if method != 'function_calling.functionCalling':
        evaluable_object = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value)

    
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
                new_results_prompts_cost = iteration.iterations(description, test_cases, number_of_prompts - 2, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, model_generation_max_tokens, old_prompts, method, functions, function_call)
            if method != 'function_calling.functionCalling':
                new_results_prompts_cost = iteration.iterations(description, test_cases, number_of_prompts - 2, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, model_generation_max_tokens, old_prompts, method, None, None)
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
            new_results_prompts_cost = iteration.iterations(description, test_cases, number_of_prompts - 2, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, model_generation_max_tokens, prompt_contents, method, None, None)
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
    #final_json = {**tokens_and_cost, **old_prompts}
    with open(json_file_path, 'w') as file:
        json.dump(old_prompts, file, indent=4)

    print(f"The cost of your evaluation was: {cost} dollars.")
    

def main():
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Name of the YAML file to read.")
    parser.add_argument("optional_string", help="Optional string parameter.", nargs='?', default=None)
    args = parser.parse_args()
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