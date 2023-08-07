import argparse
import os
import openai
import json
import matplotlib.pyplot as plt
from . import generation, iteration
from .evals import elovalue, classification, equal, includes
from .promptChange import uppercase, lowercase, random_uppercase, random_lowercase, random_lowercase_word, random_uppercase_word, synonymous_prompt, grammatical_errors
import yaml
import textwrap
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# It loads and reads the content of a given YAML file and returns its content as a Python dictionary or list.
def read_yaml(file_name):
    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)
    return content

def run_evaluation(file):

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
    model_test_temperature = yaml_content['test']['model']['temperature']
    model_test_max_tokens = yaml_content['test']['model']['max_tokens']
    prompts_value = yaml_content['prompts']['content']
    number_of_prompts = yaml_content['prompts']['number']
    prompt_features = yaml_content['prompts']['features']
    prompt_change = yaml_content['prompts']['change']
    model_generation = yaml_content['generation']['model']['name']
    model_generation_temperature = yaml_content['generation']['model']['temperature']
    iterations = yaml_content['iterations']['number']

    if method == 'elovalue.Elo':
        class_method = elovalue.Elo
    elif method == 'classification.Classification':
        class_method = classification.Classification
    elif method == 'equal.Equal':
        class_method = equal.Equal
    elif method == 'includes.Includes':
        class_method = includes.Includes

    # Initialize an object of the class obtained from the 'method'
    object_class = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, None)
    # Checks if the prompts to evaluate already exist and if not, creates them
    if prompts_value == []:
        if prompt_features != 'None':
            
            prompts_value = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.model_generation, object_class.model_generation_temperature, object_class.number_of_prompts, prompt_features)
        else:
            prompts_value = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.model_generation, object_class.model_generation_temperature, object_class.number_of_prompts)

    
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
            prompts_value = synonymous_prompt.convert_prompts(prompts_value)
        elif prompt_change == 'grammatical_errors':
            prompts_value = grammatical_errors.convert_prompts(prompts_value)

    evaluable_object = class_method(description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts_value)
    
    # Evaluate the prompts
    results = evaluable_object.evaluate_optimal_prompt()

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
        json.dump(results[0], json_file)
    print(f"Result saved in: {output_json_path}")
    old_prompts = results[1]
    print(old_prompts)
    number_of_iteration = 1
    if method != 'elovalue.Elo':
        while iterations > 0:
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            new_results = iteration.iterations(description, test_cases, number_of_prompts - 2, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, old_prompts, method)
            with open(json_file_path, 'w') as file:
                json.dump(new_results[0], file, indent=4)
            iterations = iterations - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = [sorted_data[0], sorted_data[1]]
    else:
        while iterations > 0:
            prompt_contents = [item['prompt'] for item in old_prompts]
            filename = f'output_iteration_{number_of_iteration}.json'
            json_file_path = os.path.join(yaml_folder, filename)
            combine_prompts = []
            new_results = iteration.iterations(description, test_cases, number_of_prompts - 2, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompt_contents, method)
            print(new_results[1])
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
    filename = f'output_best_prompts_and_results.json'
    json_file_path = os.path.join(yaml_folder, filename)
    with open(json_file_path, 'w') as file:
        json.dump(old_prompts, file, indent=4)
    

def main():
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Name of the YAML file to read.")
    args = parser.parse_args()
    run_evaluation(args.yaml_file)

if __name__ == "__main__":
    main()