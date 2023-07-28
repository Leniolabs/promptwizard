import argparse
import os
import openai
import json
import matplotlib.pyplot as plt
from . import generation
from .evals import elo, classification, equal, includes
import yaml
import textwrap
import numpy as np
from collections import defaultdict
openai.api_key = "ADD YOUR KEY HERE"

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
    # Extract the first key (block_name) from the YAML content
    block_name = list(yaml_content.keys())[0]
    # Extract the 'description', 'test_cases', 'number_of_prompts', 'candidate_model', 'generation_model',
    # 'generation_model_temperature', 'generation_model_max_tokens', and 'method' from the YAML content
    description = yaml_content[block_name]['description']
    test_cases = yaml_content[block_name]['test_cases']
    number_of_prompts = yaml_content[block_name]['number_of_prompts']
    candidate_model = yaml_content[block_name]['candidate_model']
    candidate_model_temperature = yaml_content[block_name]['candidate_model_temperature']
    generation_model = yaml_content[block_name]['generation_model']
    generation_model_temperature = yaml_content[block_name]['generation_model_temperature']
    generation_model_max_tokens = yaml_content[block_name]['generation_model_max_tokens']
    method = yaml_content[block_name]['method']
    # Use the 'eval' function to convert the 'method' string into a callable class_method
    class_method = eval(method)
    # Initialize an object of the class obtained from the 'method'
    object_class = class_method(description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model, candidate_model_temperature, None)
    # Checks if the prompts to evaluate already exist and if not, creates them
    if 'prompts' in yaml_content[block_name]:
        prompts_value = yaml_content[block_name]['prompts']
    else:
        prompts_value = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.candidate_model, object_class.candidate_model_temperature, object_class.number_of_prompts)
    evaluable_object = class_method(description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model, candidate_model_temperature, prompts_value)
    # Evaluate the prompts
    results = evaluable_object.evaluate_optimal_prompt()
    yaml_folder = os.path.dirname(file)
    # Full path of the output.json file in the same folder as the YAML
    output_json_path = os.path.join(yaml_folder, "output.json")
    # Convert the result to JSON format and save it to the output.json file
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file)
    print(f"Result saved in: {output_json_path}")
    if method == 'elo.Elo':
        # Group "elo" values by prompt using a dictionary
        elos_by_prompt = defaultdict(list)
        for item in results[number_of_prompts + 1]:
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

def main():
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Name of the YAML file to read.")
    args = parser.parse_args()
    run_evaluation(args.yaml_file)

if __name__ == "__main__":
    main()