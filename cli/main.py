import argparse
import os
import openai
import json
import matplotlib.pyplot as plt
from . import generation
from .evals import elovalue, classification, equal, includes
from .promptChange import uppercase, lowercase, random_uppercase, random_lowercase, random_lowercase_word, random_uppercase_word, synonymous_prompt, grammatical_errors
import yaml
import textwrap
import numpy as np
from collections import defaultdict

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
    prompt_change = yaml_content[block_name]['prompt_change']
    method = yaml_content[block_name]['method']
    iterations = yaml_content[block_name]['iterations']

    if method == 'elovalue.Elo':
        class_method = elovalue.Elo
    elif method == 'classification.Classification':
        class_method = classification.Classification
    elif method == 'equal.Equal':
        class_method = equal.Equal
    elif method == 'includes.Includes':
        class_method = includes.Includes

    # Initialize an object of the class obtained from the 'method'
    object_class = class_method(description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model, candidate_model_temperature, None)
    # Checks if the prompts to evaluate already exist and if not, creates them
    if 'prompts' in yaml_content[block_name]:
        prompts_value = yaml_content[block_name]['prompts']
    else:
        if 'prompt_features' in yaml_content[block_name]:
            prompt_features = yaml_content[block_name]['prompt_features']
            prompts_value = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.candidate_model, object_class.candidate_model_temperature, object_class.number_of_prompts, prompt_features)
        else:
            prompts_value = generation.generate_candidate_prompts(object_class.system_gen_system_prompt, object_class.test_cases, object_class.description, object_class.candidate_model, object_class.candidate_model_temperature, object_class.number_of_prompts)

    
    if prompt_change == 'yes':
        change = yaml_content[block_name]['change']
        if change == 'uppercase':
            prompts_value = uppercase.convert_prompts(prompts_value)
        elif change == 'lowercase':
            prompts_value = lowercase.convert_prompts(prompts_value)
        elif change == 'random_uppercase':
            prompts_value = random_uppercase.convert_prompts(prompts_value)
        elif change == 'random_lowercase':
            prompts_value = random_lowercase.convert_prompts(prompts_value)
        elif change == 'random_lowercase_word':
            prompts_value = random_lowercase_word.convert_prompts(prompts_value)
        elif change == 'random_uppercase_word':
            prompts_value = random_uppercase_word.convert_prompts(prompts_value)
        elif change == 'synonymous_prompt':
            prompts_value = synonymous_prompt.convert_prompts(prompts_value)
        elif change == 'grammatical_errors':
            prompts_value = grammatical_errors.convert_prompts(prompts_value)

    evaluable_object = class_method(description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model, candidate_model_temperature, prompts_value)
    
    # Evaluate the prompts
    results = evaluable_object.evaluate_optimal_prompt()
    iterations_prompts = results[1]
    prompts_to_change = results[1]

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

    while iterations > 0:
        new_prompts = []

        for item in prompts_to_change:
            prompt_content = item["prompt"]
            new_prompts.append(prompt_content)

        candidate_prompts = []
        for best_prompt in new_prompts:
            candidates = generation.generate_candidate_prompts("Your job is to generate a prompt similar to a prompt you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', just say the prompt", best_prompt, description, candidate_model, candidate_model_temperature, 1, prompt_features=None)
            candidate_prompts.extend(candidates)

        new_prompts.extend(candidate_prompts)

        evaluable_object = class_method(description, test_cases, 4, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model, candidate_model_temperature, new_prompts)
        prompts_to_change = evaluable_object.evaluate_optimal_prompt()[1]
        iterations_prompts.append(prompts_to_change)
        iterations = iterations - 1 
    
    # Full path of the output.json file in the same folder as the YAML
    output_json_path = os.path.join(yaml_folder, "output.json")
    # Convert the result to JSON format and save it to the output.json file
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file)
    print(f"Result saved in: {output_json_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Name of the YAML file to read.")
    args = parser.parse_args()
    run_evaluation(args.yaml_file)

if __name__ == "__main__":
    main()