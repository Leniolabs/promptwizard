import argparse
import os
import sys
import openai
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
method_folder = os.path.join(parent_folder, "method")
sys.path.append(method_folder)
openai.api_key = "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"
import elo
import classification
import selection
import yaml
from prettytable import PrettyTable

# It loads and reads the content of a given YAML file and returns its content as a Python dictionary or list.
def read_yaml(file_name):
    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)
    return content

def run_evaluation(file):
    # Extract the 'yaml_file' attribute from the input 'file' object
    file_name = file.yaml_file
    # Read the content of the YAML file and store it in 'yaml_content'
    yaml_content = read_yaml(file_name)
    # Extract the first key (block_name) from the YAML content
    block_name = list(yaml_content.keys())[0]
    # Extract the 'description', 'test_cases', 'number_of_prompts', 'candidate_model', 'generation_model',
    # 'generation_model_temperature', 'generation_model_max_tokens', and 'method' from the YAML content
    description = yaml_content[block_name]['description']
    test_cases = yaml_content[block_name]['test_cases']
    number_of_prompts = yaml_content[block_name]['number_of_prompts']
    candidate_model = yaml_content[block_name]['candidate_model']
    generation_model = yaml_content[block_name]['generation_model']
    generation_model_temperature = yaml_content[block_name]['generation_model_temperature']
    generation_model_max_tokens = yaml_content[block_name]['generation_model_max_tokens']
    method = yaml_content[block_name]['method']
    # Use the 'eval' function to convert the 'method' string into a callable class_method
    class_method = eval(method)
    # Initialize an object of the class obtained from the 'method'
    object_class = class_method(description, test_cases, number_of_prompts, generation_model, generation_model_temperature, generation_model_max_tokens, candidate_model)
    # Generate the optimal prompt table by calling the 'generate_optimal_prompt' method of the object_class
    table = object_class.generate_optimal_prompt()
    # Define a function to generate a unique file name with an incrementing number
    def generate_file_name(folder_path, base_name="results.txt"):
        counter = 1
        while True:
            file_name = f"{base_name[:-4]}{counter}.txt"
            full_path = os.path.join(folder_path, file_name)
            if not os.path.exists(full_path):
                return full_path
            counter += 1
    # Define the folder path for the results directory, and create the directory if it doesn't exist
    folder_path = os.path.join("..", "results")
    script_folder = os.path.dirname(os.path.abspath(__file__))
    full_path_folder = os.path.join(script_folder, folder_path)
    os.makedirs(full_path_folder, exist_ok=True)
    # Generate a unique file name in the results directory
    file_name = generate_file_name(full_path_folder)
    # Convert the table object into a string representation
    table_str = table.get_string()
    # Write the table string into the file with the generated name
    with open(file_name, 'w') as file:
        file.write(table_str) 

def main():
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Name of the YAML file to read.")
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()