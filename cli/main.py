import argparse
import os
import sys
import openai
import json
import generation
from evals import elo, classification, equal, includes
import yaml
openai.api_key = "ADD YOUR KEY HERE"

# It loads and reads the content of a given YAML file and returns its content as a Python dictionary or list.
def read_yaml(file_name):
    with open(file_name, 'r') as file:
        content = yaml.safe_load(file)
    return content

def run_evaluation(file):
    # Extract the 'yaml_file' attribute from the input 'file' object
    yaml_file_path = os.path.abspath(file)
    # Leer el contenido del archivo YAML
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
    # Ruta completa del archivo output.json en la misma carpeta que el YAML
    output_json_path = os.path.join(yaml_folder, "output.json")
    # Convertir el resultado en formato JSON y guardarlo en el archivo output.json
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file)
    print(f"Resultado guardado en: {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Read YAML file and get key values.")
    parser.add_argument("yaml_file", help="Name of the YAML file to read.")
    args = parser.parse_args()
    run_evaluation(args.yaml_file)

if __name__ == "__main__":
    main()