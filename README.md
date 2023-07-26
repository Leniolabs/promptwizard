# Prompt Engineer

## Description

The project is a tool for evaluating custom prompts using various evaluation methods. It allows you to provide your own prompts or generate them automatically and then obtain the results in a JSON file.

## Features

- Evaluate custom prompts using different evaluation methods.
- Automatically generate prompts for evaluation.
- Save the results obtained in a JSON file.

## Usage

### Prompt Evaluation

1. Make sure you have the YAML file with the prompts you want to evaluate. The YAML file should follow the proper structure. If the "prompts" variable is already defined in the YAML, it will be used for evaluation.

2. Run the main script with the YAML file as an argument:

python cli/main.py examples/YAML_FILE_PATH

3. The evaluation result will be saved in an `output.json` file in the same folder as the YAML file.

### Automatic Prompt Generation

1. If the "prompts" variable is not defined in the YAML file, the program will automatically generate prompts for evaluation.

2. Run main.py passing your YAML file as parameter:

python cli/main.py examples/YAML_FILE_PATH

3. The evaluation result will be saved in an `output.json` file in the project folder.

