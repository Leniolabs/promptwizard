# Prompt Engineer

## Description

Prompt Engineer (lenio-ai-prompt-engineer) is a package for evaluating custom prompts using various evaluation methods. It allows you to provide your own prompts or generate them automatically and then obtain the results in a JSON file.

## Features

- Evaluate custom prompts using different evaluation methods.
- Automatically generate prompts for evaluation.
- Save the results obtained in a JSON file.


## Installation

To use Prompt Engineer, follow these steps:

1. Clone or download this repository to your local machine.

2. Navigate to the project folder.

3. Install the required dependencies using pip:

```bash
pip install -r requires.txt
```

## Usage

### Prompt Evaluation

1. Make sure you have the YAML file with the prompts you want to evaluate. The YAML file should follow the proper structure. If the "prompts" variable is already defined in the YAML, it will be used for evaluation.

2. Run the package with the YAML file as an argument:

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH
```

3. The evaluation result will be saved in an `output.json` file in the same folder as the YAML file. If you choose the Elo method for prompt evaluation, a scatter plot `scatter_plot.png` will also be saved in the same folder as the YAML file.

### Automatic Prompt Generation

1. If the "prompts" variable is not defined in the YAML file, the program will automatically generate prompts for evaluation.

2. Run the package passing your YAML file as parameter:

lenio-ai-prompt-engineer YAML_FILE_PATH

3. The evaluation result will be saved in an `output.json` file in the same folder as the YAML folder. If you choose the Elo method for prompt evaluation, a scatter plot `scatter_plot.png` will also be saved in the same folder as the YAML file.
