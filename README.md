# Prompt Engineer

## Description

Prompt Engineer (lenio-ai-prompt-engineer) is a package for evaluating custom prompts using various evaluation methods. It allows you to provide your own prompts or generate them automatically and then obtain the results in a JSON file.

## Features

- Evaluate custom prompts using different evaluation methods.
- Automatically generate prompts for evaluation.
- Iterates over your best performing prompts to get better prompts.
- Save the results obtained in a JSON file.


## Installation

To use Prompt Engineer, follow these steps:

1. Clone or download this repository to your local machine.

2. Navigate to the project folder.

3. Install the required dependencies using pip:

```bash
pip install -r requires.txt
```
## Setup

To run evals, you will need to set up and specify your OpenAI API key. You can generate one at https://platform.openai.com/account/api-keys. After you obtain an API key, specify it using the OPENAI_API_KEY environment variable. Please be aware of the costs associated with using the API when running evals.

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

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH
```

3. The evaluation result will be saved in an `output.json` file in the same folder as the YAML folder. If you choose the Elo method for prompt evaluation, a scatter plot `scatter_plot.png` will also be saved in the same folder as the YAML file.

## Prompt Iteration

If you want, you can also specify the number of iterations you want to perform on your provided prompts or the ones that will be generated automatically to obtain prompts that achieve optimal behavior for the language model.

## YAML Files

We provide you with a YAML file in the 'examples/how-my-YAML-files-should-be' folder where we explain the valid structure of your YAML files and certain limitations for some variables within it. We recommend that you read it carefully before running an evaluation.

## Special features

Remember that you can also make changes to your provided prompts or those you will generate. For example, you can convert the entire content of your prompts to uppercase or rephrase them with grammatical errors.

## Cost and tokens

If you want to know how much it will cost to run your evaluation, simply enter:

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH "don't run"
```

Otherwise, run your evaluation and you'll receive the same notification about the approximate cost, along with the final cost at the end. In the final JSON file, in addition to seeing the top 2 prompts with the best results, you will also have this same information about the costs and the number of tokens effectively consumed for both GPT-3.5-turbo and GPT-4.