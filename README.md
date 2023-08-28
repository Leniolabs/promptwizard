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

To run Prompt Engineer, you will need to set up and specify your OpenAI API key. You can generate one at https://platform.openai.com/account/api-keys. After you obtain an API key, specify it using the OPENAI_API_KEY environment variable. Please be aware of the costs associated with using the API when running evals.

## Authentication

Before using Prompt-Engineer you will need to have your environment variables defined. You have two valid options, have your `OPENAI_API_KEY` defined in an `.env` in the correct folder, or, if you decide to use `azure` you need to define your `OPENAI_API_TYPE` as `azure` and your `OPENAI_API_BASE` and `OPENAI_API_VERSION` correctly in your `.env` in addition to the `OPENAI_API_KEY`.

## Usage

### Prompt Evaluation

1. Make sure you have the YAML file with the prompts you want to evaluate. The YAML file should follow the proper structure.

2. 
- Run the package with the YAML file as an argument if you have your .env file in the same directory as the package:

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH
```
- Run the package with the YAML file as an argument with the path of your .env file as shown below

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH --env_path .env_FILE_PATH
```

3. The evaluation result will be saved in an `output.json` file in the same folder as the YAML file. If you choose the Elo method for prompt evaluation, a scatter plot `scatter_plot.png` will also be saved in the same folder as the YAML file. A larger number of files will also be generated if you have indicated in your yaml file that you want to perform iterations.

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

We provide you an explanation of the valid structure of your YAML files and certain limitations for some variables within it. We recommend that you read it carefully before running an evaluation.

The following is the structure that your YAML files should have.

test:

    cases: """ Here, you have to put the test cases you are going to use to evaluate your prompts. If you are going to use the
        Elo method to evaluate them, it should be just a list of strings. If you are going to use the methods classification, 
        equal or includes, it should be a list of tuples with two elements, where the first element is the test case and the 
        second element is the correct response to the test. Remember that if you decide to use classification, only a boolean
        value is allowed as a response. the form of your test cases has to be, in case of selecting the Elo method:
            -'Test1'
            -'Test2'...
        If you choose the methods Classification, Equals, Includes they must be of the form:
            -inout: 'Test1'
            output: 'Answer1'
            -inout: 'Test2'
            output: 'Answer2'
        And in case the method is function_calling:
            -inout: 'Test1'
            output1: 'name_function'
            output2: 'variable'
            -inout: 'Test2'
            output1: 'name_function'
            output2: 'variable'"""

    description: """Here is the description of the type of task that summarizes the test cases. You only have to use this field if 
        you are going to use the 'Elo' method"""
    method: """Here, you select the evaluation method for your prompts. You must choose between 'Elo',
        'Classification', 'Equals', 'Includes' and 'function_calling'."""

    model:
        name: """The name of the GPT model you will use to evaluate the prompts."""
        temperature: """The temperature of the GPT model you will use to evaluate the prompts."""
        max_tokens: """The maximum number of tokens you will allow the GPT model to use to generate the response to the test."""

    functions: """This field must only be filled out in case the 'function_calling.functionCalling' method is intended to be used.
    If another method is used, it must not be filled out. The structure is a JSON object. Let's break down the different components:

            - Function Name (name): This is the identifier used to refer to this function within the context of your code.

            - Function Description (description): A brief description of what the function does.

            - Function Parameters (parameters): This section defines the input parameters that the function accepts.

                - Type (type): The type of the parameter being defined.

                - Properties (properties): This is an object containing properties that the input parameter object should have.

                    - File Type (file_type): This is a property of the parameter object.

                    - Enum (enum): An enumeration of allowed values for the 'file_type' property. (optional)

                    - Description (description): A description of what the 'file_type' property represents.

                - Required (required): An array listing the properties that are required within the parameter object. (optional)"""

    function_call: """This field must only be filled out in case the 'function_calling.functionCalling' method is intended to be 
            used. If another method is used, it must not be filled out."""

prompts: """You have two options, either provide your list of prompts or generate them following the instructions below."""

    list: """A list of prompts you want to evaluate. If you want to generate them with the prompt generator, leave the list empty.
        Please provide a minimum number of 4 prompts. Your prompts must be listed as follows:
            - 'Prompt1'
            - 'Prompt2'..."""

    generation:

        number: """The number of prompts you are going to evaluate. You need to provide this key value only if you are going to generate the prompts. Indicate the quantity of prompts you want to generate. Please provide a minimum number of 4 prompts. If you do not define this key by default, 4 prompts will be created."""
        constraints: """If you are going to generate prompts, this optional feature allows you to add special characteristics to the 
            prompts that will be generated. For example, if you want prompts with a maximum length of 50 characters, simply complete with 
            'Generate prompts with a maximum length of 50 characters'. If you don't want to use it, you don't need to have this key 
            defined."""
        description: """Here is the description of the type of task that summarizes the test cases."""

        model:

            name: """The name of the GPT model you will use to generate the prompts."""
            temperature: """The temperature of the GPT model you will use to generate the prompts."""
            max_tokens: """The maximum number of tokens you will allow the GPT model to use to generate your prompts."""

    iterations:
        number: """The number of iterations you want to perform on the best prompts obtained in your initial testing to arrive at 
            prompts with better final results. If you don't want to try alternatives combining your best prompts just put 0."""
        best_prompts: """The number of prompts you want to iterate over. the value must be between 2 and the number of prompts you 
            provide (or generate) minus one. If you do not define this value but do want to iterate, the default value will be 2."""

        model:

            name: """The name of the GPT model you will use to generate the prompts."""
            temperature: """The temperature of the GPT model you will use to generate the prompts."""
            max_tokens: """The maximum number of tokens you will allow the GPT model to use to generate your prompts."""

In case the YAML file you wish to evaluate has errors in its structure, don't worry. Prior to being assessed by the prompt engineer, your file will be validated, and you will receive a notification indicating where you need to make corrections to it in order to be evaluated successfully.

## Special features

Remember that when you generate your prompts you can use the `constraints` key to explicitly request that the prompts you are going to generate have a special characteristic, for example, 'generate prompts of a length not exceeding 20 words'.

## Cost and tokens

If you want to know how much it will cost to run your evaluation, simply enter:

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH "don't run"
```

Otherwise, run your evaluation and you'll receive the same notification about the approximate cost, along with the real final cost at the end. In the final JSON file, in addition to seeing the top prompts with the best results, you will also have this same information about the costs and the number of tokens effectively consumed for both GPT-3.5-turbo and GPT-4.