# Prompt Engineer

## Description

**Prompt Engineer** (lenio-ai-prompt-engineer) is a package for evaluating custom prompts using various evaluation methods. It allows you to provide your own prompts or generate them automatically and then obtain the results in a JSON file.

## Features

- Evaluate custom prompts using different evaluation methods.
- Automatically generate prompts for evaluation.
- Iterate over your best performing prompts to get better prompts.
- Save the results obtained in a JSON file.

## Installation

To use Prompt Engineer you need to install the package and all its dependencies using pip

```bash
pip install lenio-ai-prompt-engineer
```

## Setup and Authentication

To run Prompt Engineer, you will need to set up and specify your OpenAI API key. You can generate one at https://platform.openai.com/account/api-keys. Before using Prompt-Engineer you will need to have your environment variables defined. You have two valid options, have your `OPENAI_API_KEY` defined in an `.env` in the correct folder, or, if you decide to use `azure` you need to define your `OPENAI_API_TYPE` as `azure` and your `OPENAI_API_BASE` and `OPENAI_API_VERSION` correctly in your `.env` in addition to the `OPENAI_API_KEY`.

## Usage

### Prompt Evaluation

Make sure you have the YAML file with the prompts you want to evaluate. The YAML file should follow the proper structure. You have two options for the evaluation of your YAML file when you run evaluation by terminal, the first is the following:

- Run the package with the YAML file as an argument if you have your .env file in the same directory as the package:

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH
```
- Run the package with the YAML file as an argument with the path of your .env file as shown below

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH --env_path .env_FILE_PATH
```

The evaluation result will be saved in an output.json file in the same folder as the YAML file. If you choose the Elo method for prompt evaluation, a scatter plot scatter_plot.png will also be saved in the same folder as the YAML file. A larger number of files will also be generated if you have indicated in your yaml file that you want to perform iterations.

### Automatic Prompt Generation

If the "prompts" variable is not defined in the YAML file, the program will automatically generate prompts for evaluation.

### Prompt Iteration

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
            -'Prompt1'
            - 'Prompt2'..."""

    generation:

        number: """The number of prompts you are going to evaluate. You need to provide this key value only if you are going to generate the prompts.
            Indicate the quantity of prompts you want to generate. Please provide a minimum number of 4 prompts. If you do not 
            define this key by default, 4 prompts will be created."""
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

### Usage Example

test:

    cases:
        - inout: 'The goal is to read a PDF file about climate change.'
          output1: 'open_file'
          output2: 'PDF'
        - inout: 'You want to save the changes made to your .docx file.'
          output1: 'save_file'
          output2: 'DOCX'
        - inout: 'He needs to review an Excel spreadsheet containing sales data.'
          output1: 'open_file'
          output2: 'XLSX'
        - inout: 'She is trying to print a high-resolution image for the art exhibition.'
          output1: 'print_file'
          output2: 'JPEG'
        - inout: 'Please convert the PowerPoint presentation to a PDF for sharing.'
          output1: 'convert'
          output2: 'PPTX'
        - inout: "They're editing a CSV file to update the product inventory."
          output1: 'edit_file'
          output2: 'CSV'
        - inout: 'The task is to analyze the data in a JSON file using Python.'
          output1: 'open_file'
          output2: 'JSON'

    method: function_calling.functionCalling

    model:
        name: 'gpt-3.5-turbo'
        temperature: 0.8
        max_tokens: 500

    functions: [
        {
            "name": "save_file",
            "description": "Save the changes made to your file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "enum": ['PDF', 'DOCX', 'XLSX', 'JPEG', 'GIF', 'PNG', 'PPTX', 'CSV', 'JSON'],
                        "description": "File type on which actions are being performed."
                    }
                },
                "required": ["file_type"],
            }
        },
        {
            "name": "open_file",
            "description": "Open the specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "enum": ['PDF', 'DOCX', 'XLSX', 'JPEG', 'GIF', 'PNG', 'PPTX', 'CSV', 'JSON'],
                        "description": "File type on which actions are being performed."
                    }
                },
                "required": ["file_type"],
            }
        },
        {
            "name": "print_file",
            "description": "Print the specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "enum": ['PDF', 'DOCX', 'XLSX', 'JPEG', 'GIF', 'PNG', 'PPTX', 'CSV', 'JSON'],
                        "description": "File type on which actions are being performed."
                    }
                },
                "required": ["file_type"],
            }
        },
        {
            "name": "convert_file",
            "description": "Convert the specified file to an another type of file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "enum": ['PDF', 'DOCX', 'XLSX', 'JPEG', 'GIF', 'PNG', 'PPTX', 'CSV', 'JSON'],
                        "description": "File type on which actions are being performed."
                    }
                },
                "required": ["file_type"],
            }
        },
        {
            "name": "edit_file",
            "description": "Edit the specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "enum": ['PDF', 'DOCX', 'XLSX', 'JPEG', 'GIF', 'PNG', 'PPTX', 'CSV', 'JSON'],
                        "description": "File type on which actions are being performed."
                    }
                },
                "required": ["file_type"],
            }
        },
    ]
    function_call: "auto"
    

prompts:

    list: 
        - "You're an AI programmed to assist with file operation tasks. When given a sentence describing a task, your job is to decipher it and output the command that corresponds to the function you would perform. The output should be given as an array. The first element should be the function name, and the second element should be the variable representing the type of file you're dealing with. The function name could be \"open_file\" for opening a file or \"save_file\" for saving a file. The variable could be the type of the file such as \"PDF\", \"DOCX\", etc. Be mindful of the context to choose the appropriate function and variable."
        - "As an advanced AI, your task is to identify the operation to be performed and the type of file involved, in a given statement. The operation can either be opening or saving a file. The type of file will be specified in the statement as well; it can be a PDF, DOCX, or any other type. After processing the statement, you should generate a response that consists of the particular function to be called along with the variable which represents the file type. For instance, if the operation is to open a PDF file, the function and variable could be `open_file` and `PDF` respectively.\n\nYour response should only be the function call, and it must be in this format: `function_variable`. All responses must be in the context of file operations. Remember, your goal is to accurately identify the function and variable from the given statement. Be careful to accurately reflect the function and the type of file involved."
        - "As an AI, you are interacting with a user who is performing various tasks on their computer. The user describes their current action in a sentence, and your task is to interpret that action and provide the appropriate function and the variable type in a simple format. \n\nThe function will be the primary action the user is taking, such as opening a file, saving a file, etc., and the variable will be corresponding data type or file format, such as PDF, DOCX, etc. The format of your output should be: ['function', 'variable']. \n\nThe function refers to the computer operation that needs to be performed, such as 'open_file' for opening a document or 'save_file' for saving changes to a file. The variable refers to the format of the file being used, such as 'PDF' or 'DOCX'. \n\nEnsure your interpretation of the user's action is accurate, and your response aligns with the given format."
        - "As an AI designed to aid in file management tasks, your role is to interpret a given sentence that describes a task, and produce the corresponding command that matches the function you would execute. The response should be presented as an array, with the first element being the function name, and the second element being the variable that signifies the file type you're working with. The function name could be 'open_file' for opening a file or 'save_file' for saving a file. The variable could represent the file type such as 'PDF', 'DOCX', etc. Pay attention to the context to select the suitable function and variable."


    iterations:
        number: 0

In the above example we are using the `function_calling.functionCalling` method for the evaluation of our already provided prompts. Note that if this method is used the `functions` field in the `test` key has to be defined. In this example we don't want to iterate over our prompts looking for better results. for which we put 0 iterations to be carried out, this is optional, if iterations are not going to be carried out, the `iterations` key may not be defined.
This is another example:

test:

    cases:
        - inout: 'The red acrobat closed the orange raft'
          output: 'tractor'
        - inout: 'Honey your daring rationale asserts typical ignorance or neglect.'
          output: 'hydration'
        - inout: 'Rare obnoxiousness searches elevation.'
          output: 'rose'
        - inout: 'Late into geometry he theorised.'
          output: 'light'
        - inout: 'Dear Elena find irregular new energies.'
          output: 'define'

    method: includes.Includes

    model:
        name: 'gpt-3.5-turbo'
        temperature: 0.8
        max_tokens: 500

prompts:

    generation:
        number: 4
        model:
            name: 'gpt-4'
            temperature: 0.9
            max_tokens: 500
        description: "What is the word that is formed by concatenating the first letters of each given word?"

    iterations:
        number: 2
        best_prompts: 2
        model:

            name: 'gpt-4'
            temperature: 0.9
            max_tokens: 500

In the previous example we see the structure that the YAML file must have in the case that you want to generate prompts for the evaluation and iterations to improve them.

### Special features

Remember that when you generate your prompts you can use the `constraints` key to explicitly request that the prompts you are going to generate have a special characteristic, for example, 'generate prompts of a length not exceeding 20 words'. This key has to be defined in the `generation` field in `prompts` to make use of this feature.

## Cost and tokens

If you want to know how much it will cost to run your evaluation, simply enter:

```bash
lenio-ai-prompt-engineer YAML_FILE_PATH "don't run"
```
Or
```bash
lenio-ai-prompt-engineer YAML_FILE_PATH "don't run" --env_path .env_FILE_PATH
```

The other option to know the cost is to carry out the steps that we explained in the `Usage` section.
Otherwise, run your evaluation and you'll receive the same notification about the approximate cost, along with the real final cost at the end. In the final JSON file, in addition to seeing the top 2 prompts with the best results, you will also have this same information about the costs and the number of tokens effectively consumed for both GPT-3.5-turbo and GPT-4.