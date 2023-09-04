from . import generation
from cli.evals import elovalue, classification, equals, includes, function_calling


def iterations(test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, model_generation_max_tokens, old_prompts_and_rating, method, functions, function_call, description=None, best_prompts=2):

    # Initialize cost and token variables
    cost = 0
    tokens_input_gen = 0
    tokens_output_gen = 0
    tokens_input_test = 0
    tokens_output_test = 0

    # Determine the class_method based on the specified method
    if method == 'Elo':
        class_method = elovalue.Elo
    if method == 'Classification':
        class_method = classification.Classification
    if method == 'Equals':
        class_method = equals.Equals
    if method == 'Includes':
        class_method = includes.Includes
    if method == 'function_calling':
         class_method = function_calling.functionCalling

    # Generate new prompts if the method is not 'Elo'
    if method != 'Elo':
            new_prompts = []
            old_prompts = []

            # Extract old prompts from old_prompts_and_rating
            for item in old_prompts_and_rating:
                prompt_content = item["prompt"]
                old_prompts.append(prompt_content)

            candidate_prompts = []
            # Generate candidate prompts for the next iteration
            candidates_iteration = generation.generate_candidate_prompts("Your job is to generate a similar prompt to prompts you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', only provide a prompt, nothing else.", old_prompts, 'Generate a new prompt from the ones written here.', model_generation, model_generation_temperature, model_generation_max_tokens, new_number_of_prompts, prompt_features=None)
            candidates = candidates_iteration[0]
            cost = cost + candidates_iteration[1]
            tokens_input_gen = tokens_input_gen + candidates_iteration[2]
            tokens_output_gen = tokens_output_gen + candidates_iteration[3]
            candidate_prompts.extend(candidates)

            new_prompts.extend(candidate_prompts)

            # Determine which evaluation method to use and evaluate optimal prompts
            if method == 'function_calling':
                evaluable_object = class_method(test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, new_prompts, functions, function_call, best_prompts)
            if method == 'Elo':
                evaluable_object = class_method(description, test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, new_prompts, best_prompts)
            if method != 'function_calling' and method != 'Elo':
                evaluable_object = class_method(test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, new_prompts, best_prompts)
            results = evaluable_object.evaluate_optimal_prompt()
            cost = cost + results[2]
            tokens_input_test = tokens_input_test + results[3]
            tokens_output_test = tokens_output_test + results[4]

    # If the method is 'Elo', generate new prompts differently
    elif method=='Elo':
            new_prompts = []
            candidate_prompts = []
            # Generate candidate prompts for the next iteration
            candidates_iteration = generation.generate_candidate_prompts("Your job is to generate a similar prompt to prompts you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', just say the prompt", old_prompts_and_rating, 'Generate a new prompt from the ones written here.', model_generation, model_generation_temperature, model_generation_max_tokens, new_number_of_prompts, prompt_features=None)
            candidates = candidates_iteration[0]
            cost = cost + candidates_iteration[1]
            candidate_prompts.extend(candidates)

            # Include old prompts as candidates
            for prompt in old_prompts_and_rating:
                 candidate_prompts.append(prompt)

            # Evaluate optimal prompts using the 'Elo' method
            evaluable_object = class_method(description, test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, candidate_prompts, best_prompts)
            results = evaluable_object.evaluate_optimal_prompt()
            cost = cost + results[2]
            tokens_input_test = tokens_input_test + results[3]
            tokens_output_test = tokens_output_test + results[4]
    return results, cost, tokens_input_gen, tokens_input_test, tokens_output_gen, tokens_output_test
