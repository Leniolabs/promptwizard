from . import generation
from promptwizard.evals import elovalue, classification, equals, includes, function_calling, code_generation, json_validation, semantic_similarity, logprobs, assistants
from typing import List, Dict

def iterations(test_cases: List[Dict], method: str, old_prompts_and_rating: List[str], new_number_of_prompts: int, functions: Dict=None, model_test: str='gpt-3.5-turbo', model_test_temperature: float=1.2, model_test_max_tokens: int=1000, model_generation: str='gpt-4-turbo', model_generation_temperature: float=0.6, model_generation_max_tokens: int=500, function_call: str='auto', description: str=None, best_prompts: int=2, model_embeddings: str='text-embedding-ada-002', timeout: int=10, n_retries: int=5):

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
    if method == 'Function Calling':
         class_method = function_calling.functionCalling
    if method == 'Code Generation':
        class_method = code_generation.codeGeneration
    if method == 'JSON Validation':
        class_method = json_validation.jsonValidation

    if method == 'Semantic Similarity':
         class_method = semantic_similarity.semanticSimilarity
    if method == 'LogProbs':
         class_method = logprobs.LogProbs

    if method == 'Assistants':
         class_method = assistants.Assistants

    # Generate new prompts if the method is not 'Elo'
    if method != 'Elo' and method != 'Semantic Similarity' and method != 'Assistants':
            new_prompts = []
            old_prompts = []

            # Extract old prompts from old_prompts_and_rating
            for item in old_prompts_and_rating:
                prompt_content = item["prompt"]
                old_prompts.append(prompt_content)

            candidate_prompts = []
            # Generate candidate prompts for the next iteration
            candidates_iteration = generation.generate_candidate_prompts("Your job is to generate a similar prompt to prompts you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', only provide a prompt, nothing else.", old_prompts, 'Generate a new prompt from the ones written here.', model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts=new_number_of_prompts, prompt_features=None, timeout=10, n_retries=5)
            candidates = candidates_iteration[0]
            cost = cost + candidates_iteration[1]
            tokens_input_gen = tokens_input_gen + candidates_iteration[2]
            tokens_output_gen = tokens_output_gen + candidates_iteration[3]
            candidate_prompts.extend(candidates)

            new_prompts.extend(candidate_prompts)

            # Determine which evaluation method to use and evaluate optimal prompts
            if method == 'Function Calling':
                evaluable_object = class_method(test_cases, new_prompts, functions, model_test, model_test_temperature, model_test_max_tokens, best_prompts, function_call, timeout, n_retries)
            if method != 'Function Calling':
                evaluable_object = class_method(test_cases, new_prompts, model_test, model_test_temperature, model_test_max_tokens, best_prompts, timeout, n_retries)
            results = evaluable_object.evaluate_optimal_prompt()
            cost = cost + results[2]
            tokens_input_test = tokens_input_test + results[3]
            tokens_output_test = tokens_output_test + results[4]

    if method == 'Semantic Similarity':
            new_prompts = []
            old_prompts = []

            # Extract old prompts from old_prompts_and_rating
            for item in old_prompts_and_rating:
                prompt_content = item["prompt"]
                old_prompts.append(prompt_content)

            candidate_prompts = []
            # Generate candidate prompts for the next iteration
            candidates_iteration = generation.generate_candidate_prompts("Your job is to generate a similar prompt to prompts you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', only provide a prompt, nothing else.", old_prompts, 'Generate a new prompt from the ones written here.', model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts=new_number_of_prompts, prompt_features=None, timeout=10, n_retries=5)
            candidates = candidates_iteration[0]
            cost = cost + candidates_iteration[1]
            tokens_input_gen = tokens_input_gen + candidates_iteration[2]
            tokens_output_gen = tokens_output_gen + candidates_iteration[3]
            candidate_prompts.extend(candidates)

            new_prompts.extend(candidate_prompts)

            evaluable_object = class_method(test_cases, new_prompts, model_test, model_test_temperature, model_test_max_tokens, model_embeddings, timeout, n_retries)
            results = evaluable_object.evaluate_optimal_prompt()
            cost = cost + results[2]
            tokens_input_test = tokens_input_test + results[3]
            tokens_output_test = tokens_output_test + results[4]
            tokens_embeddings = results[5]

    # If the method is 'Elo', generate new prompts differently
    if method=='Elo':
            new_prompts = []
            candidate_prompts = []
            # Generate candidate prompts for the next iteration
            candidates_iteration = generation.generate_candidate_prompts("Your job is to generate a similar prompt to prompts you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', just say the prompt", old_prompts_and_rating, 'Generate a new prompt from the ones written here.', model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts=new_number_of_prompts, prompt_features=None, timeout=10, n_retries=5)
            candidates = candidates_iteration[0]
            cost = cost + candidates_iteration[1]
            candidate_prompts.extend(candidates)

            # Include old prompts as candidates
            for prompt in old_prompts_and_rating:
                 candidate_prompts.append(prompt)

            # Evaluate optimal prompts using the 'Elo' method
            evaluable_object = class_method(test_cases, candidate_prompts, description, model_test, model_test_temperature, model_test_max_tokens, timeout, n_retries)
            results = evaluable_object.evaluate_optimal_prompt()
            cost = cost + results[2]
            tokens_input_test = tokens_input_test + results[3]
            tokens_output_test = tokens_output_test + results[4]

    if method == 'Assistants':
        new_prompts = []
        old_prompts = []

        # Extract old prompts from old_prompts_and_rating
        for item in old_prompts_and_rating:
            prompt_content = item["prompt"]
            old_prompts.append(prompt_content)

        candidate_prompts = []
        # Generate candidate prompts for the next iteration
        candidates_iteration = generation.generate_candidate_prompts("Your job is to generate a similar prompt to prompts you are going to receive. Generate a new one by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', only provide a prompt, nothing else.", old_prompts, 'Generate a new prompt from the ones written here.', model_generation, model_generation_temperature, model_generation_max_tokens, number_of_prompts=new_number_of_prompts, prompt_features=None, timeout=10, n_retries=5)
        candidates = candidates_iteration[0]
        cost = cost + candidates_iteration[1]
        tokens_input_gen = tokens_input_gen + candidates_iteration[2]
        tokens_output_gen = tokens_output_gen + candidates_iteration[3]
        candidate_prompts.extend(candidates)

        new_prompts.extend(candidate_prompts)

        evaluable_object = class_method(test_cases, new_prompts, model_test, model_test_temperature, model_test_max_tokens, best_prompts, timeout, n_retries)
        results = evaluable_object.evaluate_optimal_prompt()

    if method == 'Semantic Similarity':
         return results, cost, tokens_input_gen, tokens_input_test, tokens_output_gen, tokens_output_test, tokens_embeddings
    
    if method == 'Assistants':
        return results
    return results, cost, tokens_input_gen, tokens_input_test, tokens_output_gen, tokens_output_test

def iterate(test_cases: List[Dict], method: str, prompts, iter: int, best_percentage=100, best_prompts=2, functions: Dict=None, model_test: str='gpt-3.5-turbo', model_test_temperature: float=1.2, model_test_max_tokens: int=1000, model_iteration: str='gpt-4-turbo', model_iteration_temperature: float=0.6, model_iteration_max_tokens: int=500, function_call: str='auto', description: str=None, model_embeddings: str='text-embedding-ada-002', timeout: int=10, n_retries: int=5):
    sorted_prompts = sorted(prompts, key=lambda x: x['rating'], reverse=True)
    old_prompts = sorted_prompts[:best_prompts]
    number_of_prompts = len(old_prompts)
    final_results = []
    number_of_iteration = 1
    tokens_embeddings = 0
    cost = 0

    if method != 'Elo' and method != 'Semantic Similarity' and method != 'Assistants':
        tokens_input_gen_iter = 0
        tokens_output_gen_iter = 0
        tokens_input_test_iter = 0
        tokens_output_test_iter = 0
        all_have_rating_percentage = True
        # Iterate through the list of elements and check the 'rating' value.
        for element in old_prompts:
            if element["rating"] < best_percentage:
                all_have_rating_percentage = False
                break  # If an element with a different rating is found, stop the iteration.
        while (iter > 0 and (not all_have_rating_percentage)):
            print(f"Iteration number {number_of_iteration}")
            combine_prompts = []
            if method == 'Function Calling':
                new_results_prompts_cost = iterations(test_cases, method, old_prompts, len(prompts) - best_prompts, functions, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, function_call, None, best_prompts, timeout, n_retries)
                final_results.append(new_results_prompts_cost)

            if method != 'Function Calling':
                new_results_prompts_cost = iterations(test_cases, method, old_prompts, len(prompts) - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, None, best_prompts, timeout, n_retries)
                final_results.append(new_results_prompts_cost)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen_iter = tokens_input_gen_iter + new_results_prompts_cost[2]
            tokens_output_gen_iter = tokens_output_gen_iter + new_results_prompts_cost[4]
            tokens_input_test_iter = tokens_input_test_iter + new_results_prompts_cost[3]
            tokens_output_test_iter = tokens_output_test_iter + new_results_prompts_cost[5]
            iter = iter - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            print(new_results[1])
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = sorted_data[:best_prompts]
            all_have_rating_percentage = True
            for element in old_prompts:
                if element["rating"] < best_percentage:
                    all_have_rating_percentage = False
                    break  # If an element with a different rating is found, stop the iteration.

    if method == 'Semantic Similarity':
        final_results = []
        tokens_input_gen_iter = 0
        tokens_output_gen_iter = 0
        tokens_input_test_iter = 0
        tokens_output_test_iter = 0
        tokens_embeddings_iter = 0
        while (iter > 0):
            print(f"Iteration number {number_of_iteration}")
            combine_prompts = []

            new_results_prompts_cost = iterations(test_cases, method, old_prompts, len(prompts) - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, None, best_prompts, model_embeddings, timeout, n_retries)
            final_results.append(new_results_prompts_cost)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen_iter = tokens_input_gen_iter + new_results_prompts_cost[2]
            tokens_output_gen_iter = tokens_output_gen_iter + new_results_prompts_cost[4]
            tokens_input_test_iter = tokens_input_test_iter + new_results_prompts_cost[3]
            tokens_output_test_iter = tokens_output_test_iter + new_results_prompts_cost[5]
            tokens_embeddings_iter = tokens_embeddings_iter + new_results_prompts_cost[6]
            iter = iter - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = sorted_data[:best_prompts]
        tokens_embeddings = tokens_embeddings + tokens_embeddings_iter

    if method == 'Elo':
        final_results = []
        tokens_input_gen_iter = 0
        tokens_output_gen_iter = 0
        tokens_input_test_iter = 0
        tokens_output_test_iter = 0
        while iter > 0:
            print(f"Iteration number {number_of_iteration}")
            prompt_contents = [item['prompt'] for item in old_prompts]
            combine_prompts = []
            new_results_prompts_cost = iterations(test_cases, method, prompt_contents, len(prompts) - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, description, best_prompts, timeout, n_retries)
            final_results.append(new_results_prompts_cost)
            new_results = new_results_prompts_cost[0]
            cost = cost + new_results_prompts_cost[1]
            tokens_input_gen_iter = tokens_input_gen_iter + new_results_prompts_cost[2]
            tokens_output_gen_iter = tokens_output_gen_iter + new_results_prompts_cost[4]
            tokens_input_test_iter = tokens_input_test_iter + new_results_prompts_cost[3]
            tokens_output_test_iter = tokens_output_test_iter + new_results_prompts_cost[5]

            iter = iter - 1
            number_of_iteration = number_of_iteration + 1
            old_prompts = new_results[1]

    if method == 'Assistants':
        all_have_rating_percentage = True
        # Iterate through the list of elements and check the 'rating' value.
        for element in old_prompts:
            if element["rating"] < best_percentage:
                all_have_rating_percentage = False
                break  # If an element with a different rating is found, stop the iteration.
        while (iter > 0 and (not all_have_rating_percentage)):
            print(f"Iteration number {number_of_iteration}")
            combine_prompts = []
            new_results_prompts_cost = iterations(test_cases, method, old_prompts, len(prompts) - best_prompts, None, model_test, model_test_temperature, model_test_max_tokens, model_iteration, model_iteration_temperature, model_iteration_max_tokens, None, None, best_prompts, timeout, n_retries)
            final_results.append(new_results_prompts_cost)
            new_results = new_results_prompts_cost
            iter = iter - 1
            number_of_iteration = number_of_iteration + 1
            combine_prompts.append(old_prompts)
            print(new_results[1])
            combine_prompts.append(new_results[1])
            combined_data = [item for sublist in combine_prompts for item in sublist]
            sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
            old_prompts = sorted_data[:best_prompts]
            all_have_rating_percentage = True
            for element in old_prompts:
                if element["rating"] < best_percentage:
                    all_have_rating_percentage = False
                    break  # If an element with a different rating is found, stop the iteration.
    
    return final_results
