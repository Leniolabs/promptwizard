from . import generation
from .evals import elovalue, classification, equal, includes


def iterations(description, test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, old_prompts_and_rating, method):
    if method == 'elovalue.Elo':
        class_method = elovalue.Elo
    elif method == 'classification.Classification':
        class_method = classification.Classification
    elif method == 'equal.Equal':
        class_method = equal.Equal
    elif method == 'includes.Includes':
        class_method = includes.Includes
    if method != 'elovalue.Elo':
            new_prompts = []
            old_prompts = []

            for item in old_prompts_and_rating:
                prompt_content = item["prompt"]
                old_prompts.append(prompt_content)

            candidate_prompts = []
            candidates = generation.generate_candidate_prompts("Your job is to generate similar prompts to prompts you are going to receive. Generate new ones by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', just say the prompt", old_prompts, 'Generate new prompts from the ones written here.', model_generation, model_generation_temperature, new_number_of_prompts, prompt_features=None)
            candidate_prompts.extend(candidates)

            new_prompts.extend(candidate_prompts)

            evaluable_object = class_method(description, test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, new_prompts)
            results = evaluable_object.evaluate_optimal_prompt()
    elif method=='elovalue.Elo':
            new_prompts = []
            candidate_prompts = []
            candidates = generation.generate_candidate_prompts("Your job is to generate similar prompts to prompts you are going to receive. Generate new ones by modifying words or phrases but in such a way that the meaning of the prompt is preserved. What you return has to be a reformulation of what you received and nothing more, no explanation is necessary. Don't return phrases like 'Here are some examples:', just say the prompt", old_prompts_and_rating, 'Generate new prompts from the ones written here.', model_generation, model_generation_temperature, new_number_of_prompts, prompt_features=None)
            candidate_prompts.extend(candidates)
            for prompt in old_prompts_and_rating:
                 candidate_prompts.append(prompt)
            evaluable_object = class_method(description, test_cases, new_number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, candidate_prompts)
            results = evaluable_object.evaluate_optimal_prompt()
    return results
        
        
    