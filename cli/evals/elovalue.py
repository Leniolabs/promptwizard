import openai
from tqdm import tqdm
import itertools
from tenacity import retry, stop_after_attempt, wait_exponential
from ..cost import input, output

# K is a constant factor that determines how much ratings change
K = 32

N_RETRIES = 3  # number of times to retry a call to the ranking model if it fails

class Elo:
    def __init__(self, description, test_cases, number_of_prompts, model_test, model_test_temperature, model_test_max_tokens, model_generation, model_generation_temperature, prompts):
        self.description = description
        self.test_cases = test_cases
        self.number_of_prompts = number_of_prompts
        self.model_test = model_test
        self.model_test_temperature = model_test_temperature
        self.model_test_max_tokens = model_test_max_tokens
        self.model_generation = model_generation
        self.model_generation_temperature = model_generation_temperature
        self.system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

The prompts you will be generating will be for freeform tasks, such as generating a landing page headline, an intro paragraph, solving a math problem, etc.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message. Your resulting prompt must be enclosed in double quotation marks."""
        self.ranking_system_prompt = """Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.

You will be provided with the task description, the test prompt, and two generations - one for each system prompt.

Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.

Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.

Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.

Respond with your ranking, and nothing else. Be fair and unbiased in your judgement."""
        self.prompts = prompts

    def expected_score(self, r1, r2):
        return 1 / (1 + 10**((r2 - r1) / 400))

    def update_elo(self, r1, r2, score1):
        e1 = self.expected_score(r1, r2)
        e2 = self.expected_score(r2, r1)
        return r1 + K * (score1 - e1), r2 + K * ((1 - score1) - e2)

    # Get Score - retry up to N_RETRIES times, waiting exponentially between retries.
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
    def get_score(self, test_case, pos1, pos2):
        score = openai.ChatCompletion.create(
            model=self.model_test,
            messages=[
                {"role": "system", "content": self.ranking_system_prompt},
                {"role": "user", "content": f"""Task: {self.description.strip()}
    Prompt: {test_case}
    Generation A: {pos1}
    Generation B: {pos2}"""}
            ],
            logit_bias={
                '32': 100,  # 'A' token
                '33': 100,  # 'B' token
            },
            max_tokens=1,
            temperature=self.model_test_temperature,
        )
        tokens_input = score["usage"]["prompt_tokens"]
        tokens_output = score["usage"]["completion_tokens"]
        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost = cost_input + cost_output
        return score.choices[0].message.content, cost, tokens_input, tokens_output

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
    def get_generation(self, prompt, test_case):
        generation = openai.ChatCompletion.create(
            model=self.model_test,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{test_case}"}
            ],
            max_tokens=self.model_test_max_tokens,
            temperature=self.model_test_temperature,
        )
        tokens_input = generation["usage"]["prompt_tokens"]
        tokens_output = generation["usage"]["completion_tokens"]
        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost = cost_input + cost_output
        return generation.choices[0].message.content, cost, tokens_input, tokens_output

    def test_candidate_prompts(self):
        cost = 0
        tokens_input = 0
        tokens_output = 0

    # Initialize each prompt with an ELO rating of 1200
        prompt_ratings = {}
        i = 0
        while i < len(self.prompts):
            prompt = self.prompts[i]
            prompt_ratings[prompt] = 1200
            i += 1

        # Calculate total rounds for progress bar
        total_rounds = len(self.test_cases) * len(self.prompts) * (len(self.prompts) - 1) // 2

        # Initialize progress bar
        pbar = tqdm(total=total_rounds, ncols=70)

        battles = [{"description": self.description, "method": 'ELO'}]
        elo_prompt = []
        for prompt in self.prompts:
            elo_prompt.append({"prompt": prompt, "elo": 1200})

        # For each pair of prompts
        for prompt1, prompt2 in itertools.combinations(self.prompts, 2):
            # For each test case
            for test_case in self.test_cases:

                # Get the value of the "prompt" field as a string
                prompt_content = test_case

                # Update progress bar
                pbar.update()

                # Generate outputs for each prompt
                generation1 = self.get_generation(prompt1, test_case)
                generation2 = self.get_generation(prompt2, test_case)
                response1 = generation1[0]
                response2 = generation2[0]
                cost_generation1 = generation1[1]
                cost_generation2 = generation2[1]
                tokens_input1 = generation1[2]
                tokens_input2 = generation2[2]
                tokens_output1 = generation1[3]
                tokens_output2 = generation2[3]

                # Rank the outputs
                ranking1 = self.get_score(test_case, response1, response2)
                ranking2 = self.get_score(test_case, response2, response1)
                score1 = ranking1[0]
                score2 = ranking2[0]
                cost_score1 = ranking1[1]
                cost_score2 = ranking2[1]
                tokens_input_rank1 = ranking1[2]
                tokens_input_rank2 = ranking2[2]
                tokens_output_rank1 = ranking1[3]
                tokens_output_rank2 = ranking2[3]

                # Convert scores to numeric values
                score1 = 1 if score1 == 'A' else 0 if score1 == 'B' else 0.5
                score2 = 1 if score2 == 'B' else 0 if score2 == 'A' else 0.5

                # Average the scores
                score = (score1 + score2) / 2

                # Update ELO ratings
                r1, r2 = prompt_ratings[prompt1], prompt_ratings[prompt2]
                r1, r2 = self.update_elo(r1, r2, score)
                elo_prompt.append({"prompt": prompt1, "elo": r1})
                elo_prompt.append({"prompt": prompt2, "elo": r2})
                prompt_ratings[prompt1], prompt_ratings[prompt2] = r1, r2

                # Print the winner of this round and save the results
                if score > 0.5:
                    battles.append({"test": prompt_content, "prompt1": prompt1, "generation1": generation1, "prompt2": prompt2, "generation2": generation2, "winner": prompt1})
                    print(f"Winner: {prompt1}")
                elif score < 0.5:
                    battles.append({"test": prompt_content, "prompt1": prompt1, "generation1": generation1, "prompt2": prompt2, "generation2": generation2, "winner": prompt2})
                    print(f"Winner: {prompt2}")
                else:
                    battles.append({"test": prompt_content, "prompt1": prompt1, "generation1": generation1, "prompt2": prompt2, "generation2": generation2, "winner": 'Draw'})
                    print("Draw")

                cost = cost + cost_generation1 + cost_generation2 + cost_score1 + cost_score2
                tokens_input = tokens_input + tokens_input1 + tokens_input_rank1 + tokens_input2 + tokens_input_rank2
                tokens_output = tokens_output + tokens_output1 + tokens_output_rank1 + tokens_output2 + tokens_output_rank2

        elo_prompt_sorted = sorted(elo_prompt, key=lambda x: x["prompt"])
        # Close progress bar
        pbar.close()
        return prompt_ratings, battles, elo_prompt_sorted, cost, tokens_input, tokens_output


    def evaluate_optimal_prompt(self): 
        prompt_ratings = self.test_candidate_prompts()
        data_list = []
        for prompt, rating in sorted(prompt_ratings[0].items(), key=lambda item: item[1], reverse=True):
            data_list.append({"prompt": prompt, "rating": rating})
        data_list.append(prompt_ratings[1])
        data_list.append(prompt_ratings[2])
        best_prompts = [data_list[0], data_list[1]]
        return data_list, best_prompts, prompt_ratings[3], prompt_ratings[4], prompt_ratings[5]