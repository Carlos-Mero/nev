import argparse
import json
import logging

import openai
import concurrent.futures
from tqdm import tqdm

with open('.apiconfig.json', 'r', encoding='utf-8') as file:
    apiconfig = json.load(file)

client = openai.OpenAI(
    api_key = apiconfig['OPENAI_API_KEY'],
    base_url = apiconfig['OPENAI_BASE_URL']
)

def find_boxed(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

class AgentBase:
    temperature = 0.6
    seed = 1121
    max_retry = 7

    def __init__(self, model: str):
        self.model = model

    def format_prompt(self, **kwargs):
        """
        This method should be overrided by subclasses for specific useage
        """
        raise NotImplementedError("format_prompt should be implemented by subclasses of agent")
    def __call__(self, *args, **kwargs):
        prompt = self.format_prompt(*args, **kwargs)
        for attempt in range(self.max_retries):
            try:
                stream = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    seed=self.seed,
                    timeout=3000,
                    messages=prompt,
                    stream=True
                )
                response_content = ""
                for chunk in stream:
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        if 'debug' in kwargs and kwargs['debug']:
                            print(chunk_content, end="", flush=True)
                        response_content += chunk_content
                
                if response_content.strip():
                    return response_content
                else:
                    logging.warning(f"Attempt {attempt+1}: Response was empty. Retrying...")
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed with exception: {e}")

class Solver(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, **kwargs):
        prompt = [{'role': 'user', 'content': 'Please provide a complete and rigorous proof for this problem.'},
                  {'role': 'user', 'content': problem}]
        return prompt

class VanillaJudger(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str, **kwargs):
        prompt = [{'role': 'user', 'content': 'Here is a proof problem in math and a candidate of proof to it. You need to carefully examine and verify this proof and determine whether it is correct and rigorous. State your judgement as \\boxed{true} or \\boxed{false}.'},
                  {'role': 'user', 'content': f'### Problem\n\n{problem}\n\n### Candidate Proof\n\n{proof}'}]
        return prompt

def naive_process_pipeline(
    problem: str,
    solver: Solver,
    judger: VanillaJudger,
    debug: bool = False
) -> dict[str, any]:
    """
    Helper function that:
    1. Calls solver to get the proof.
    2. Calls judger to evaluate the proof.
    3. Extracts the 'boxed' result from judger's output.
    4. Returns a dictionary with all relevant logs.
    """
    proof = solver(problem, debug=debug)
    judge_process = judger(problem, proof, debug=debug)
    result = find_boxed(judge_process)
    return {
        'problem': problem,
        'proof': proof,
        'evaluation': judge_process,
        'judgement': True if result == "true" else False
    }

def run_naive(args):
    """
    Running naive process pipeline on given problems
    """
    solver = Solver(args.proof_model)
    judger = VanillaJudger(args.eval_model)
    with open(args.problems, 'r', encoding='utf-8') as problems_file:
        problems = json.load(problems_file)
    logging.info(f"Running naive process pipeline with {len(problems)} problems")
    logs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit each problem to the thread pool
        future_to_problem = {
            executor.submit(naive_process_pipeline, p, solver, judger, args.debug): p
            for p in problems
        }

        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_problem), 
                           total=len(future_to_problem), 
                           desc="Processing problems"):
            result_dict = future.result()
            logs.append(result_dict)

    if args.save_path is not None:
        with open(args.save_path, "w", encoding='utf-8') as log_path:
            json.dump(logs, log_path, indent=4, ensure_ascii=False)
            logging.info(f"Saved logs to path: {args.save_path}!")

def view_samples(args):
    with open(args.view, "r", encoding="utf-8") as file:
        samples = json.load(file)

    samples = samples[args.start : args.start + args.n_samples]

    for i, s in enumerate(samples):
        print("#" * 50 + "\n")
        print(f"viewing sample {args.start + i}\n")
        for key, value in s.items():
            print(f"\033[92m{key.upper()}\033[0m: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Natural language evaluation for proof problems in math")
    parser.add_argument('--proof_model', type=str, default='deepseek-r1', help="The action model the does the proof process")
    parser.add_argument('--eval_model', type=str, default='deepseek-r1', help="The base model for the natural language evaluation process")
    parser.add_argument('--reform_model', type=str, default='deepseek-v3', help="The model used for reformat the contents in the workflow")
    parser.add_argument('--naive', default=False, action='store_true', help="Enable the naive direct evaluation method")
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help="The argument sets the global temperature of all agents")
    parser.add_argument('--seed', type=int, default=1121, help="The global seed for the whole program")
    parser.add_argument('-p', '--problems', type=str, default='', help="The path to the problems to be solved. It should be a file in json format, which contains a list of problems in natural language")
    parser.add_argument('--debug', default=False, action='store_true', help="Enable debug mode for more information output")
    parser.add_argument('--save_path', type=str, default=None, help="The path to save proof and judge results in the process")
    parser.add_argument('-w', '--workers', type=int, default=1, help="The threads used in this program.")
    parser.add_argument('--max_retries', type=int, default=7, help="The maximum retry times when calling the API")

    parser.add_argument('-v', '--view', type=str, default="", help="pass the path to enable viewing mode for the log elements")
    parser.add_argument('-s', '--start', type=int, default=0, help="the start point of viewing samples")
    parser.add_argument('-n', '--n_samples', type=int, default=1, help="The length of samples to be viewed")

    args = parser.parse_args()

    if args.view:
        view_samples(args)
        return

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        if args.workers > 1:
            logging.debug("Enabling debug mode, resetting multi thread workers to 1.")
            args.workers = 1
    else:
        logging.basicConfig(level=logging.INFO)

    # setting global variables for all agents
    AgentBase.temperature = args.temperature
    AgentBase.seed = args.seed
    AgentBase.max_retries = args.max_retries

    if args.naive:
        run_naive(args)
    else:
        raise NotImplementedError("Other mode is not implemented yet.")

if __name__ == "__main__":
    main()
