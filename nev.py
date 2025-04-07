import argparse
import json
import logging

import openai
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
    def __init__(self, model: str, temperature: float, seed: int):
        self.model = model
        self.temperature = temperature
        self.seed = seed
    def format_prompt(self, input: str):
        """
        This method should be overrided by subclasses for specific useage
        """
        raise NotImplementedError("format_prompt should be implemented by subclasses of agent")
    def __call__(self, *args, **kwargs):
        prompt = self.format_prompt(*args, **kwargs)
        response = client.chat.completions.create(
            model = self.model,
            temperature = self.temperature,
            seed = self.seed,
            timeout = 300,
            messages = prompt
        )
        cnt = response.choices[0].message.content
        logging.debug(f"Full response from agent: {cnt}")
        return cnt

class Solver(AgentBase):
    def __init__(self, model: str, temperature: float, seed: int):
        super().__init__(model, temperature, seed)
    def format_prompt(self, problem: str):
        prompt = [{'role': 'user', 'content': 'Please provide a complete and rigorous proof for this problem.'},
                  {'role': 'user', 'content': problem}]
        logging.debug(f"Running Solver with prompt: {prompt}")
        return prompt

class VanillaJudger(AgentBase):
    def __init__(self, model: str, temperature: float, seed: int):
        super().__init__(model, temperature, seed)
    def format_prompt(self, problem: str, proof: str):
        prompt = [{'role': 'user', 'content': 'Here is a proof problem in math and a candidate of proof to it. You need to carefully examine and verify this proof and determine whether it is correct and rigorous. State your judgement as \\boxed{true} or \\boxed{false}.'},
                  {'role': 'user', 'content': f'### Problem\n\n{problem}\n\n### Candidate Proof\n\n{proof}'}]
        logging.debug(f"Running Judger with prompt: {prompt}")
        return prompt

def run_naive(args):
    solver = Solver(args.proof_model, args.temperature, args.seed)
    judger = VanillaJudger(args.eval_model, args.temperature, args.seed)
    with open(args.problems, 'r', encoding='utf-8') as problems_file:
        problems = json.load(problems_file)
    logs = []
    for p in tqdm(problems):
        logging.info(f"Working with problem: {p}")
        proof = solver(p)
        judge_process = judger(p, proof)
        result = find_boxed(judge_process)
        logs.append({'problem': p, 'proof': proof, 'evaluation': judge_process, 'judgement': result})

    if args.save_path is not None:
        with open(args.save_path, "w", encoding='utf-8') as log_path:
            json.dump(logs, log_path, indent=4, ensure_ascii=False)
            logging.info(f"Saved logs to path: {args.save_path}!")


def main():
    parser = argparse.ArgumentParser(description="Natural language evaluation for proof problems in math")
    parser.add_argument('--proof_model', type=str, default='deepseek-r1', help="The action model the does the proof process")
    parser.add_argument('--eval_model', type=str, default='deepseek-r1', help="The base model for the natural language evaluation process")
    parser.add_argument('--reform_model', type=str, default='deepseek-v3', help="The model used for reformat the contents in the workflow")
    parser.add_argument('--naive', default=False, action='store_true', help="Enable the naive direct evaluation method")
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help="The argument sets the global temperature of all agents")
    parser.add_argument('-s', '--seed', type=int, default=1121, help="The global seed for the whole program")
    parser.add_argument('-p', '--problems', type=str, default='', help="The path to the problems to be solved. It should be a file in json format, which contains a list of problems in natural language")
    parser.add_argument('--debug', default=False, action='store_true', help="Enable debug mode for more information output")
    parser.add_argument('--save_path', type=str, default=None, help="The path to save proof and judge results in the process")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.naive:
        run_naive(args)
    else:
        raise NotImplementedError("Other mode is not implemented yet.")

if __name__ == "__main__":
    main()
