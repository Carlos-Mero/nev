import argparse
import json
import logging
import re
from typing import Union

import openai
import concurrent.futures
from tqdm import tqdm

with open('.apiconfig.json', 'r', encoding='utf-8') as file:
    apiconfig = json.load(file)

client = openai.OpenAI(
    api_key = apiconfig['OPENAI_API_KEY'],
    base_url = apiconfig['OPENAI_BASE_URL']
)

def find_box(pred_str: str):
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

def extract_boxed(text):
    matches = re.findall(r'(true|false)', find_box(text), flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    else:
        return "false"

def extract_judgement(text):
    matches = re.findall(r'\*\*(true|false)\*\*', text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return "false"

def remove_think_tags(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text

class AgentBase:
    temperature = 0.6
    seed = 1121
    max_retries = 7

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
        prompt = [{'role': 'user', 'content': 'Please provide a complete and rigorous proof of this problem.'},
                  {'role': 'user', 'content': problem}]
        return prompt

class VanillaJudger(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str, **kwargs):
        prompt = [{'role': 'user', 'content':
                   'Here is a proof problem in math and a candidate proof of it. You need to carefully examine and verify this proof and determine whether it is:\n'
                   '\n'
                   '1. Complete\n'
                   '2. Correct\n'
                   '3. Rigorous\n'
                   'You need to explain your rationales and state your judgement as $\\boxed{true}$ or $\\boxed{false}$ at the end of your response.\n'
                   '\n'
                   '### Problem\n'
                   '\n'
                   f'{problem}\n'
                   '\n'
                   '### Candidate Proof\n'
                   '\n'
                   f'{proof}'}]
        return prompt

class DiscussionReviewer(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str, **kwargs):
        # change the seed each time
        self.seed += 1
        prompt = [
            {'role': 'user', 'content':
             'You are a reviewer for this math problem. You are provided a candidate proof of this problem, and you need to analyze and point out one part of this proof that might be incomplete or contain some flaws. We will use your advice for further judgement of this proof.\n'
             '\n'
             '### Problem\n'
             '\n'
             f'{problem}\n'
             '\n'
             '### Candidate Proof\n'
             '\n'
             f'{proof}\n'
             'You need to explain your rationales and decide whether this candidate can be accepted as a valid proof of this problem. State your judgement as $\\boxed{true}$ or $\\boxed{false}$ at the end of your response.\n'
             }]
        return prompt

class DiscussionJudger(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str, advices: list[str], **kwargs):
        advices = ""
        for i, c in enumerate(advices):
            advices += f"#### Review {i+1}\n\n{c}\n"
        prompt = [
            {'role': 'user', 'content': 
             "You are a judger that needs to carefully review the candidate proof of this math problem. You are given some advices from other reviewers, and you need to determine whether we can accept this candidate as a valid proof of this problem based on them.\n"
             "\n"
             "### Problem\n"
             "\n"
             f"{problem}\n"
             "\n"
             "### Candidate Proof\n"
             "\n"
             f"{proof}\n"
             "\n"
             "### Reviews\n"
             "\n"
             f"{advices}\n"
             "State your judgement as $\\boxed{true}$ or $\\boxed{false}$ at the end of your response."
             }
        ]
        return prompt

class ProofRefiner(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str, review: str, **kwargs):
        return [
            {'role': 'user', 'content':
             "You are an expert that is knowledgeable across all domains in math. Here is a math problem and a candidate proof of it. However, our reviewer have found some flaws in this proof. The problem, proof and corresponding review are provided as follows. You need to refine or even rewrite this proof so that the given problem can be correctly solved.\n"
             "\n"
             "### Problem\n"
             "\n"
             f"{problem}\n"
             "\n"
             "### Candidate Proof\n"
             "\n"
             f"{proof}\n"
             "\n"
             "### Review\n"
             "\n"
             f"{review}\n"
             }
        ]

def naive_eval_pipeline(
    problem: str,
    proof: str,
    judger: VanillaJudger,
    manual_judgement: bool = None,
    debug: bool = False
    ):
    judge_process = judger(problem, proof, debug=debug)
    result = True if extract_boxed(judge_process) == 'true' else False
    return {
        'problem': problem,
        'proof': proof,
        'evaluation': judge_process,
        'judgement': result,
        'manual_judgement': manual_judgement
    }

def pessimistic_eval_pipeline(
    problem: str,
    proof: str,
    reviewer: DiscussionReviewer,
    reviews: int,
    manual_judgement: bool = None,
    debug: bool = False
):
    result = True
    review = ""
    for i in range(reviews):
        review = reviewer(problem, proof)
        review_res = True if extract_boxed(review) == "true" else False
        result = result and review_res
        if not result:
            break
    return {
        'problem': problem,
        'proof': proof,
        'evaluation': review,
        'judgement': result,
        'manual_judgement': manual_judgement
    }

def discussion_eval_pipeline(
    problem: str,
    proof: str,
    reviewer: DiscussionReviewer,
    reviews: int,
    judger: DiscussionJudger,
    manual_judgement: bool = None,
    debug: bool = False
) -> dict[str, any]:
    advices = []
    for i in range(reviews):
        advice = reviewer(problem, proof)
        advices.append(advice)
    judge_process = judger(problem, proof, advices)
    result = True if extract_boxed(judge_process) == 'true' else False
    return {
        'problem': problem,
        'proof': proof,
        'reviews': advices,
        'evaluation': judge_process,
        'judgement': result,
        'manual_judgement': manual_judgement
    }

def naive_process_pipeline(
    problem: Union[str, dict],
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
    if isinstance(problem, dict):
        problem = problem['problem']
    proof = remove_think_tags(solver(problem, debug=debug))
    return naive_eval_pipeline(problem, proof, judger, debug=debug)

def pessimistic_process_pipeline(
    problem: Union[str, dict],
    solver: Solver,
    reviewer: DiscussionReviewer,
    reviews: int,
    debug: bool = False
) -> dict[str, any]:
    """
    Helper function that:
    1. Calls solver to get the proof.
    2. Calls the reviewers to provide different advices to this proof.
    3. The answer is marked false if any one of the reviewers reports false.
    4. Extracts the 'boxed' result from judger's output.
    5. Returns a dictionary with all relevant logs.
    """
    if isinstance(problem, dict):
        problem = problem['problem']
    proof = remove_think_tags(solver(problem, debug=debug))
    return pessimistic_eval_pipeline(problem, proof, reviewer, reviews, debug=debug)

def pessimistic_refine_pipeline(
    problem: Union[str, dict],
    solver: Solver,
    reviewer: DiscussionReviewer,
    reviews: int,
    refiner: ProofRefiner,
    iterations: int,
    debug: bool = False
) -> dict[str, any]:
    """
    Helper function that:
    1. Calls solver to get the proof.
    2. Iteratively evaluate this proof with pessimistic_vote and refines this proof.
    5. Returns a dictionary with all relevant logs.
    """
    if isinstance(problem, dict):
        problem = problem['problem']
    proof = remove_think_tags(solver(problem, debug=debug))
    result = {}
    for _ in range(iterations):
        result = pessimistic_eval_pipeline(problem, proof, reviewer, reviews, debug=debug)
        if result['judgement']:
            break
        else:
            proof = remove_think_tags(refiner(problem, proof, result['evaluation'], debug=debug))
    return result

def discussion_process_pipeline(
    problem: Union[str, dict],
    solver: Solver,
    reviewer: DiscussionReviewer,
    reviews: int,
    judger: DiscussionJudger,
    debug: bool = False
) -> dict[str, any]:
    """
    Helper function that:
    1. Calls solver to get the proof.
    2. Calls the reviewers to provide different advices to this proof.
    3. Calls judger to evaluate the proof based on these advices.
    4. Extracts the 'boxed' result from judger's output.
    5. Returns a dictionary with all relevant logs.
    """
    if isinstance(problem, dict):
        problem = problem['problem']
    proof = remove_think_tags(solver(problem, debug=debug))
    return discussion_eval_pipeline(problem, proof, reviewer, reviews, judger, debug=debug)

def run(args):
    with open(args.problems, 'r', encoding='utf-8') as problems_file:
        problems = json.load(problems_file)

    logging.info(f"Running naive process pipeline with {len(problems)} problems")
    logs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit each problem to the thread pool
        if args.method == "naive":
            logging.info(f"Running naive pipeline with proof_model: {args.proof_model}, eval_model: {args.eval_model}")
            solver = Solver(args.proof_model)
            judger = VanillaJudger(args.eval_model)
            future_to_problem = {
                executor.submit(naive_process_pipeline, p, solver, judger, args.debug): p
                for p in problems
            }
        elif args.method == "pessimistic_vote":
            logging.info(f"Running pessimistic_vote pipeline with proof_model: {args.proof_model}, eval_model: {args.eval_model}")
            logging.info(f"Total reviews: {args.reviews}")
            solver = Solver(args.proof_model)
            reviewer = DiscussionReviewer(args.eval_model)
            future_to_problem = {
                executor.submit(pessimistic_process_pipeline, p, solver, reviewer, args.reviews, args.debug): p
                for p in problems
            }
        elif args.method == "pessimistic_refine":
            logging.info(f"Running pessimistic_refine pipeline with proof_model: {args.proof_model}, eval_model: {args.eval_model}")
            logging.info(f"Total reviews: {args.reviews}")
            logging.info(f"Total iterations: {args.iterations}")
            solver = Solver(args.proof_model)
            reviewer = DiscussionReviewer(args.eval_model)
            refiner = ProofRefiner(args.proof_model)
            future_to_problem = {
                executor.submit(pessimistic_refine_pipeline, p, solver, reviewer, args.reviews, refiner, args.iterations, args.debug): p
                for p in problems
            }
        elif args.method == "discussion":
            logging.info(f"Running discussion pipeline with proof_model: {args.proof_model}, eval_model: {args.eval_model}")
            logging.info(f"Total reviews: {args.reviews}")
            solver = Solver(args.proof_model)
            reviewer = DiscussionReviewer(args.eval_model)
            judger = DiscussionJudger(args.eval_model)
            future_to_problem = {
                executor.submit(discussion_process_pipeline, p, solver, reviewer, args.reviews ,judger, args.debug): p
                for p in problems
            }

        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_problem), 
                           total=len(future_to_problem), 
                           desc="Processing problems"):
            result_dict = future.result()
            logs.append(result_dict)

    logging.info(f"Problem Count: {len(problems)}")
    solved_logs = [ll for ll in logs if ll['judgement']]
    logging.info(f"Pass Count under {args.eval_model}: {len(solved_logs)}")
    logging.info(f"Failure Count under {args.eval_model}: {len(problems) - len(solved_logs)}")

    if args.save_path is not None:
        with open(args.save_path, "w", encoding='utf-8') as log_path:
            json.dump(logs, log_path, indent=4, ensure_ascii=False)
            logging.info(f"Saved logs to path: {args.save_path}!")


def reevaluate(args):
    with open(args.reevaluate, "r", encoding="utf-8") as file:
        samples = json.load(file)

    if args.false_only:
        samples = [s for s in samples if not s['manual_judgement']]

    logging.info(f"Reevaluating dataset {args.reevaluate} with model {args.eval_model}")
    logs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit each problem to the thread pool
        if args.method == "naive":
            logging.info(f"Running naive eval pipeline with eval_model: {args.eval_model}")
            judger = VanillaJudger(args.eval_model)
            future_to_problem = {
                executor.submit(naive_eval_pipeline, s['problem'], s['proof'], judger, debug=args.debug, manual_judgement=s['manual_judgement']): s
                for s in samples
            }
        elif args.method == "pessimistic_vote" or "pessimistic_refine": # These two methods refers to the same sampling mechanism
            logging.info(f"Running pessimistic_vote eval pipeline with eval_model: {args.eval_model}")
            logging.info(f"Total reviews: {args.reviews}")
            reviewer = DiscussionReviewer(args.eval_model)
            future_to_problem = {
                executor.submit(pessimistic_eval_pipeline, s['problem'], s['proof'], reviewer, args.reviews, debug=args.debug, manual_judgement=s['manual_judgement']): s
                for s in samples
            }
        elif args.method == "discussion":
            logging.info(f"Running discussion eval pipeline with eval_model: {args.eval_model}")
            logging.info(f"Total reviews: {args.reviews}")
            reviewer = DiscussionReviewer(args.eval_model)
            judger = DiscussionJudger(args.eval_model)
            future_to_problem = {
                executor.submit(discussion_eval_pipeline, s['problem'], s['proof'], reviewer, args.reviews, judger, debug=args.debug, manual_judgement=s['manual_judgement']): s
                for s in samples
            }

        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_problem), 
                           total=len(future_to_problem), 
                           desc="Processing problems"):
            result_dict = future.result()
            logs.append(result_dict)

    logging.info("Completed reevaluation")
    logging.info(f"Total Problem Count: {len(samples)}")
    pass_count = len([s for s in logs if s['judgement'] == s['manual_judgement']])
    logging.info(f"Total Pass Count: {pass_count}")
    logging.info(f"Total Failure Count: {len(samples) - pass_count}")

    if args.save_path is not None:
        with open(args.save_path, "w", encoding='utf-8') as log_path:
            json.dump(logs, log_path, indent=4, ensure_ascii=False)
            logging.info(f"Saved logs to path: {args.save_path}!")

def view_samples(args):
    with open(args.view, "r", encoding="utf-8") as file:
        samples = json.load(file)

    if args.false_only:
        samples = [s for s in samples if not s['judgement'] or ('manual_judgement' in s.keys() and not s['manual_judgement'])]
    logging.info(f"Total samples count: {len(samples)}")

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
    # parser.add_argument('--naive', default=False, action='store_true', help="Enable the naive direct evaluation method")
    parser.add_argument(
        '--method',
        type=str,
        choices=['naive', 'discussion', 'pessimistic_vote', 'pessimistic_refine'],
        default="naive",
        help="The type of pipeline used in our program"
    )
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help="The argument sets the global temperature of all agents")
    parser.add_argument('--seed', type=int, default=1121, help="The global seed for the whole program")
    parser.add_argument('-p', '--problems', type=str, default='', help="The path to the problems to be solved. It should be a file in json format, which contains a list of problems in natural language")
    parser.add_argument('--debug', default=False, action='store_true', help="Enable debug mode for more information output")
    parser.add_argument('--save_path', type=str, default=None, help="The path to save proof and judge results in the process")
    parser.add_argument('-w', '--workers', type=int, default=1, help="The threads used in this program.")
    parser.add_argument('--max_retries', type=int, default=7, help="The maximum retry times when calling the API")

    # for evaluation with different models
    parser.add_argument('-ee', '--reevaluate', type=str, default=None, help="The path to the annotated dataset to be reevaluated with eval_model. These results are then compared with annotated ground truth judgement.")

    # for viewer utilities
    parser.add_argument('-v', '--view', type=str, default=None, help="pass the path to enable viewing mode for the log elements")
    parser.add_argument('-s', '--start', type=int, default=0, help="the start point of viewing samples")
    parser.add_argument('-n', '--n_samples', type=int, default=1, help="The length of samples to be viewed")
    parser.add_argument('-fo', '--false_only', action='store_true', default=False, help="view false cases only in the dataset")

    # for pessimistic multi vote and discussion pipeline
    parser.add_argument('-rs', '--reviews', type=int, default=3, help="The number of reviews or advices collected before judgement")

    # for pessimistic refining pipeline
    parser.add_argument('-its', '--iterations', type=int, default=3, help="The maximum refining iterations in the pipeline")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        if args.workers > 1:
            logging.debug("Enabling debug mode, resetting multi thread workers to 1.")
            args.workers = 1
    else:
        logging.basicConfig(level=logging.INFO)

    if args.view:
        view_samples(args)
        return

    if args.reevaluate:
        reevaluate(args)
        return

    # setting global variables for all agents
    AgentBase.temperature = args.temperature
    AgentBase.seed = args.seed
    AgentBase.max_retries = args.max_retries

    run(args)

if __name__ == "__main__":
    main()
