import argparse
import json
import logging

import concurrent.futures
from tqdm import tqdm

from utils import convert_json_to_md, view_samples
from agents import AgentBase, Solver, VanillaJudger, DiscussionReviewer, DiscussionJudger, ProofRefiner
from pipeline import naive_eval_pipeline, pessimistic_eval_pipeline, discussion_eval_pipeline, naive_process_pipeline, pessimistic_process_pipeline, pessimistic_refine_pipeline, discussion_process_pipeline

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
        convert_json_to_md(args.save_path, args.save_path.replace('.json', '.md')) # convert to md file
        logging.info(f"Converted logs to markdown and saved to {args.save_path.replace('.json', '.md')}!")

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
