import argparse
import json
import logging

from tqdm import tqdm

from utils import convert_json_to_md, view_samples
from agents import AgentBase
from pipeline import peval_pipeline, prefine_pipeline, MathAgentPipeline

def run_mathagent(problems, args):
    # MathAgent does not require explicit parallel sampling and logging utils
    # It will immediately return after completed ma sampling loop
    logging.info(f"Running MathAgent loop with proof_model: {args.proof_model}")
    logging.info(f"using pessimistic verification with eval_model: {args.eval_model}")
    logging.info(f"doing chores with reform_model: {args.reform_model}")
    logging.info("Some key hyperparameters are as follows:")
    logging.info(f"Max exploration steps: {args.steps}")
    logging.info(f"Num reviews per proof: {args.reviews}")
    logging.info(f"Max refine iterations: {args.iterations}")
    logging.info(f"Max solver parallel for a single conjecture: {args.solver_parallel}")

    agent = MathAgentPipeline(
        method=args.method,
        proof_model=args.proof_model,
        eval_model=args.eval_model,
        reform_model=args.reform_model,
        max_steps=args.steps,
        reviews=args.reviews,
        refine_iterations=args.iterations,
        parallel_solve_iterations=args.solver_parallel,
        log_dir=args.log_dir,
        log_per_steps=args.log_per_steps,
    )

    if args.context is not None:
        logging.info(f"Loading context for this problem from: {args.context}")
        agent.get_context(args.context)

    if args.resume is not None:
        logging.info(f"Resuming from existing explore memory: {args.resume}")
        agent.get_context(args.resume + '/memory.json')

    for p in tqdm(problems, desc="Problems"):
        if isinstance(p, dict):
            p = p['problem']
        logging.info(f"Dealing with problem: {p}")
        agent(p)

def run(args):
    with open(args.problems, 'r', encoding='utf-8') as problems_file:
        problems = json.load(problems_file)

    logging.info(f"Running naive process pipeline with {len(problems)} problems")
    logs = []

    if args.method == "ma" or args.method == "mas":
        run_mathagent(problems, args)
        return

    if args.method == "prefine":
        logging.info(f"Running pessimistic_refine pipeline with proof_model: {args.proof_model}, eval_model: {args.eval_model}")
        logging.info(f"Total reviews: {args.reviews}")
        logging.info(f"Total iterations: {args.iterations}")
        logs = prefine_pipeline(
            problems=problems,
            solver=args.proof_model,
            reviewer=args.eval_model,
            refiner=args.proof_model,
            reviews=args.reviews,
            iterations=args.iterations,
            workers=args.workers
        )
    else:
        raise NotImplementedError("Unknown sampling method")

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
    # Submit each problem to the thread pool
    if args.method == "prefine": # These two methods refers to the same sampling mechanism
        logging.info(f"Running pessimistic_vote eval pipeline with eval_model: {args.eval_model}")
        logging.info(f"Total reviews: {args.reviews}")
        logs = peval_pipeline(
            problems=[s['problem'] for s in samples],
            proofs=[s['proof'] for s in samples],
            reviewer=args.eval_model,
            reviews=args.reviews,
            workers=args.workers
        )
    else:
        raise NotImplementedError("Unknown method")

    logging.info("Completed reevaluation")
    logging.info(f"Total Problem Count: {len(samples)}")
    logging.info(f"Pass Count under this setting: {len([True for s in logs if s['judgement']])}")

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
        choices=['prefine', 'ma', 'mas'],
        default="prefine",
        help="The type of pipeline used in our program, ma for MathAgent loop."
    )
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help="The argument sets the global temperature of all agents")
    parser.add_argument('--seed', type=int, default=1121, help="The global seed for the whole program")
    parser.add_argument('--max_tokens', type=int, default=16384, help="The maximum tokens in each api call.")
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

    # for pessimistic multi vote pipeline
    parser.add_argument('-rs', '--reviews', type=int, default=1, help="The number of reviews or advices collected before judgement")

    # for pessimistic refining pipeline
    parser.add_argument('-its', '--iterations', type=int, default=0, help="The maximum refining iterations in the pipeline")

    # for MathAgent pipeline
    parser.add_argument('--steps', type=int, default=6, help="The maximum explore iterations of our math agent")
    parser.add_argument('--solver_parallel', type=int, default=1, help="The maximum parallel solve process for a single conjecture")
    parser.add_argument('--log_dir', type=str, default="samples", help="The target log directory for math agent")
    parser.add_argument('--log_per_steps', type=int, default=10, help="Save logs in MathAgent after these steps")
    parser.add_argument('-c', '--context', type=str, default=None, help="path to the json file containing the context of your problem.")
    parser.add_argument('--resume', type=str, default=None, help="Resume from existing exploration memory. Pass the previous log_dir to continue exploring.")

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
    AgentBase.debug = args.debug
    AgentBase.max_tokens = args.max_tokens

    run(args)

if __name__ == "__main__":
    main()
