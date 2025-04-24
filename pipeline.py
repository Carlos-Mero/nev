from typing import Union

from utils import extract_boxed, remove_think_tags
from agents import Solver, VanillaJudger, DiscussionReviewer, DiscussionJudger, ProofRefiner

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

