from utils import extract_boxed, remove_think_tags, find_box, extract_tag_content
from agents import Solver, VanillaJudger, Reviewer, DiscussionJudger, ProofRefiner, Planner, SolverWithContext, VerifierWithContext, RefinerWithContext
from typing import Optional
import os
import json
import logging
from typing import Union
import concurrent.futures

from tqdm import tqdm

def naive_eval_pipeline(
    problem: str,
    proof: str,
    judger: VanillaJudger,
    manual_judgement: bool = None,
    ):
    judge_process = judger(problem, proof)
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
    reviewer: Reviewer,
    reviews: int,
    manual_judgement: bool = None,
):
    result = True
    review = ""
    for i in range(reviews):
        review = reviewer(problem, proof)
        review_res = False if find_box(review) == "false" else True
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
    reviewer: Reviewer,
    reviews: int,
    judger: DiscussionJudger,
    manual_judgement: bool = None,
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
    proof = remove_think_tags(solver(problem))
    return naive_eval_pipeline(problem, proof, judger)

def pessimistic_process_pipeline(
    problem: Union[str, dict],
    solver: Solver,
    reviewer: Reviewer,
    reviews: int,
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
    proof = remove_think_tags(solver(problem))
    return pessimistic_eval_pipeline(problem, proof, reviewer, reviews)

def pessimistic_refine_pipeline(
    problem: Union[str, dict],
    solver: Solver,
    reviewer: Reviewer,
    reviews: int,
    refiner: ProofRefiner,
    iterations: int,
) -> dict[str, any]:
    """
    Helper function that:
    1. Calls solver to get the proof.
    2. Iteratively evaluate this proof with pessimistic_vote and refines this proof.
    5. Returns a dictionary with all relevant logs.
    """
    if isinstance(problem, dict):
        problem = problem['problem']
    proof = remove_think_tags(solver(problem))
    result = {}
    for _ in range(iterations):
        result = pessimistic_eval_pipeline(problem, proof, reviewer, reviews)
        if result['judgement']:
            break
        else:
            proof = extract_tag_content(remove_think_tags(refiner(problem, proof, result['evaluation'])), 'proof')
    return result

def discussion_process_pipeline(
    problem: Union[str, dict],
    solver: Solver,
    reviewer: Reviewer,
    reviews: int,
    judger: DiscussionJudger,
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
    proof = remove_think_tags(solver(problem))
    return discussion_eval_pipeline(problem, proof, reviewer, reviews, judger)


class MathAgentPipeline():
    """
    The complete math agent pipeline, equiped with
    1. Planner for proposing new conjectures
    2. Solver for proving or disproving the conjecture
    3. Reviewer for assesing the proof generated by the solver
    4. Refiner for refining or rewriting proofs based on the feedback from the verifier
    5. Memory mechanism to consistantly collect and facilitate exploration of the open problem
    """
    def __init__(self,
                 proof_model: str,
                 eval_model: str,
                 reform_model: str,
                 max_steps: int = 6,
                 reviews: int = 3,
                 refine_iterations: int = 2,
                 parallel_solve_iterations: int = 1,
                 log_dir: str = "samples",
                 log_per_steps: int = 10,
                 ):
        self.proof_model = proof_model
        self.eval_model = eval_model
        self.reform_model = reform_model
        self.max_steps = max_steps
        self.current_steps = 0
        self.reviews = reviews
        self.refine_iterations = refine_iterations
        self.parallel_solve_iterations = parallel_solve_iterations
        self.log_dir = log_dir
        self.log_per_steps = log_per_steps

        self.memory = []
        self.planner = Planner(self.proof_model)
        self.solver = SolverWithContext(self.proof_model)
        self.reviewer = VerifierWithContext(self.eval_model)
        self.refiner = RefinerWithContext(self.proof_model)

    def pessimistic_eval(self, conjecture: str, judgement: str, proof: str) -> Optional[str]:
        """
        This function evaluates the judgement and proof of the given conjecture.
        It will return:
        1. None if no flaw in the proof is found
        2. A string containing comments about the flaws in the proof
        """
        logging.info("Start evaluation with pessimistic verification")

        # parallel pessimistic evaluation
        args = [(conjecture, judgement, proof, self.memory)] * self.reviews
        with concurrent.futures.ThreadPoolExecutor(max_workers = 1 if self.reviewer.debug else self.reviews) as executor:
            futures = [
                executor.submit(self.reviewer, *arg)
                for arg in args
            ]

            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=self.reviews,
                               desc="pverify"):
                raw_review = future.result()
                if find_box(raw_review) == "invalid":
                    for f in futures:
                        f.cancel()
                    return raw_review

        return None

    def refine_proof(self, conjecture: str, judgement: str, proof: str, verification: str) -> tuple[str, str]:
        """
        This function refines the proof with given verification when the corresponding judgement or proof is not valid.
        It will return a new tuple containing:
        [new_judgement, new_proof]
        """
        logging.info("Start refining proof")
        raw_refinement = self.refiner(conjecture, judgement, proof, verification, self.memory)
        new_judgement = find_box(raw_refinement)
        new_proof = extract_tag_content(raw_refinement, "proof")
        return (new_judgement, new_proof)

    def update_memory(self,
                      type: str,
                      content: str,
                      correctness: Optional[str],
                      proof: Optional[str],
                      comment: Optional[str]):
        """
        We can update memory using this function.
        The short term memory used in an open problem need to be a dictionary with the following elements:
        {
            "type": "conjecture | context",
            "content": "natural language statement of this element",
            "correctness": "true | false | unknown", # the correctness of retrieved context will always be true 
            "proof": "proof of this conjecture, not included in formatted string but is useful as well",
            "comment": "comments on this element"
        }
        """
        logging.info(f"Memory updated with conjecture / context:\n{content}")
        self.memory.append({
            'type': type,
            'content': content,
            'correctness': correctness,
            'proof': proof,
            'comment': comment
        })

    def explore_iteration(self, problem: str) -> Optional[bool]:
        """
        Explore for proof of this problem for one step.
        It will propose a new goal, solve and verify it to update its memory.
        There is three possible return values of this function:
        1. true. The given problem is solved and the correctness is true.
        2. false. The given problem is solved and the correctness is false.
        3. None. The given problem is not solved yet, and we need another iteration to finally solve this problem.
        """
        raw_plan = self.planner(problem, self.memory)
        final_result = extract_boxed(raw_plan)

        # Directly return the result if the solution is already found.
        if final_result is not None and final_result == "solved":
            logging.info("Found a final answer to this problem!")
            return True

        # Or else we will try to solve this conjecture.
        conjecture = extract_tag_content(raw_plan, "conjecture")
        if conjecture is None:
            logging.error("Failed to extract the conjecture!")
            return None
        
        solved = False
        parallel_solves = 0
        while not solved and parallel_solves < self.parallel_solve_iterations:
            parallel_solves += 1
            logging.info(f"Trying to solve conjecture:\n{conjecture}")
            raw_proof = self.solver(conjecture, self.memory)
            judgement = find_box(raw_proof)
            proof = extract_tag_content(raw_proof, "proof")
            if judgement is None or proof is None:
                continue

            # Pessimistic Verification and Refining
            for _ in range(self.refine_iterations):
                verification = self.pessimistic_eval(conjecture, judgement, proof)
                if verification is None:
                    solved = True
                    # The conjecture is solved, update memory
                    self.update_memory(
                        type='conjecture',
                        content=conjecture,
                        correctness=judgement,
                        proof=proof,
                        comment=None if judgement == "true" else proof
                    )
                    break
                else:
                    judgement, proof = self.refine_proof(conjecture, judgement, proof, verification)

        if not solved:
            # The memory will also be updated if the conjecture is not solved
            self.update_memory(
                type='conjecture',
                content=conjecture,
                correctness='unknown',
                proof=None,
                comment=None
            )

        return None

    def init_context(self, problem: str):
        """
        This function will initialize the solver's memory based on the problem statement.
        TODO
        """
        pass

    def save_logs(self, problem: str, correctness: Optional[bool]):
        os.makedirs(self.log_dir, exist_ok=True)
        main_log_path = self.log_dir + '/logs.json'
        memory_path = self.log_dir + '/memory.json'
        log_data = {
            'method': 'MathAgent',
            'problem': problem,
            'result': correctness,
            'memory_path': memory_path,
            'proof_model': self.proof_model,
            'eval_model': self.eval_model,
            'reform_model': self.reform_model,
            'max_steps': self.max_steps,
            'current_steps': self.current_steps,
            'reviews': self.reviews,
            'refine_iterations': self.refine_iterations,
            'parallel_solve_iterations': self.parallel_solve_iterations,
            **self.memory[-1]
        }
        with open(main_log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

        logging.info(f"Saved logs to path {self.log_dir}")

    def __call__(self, problem: str) -> dict:
        """
        Directly calls this pipeline to solve a hard open problem.
        This function will return if it successfully solved the desired problem or it hits the maximum iteration cycles.
        """
        self.current_steps = 0
        self.memory = []
        correctness = None

        with tqdm(total=self.max_steps, desc="Exploring") as pbar:
            while (correctness is None and self.current_steps < self.max_steps):
                if self.current_steps % self.log_per_steps == 1:
                    self.save_logs(problem, correctness)

                self.current_steps += 1
                correctness = self.explore_iteration(problem)
                pbar.update(1)

        self.save_logs(problem, correctness)

        return {
            "problem": problem,
            "correctness": correctness,
            "context": self.memory
        }
