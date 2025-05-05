import openai
import json
import logging

with open('.apiconfig.json', 'r', encoding='utf-8') as file:
    apiconfig = json.load(file)

client = openai.OpenAI(
    api_key = apiconfig['OPENAI_API_KEY'],
    base_url = apiconfig['OPENAI_BASE_URL']
)

class AgentBase:
    temperature = 0.6
    seed = 1121
    max_retries = 7
    max_tokens = 16384
    debug = False

    def __init__(self, model: str):
        self.model = model

    def format_prompt(self):
        """
        This method should be overrided by subclasses for specific useage
        """
        raise NotImplementedError("format_prompt should be implemented by subclasses of agent")
    def __call__(self, *args):
        prompt = self.format_prompt(*args)
        for attempt in range(self.max_retries):
            try:
                client_params = {
                    'model': self.model,
                    'temperature': self.temperature,
                    'timeout': 300000,
                    'messages': prompt,
                    'max_tokens': self.max_tokens,
                    'stream': True
                }
                if self.debug:
                    client_params['seed'] = self.seed
                stream = client.chat.completions.create(**client_params)
                response_content = ""
                for chunk in stream:
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        if self.debug:
                            print(chunk_content, end="", flush=True)
                        response_content += chunk_content
                
                if response_content.strip():
                    return response_content
                else:
                    logging.warning(f"Attempt {attempt+1}: Response was empty. Retrying...")
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed with exception: {e}")
                
        # If all attempts fail, log the error and raise an exception
        error_msg = f"All {self.max_retries} attempts failed. Terminating."
        logging.error(error_msg)
        raise RuntimeError(error_msg)

class Solver(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str):
        prompt = [{'role': 'user', 'content': 'Please provide a complete and rigorous proof of this problem.'},
                  {'role': 'user', 'content': problem}]
        return prompt

class VanillaJudger(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str):
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

class Reviewer(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str):
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
             'You need to explain your rationales and decide whether this candidate can be accepted as a valid proof of this problem. State your judgement inside $\\boxed{}$ as $\\boxed{true}$ or $\\boxed{false}$ at the end of your response.\n'
             }]
        return prompt

class DiscussionJudger(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, proof: str, advices: list[str]):
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
    def format_prompt(self, problem: str, proof: str, review: str):
        return [
            {'role': 'user', 'content':
             "You are an expert that is knowledgeable across all domains in math. Here is a math problem and a candidate proof of it. However, our reviewer have found some flaws in this proof. The problem, proof and corresponding review are provided as follows. You need to refine or even rewrite this proof so that the given problem can be correctly solved. Please wrap your refined proof inside tags <proof></proof>.\n"
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

def format_context_element(e: dict) -> str:
    """
    This function formats an element in our model's history memory into a markdown string.
    Each element in memory block should be in the following format (expressed as json)

    {
        "type": "conjecture | context",
        "content": "natural language statement of this element",
        "correctness": "true | false | unknown", # the correctness of retrieved context will always be true
        "proof": "proof of this conjecture, not included in formatted string but is useful as well",
        "comment": "comments on this element"
    }
    """
    correctness = f'**correctness**: **{e['correctness']}**' if 'correctness' in e.keys() else ''
    return (
        f'<{e['type']}>\n'
        f'**content**: {e['content']}\n'
        + correctness +
        # f'**correctness**: **{e['correctness']}**\n'
        # f'**comment**: {e['comment']}\n' if e['comment'] is not None else ''
        f'</{e['type']}>'
    )

class Planner(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, context: list[dict]):
        self.seed += 1
        context_instruct = '\n\nHere is a list of context that we have collected for this problem or our history explorations. They can serve as the background of the conjecture and proof. If you find the target problem is already in the context and marked as explicit true or false, you can directly report $\\boxed{solved}$ in your response. You do not need to propose new conjectures in this case. Remember only the conjectures with **true** correctness can be used as the base of new conjectures. False conjectures are only included for reference.\n\n### Context and History Explorations\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. This time you are given a complex and difficult open problem. Its statement is as follows:\n'
             '\n'
             f'<problem>{problem}</problem>\n'
             '\n'
             'Currently there is no known solution to this problem, and it might require a lot of hard work to obtain an answer to it. You need to carefully analyze any contents relevant to this problem and propose a new conjecture that might help solve our final goal. This conjecture will then be analyzed and verified by other agents. You need to wrap your conjecture inside two tags <conjecture></conjecture>. When you think the time is right, you can directly present the original problem as your conjecture. The solver will then try to finally solve it for us.'
             + context_instruct
             }
        ]

class SolverWithContext(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, conjecture: str, context: list[dict]):
        self.seed += 1
        context_instruct = '\n\nHere is a list of context that we have collected for this problem or our history findings during exploration. You can verify the above conjecture based on them.\n\n### Context and History Explorations\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. This time you are asked to help with frontier math research. You are given a new conjecture in math, and required to prove or disprove this conjecture. Please make your judgement on the correctness of this conjecture after comprehensive analysis on this conjecture. State your judgement inside $\\boxed{}$ as $\\boxed{true}$ or $\\boxed{false}$. At the same time you should also include a complete and rigorous proof of your judgement on the conjecture. You should wrap it inside <proof></proof> tags.\n'
             '\n'
             'Here is the statement of this conjecture:\n'
             '\n'
             f'<conjecture>{conjecture}</conjecture>'
             + context_instruct
             }
        ]

class VerifierWithContext(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, conjecture: str, judgement: str, proof: str, context: list[dict]):
        # change the seed each time
        self.seed += 1
        context_instruct = '\n\nHere is a list of context that we have collected for this problem or our history findings during exploration. They serve as the background of the conjecture and proof.\n\n### Context and History Explorations\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. This time you are asked to help with frontier math research. You are given a newly proposed conjecture in math, a judgement and a proof of it. You need to act as a reviewer of this problem. After carefully examined this proof, you need to determine whether this judgement compared with the proof is correct, complete, and rigorous. State your verification result inside $\\boxed{}$ as $\\boxed{valid}$ or $\\boxed{invalid}$. You also need to include the rationale on your decision in your response.\n'
             '\n'
             '### Conjecture and Judgement\n'
             '\n'
             f'{conjecture}\n**JUDGEMENT**: **{judgement}**\n'
             '\n'
             '### Proof\n'
             '\n'
             f'{proof}'
             + context_instruct
             }
        ]

class RefinerWithContext(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, conjecture: str, judgement: str, proof: str, verification: str, context: list[dict]):
        self.seed += 1
        context_instruct = '\n\nHere is a list of context that we have collected for this problem or our history findings during exploration. They serve as the background of the conjecture and proof.\n\n### Context and History Explorations\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. This time you are asked to help with frontier math research. We have proposed a new conjecture, and tried to prove or disprove it. However, one reviewer have found a flaw in our proof that our judgement may not be valid. You need to help refine or even completely rewrite the proof so that it can be correct, complete and rigorous. You can also alter the judgement on this conjecture if needed. You should state the correct judgement on this conjecture inside $\\boxed{}$ as $\\boxed{true}$ or $\\boxed{false}$, and wrap your refined proof inside <proof></proof> tags in your response.\n'
             '\n'
             '### Conjecture and Judgement\n'
             '\n'
             f'{conjecture}\n**JUDGEMENT**: **{judgement}**\n'
             '\n'
             '### Proof\n'
             '\n'
             f'{proof}\n'
             '\n'
             '### Review of the Proof'
             '\n'
             f'{verification}'
             + context_instruct
            }
        ]

class Explorer(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, problem: str, context: list[dict]):
        self.seed += 1
        context_instruct = '\n\nHere is a list of context that we have collected for this problem or our history findings during exploration. You can begin your new explorations based on them.\n\n### Context and History Explorations\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. This time you are asked to help solve a frontier math problem. Its statement is as follows:\n'
             '\n'
             f'<problem>{problem}</problem>\n'
             '\n'
             'However this is a quite difficult problem that can not be directly solved. You need to explore different approaches or directions that might help with our final goal. You can include one or more interesting findings in your explorations as conjectures in your response. Each of these conjectures must include complete definitions and background so that they can be viewed as standalone statements or lemmas. You should wrap them inside two tags of xml style: <conjecture></conjecture>, and each of them should be equiped with a complete and rigorous proof. The proof should be wrapped in <proof></proof> tags directly followed by the conjecture. More accurately the format should look like:\n'
             '\n'
             '<conjecture>Your new findings here</conjecture>\n'
             '<proof>Your proof of the conjecture above</proof>\n'
             '\n'
             'These conjectures will then be verified and collected as the basis for future explorations. Moreover, when you think the time is right that you are able to prove the original problem, you can simply state your proof inside <final_proof></final_proof>. Do not include these components if you are not sure about the final proof.'
             + context_instruct
             }
        ]

class ExpReviewer(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, conjecture: str, proof: str, context: list[dict]):
        self.seed += 1
        context_instruct = 'Here is a list of context that we have collected for this problem or our history findings during exploration. They serve as the background of the conjecture and proof.\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. Here you will be given a conjecture and a corresponding proof in math. You need to act as a reviewer of this proof, carefully examine and determine whether this proof is correct, complete and rigorous. State your verification result inside $\\boxed{}$ as $\\boxed{valid}$ or $\\boxed{invalid}$. You also need to include the rationale on your decision in your response.\n'
             '\n'
             '### Conjecture\n'
             '\n'
             f'{conjecture}\n'
             '\n'
             '### Proof\n'
             '\n'
             f'{proof}\n'
             '\n'
             '### Context and History Explorations\n'
             '\n'
             + context_instruct
            }
        ]

class ExpRefiner(AgentBase):
    def __init__(self, model: str):
        super().__init__(model)
    def format_prompt(self, conjecture: str, proof: str, review: str, context: list[dict]):
        self.seed += 1
        context_instruct = '\n\n### Context and History Explorations\n\nHere is a list of context that we have collected for this problem or our history findings during exploration. They serve as the background of the conjecture and proof.\n\n' + '\n'.join([format_context_element(c) for c in context]) if context else ''
        return [
            {'role': 'user', 'content':
             '### Instruction\n'
             '\n'
             'You are an expert that is knowledgeable across all domains in math. This time you are asked to help with frontier math research. We have proposed a new conjecture, and tried to prove it. However, one reviewer have found a flaw in our proof. You need to help refine or even completely rewrite the proof so that it can be correct, complete and rigorous. You can also modify the statement of this proof if needed. You should wrap the conjecture in <conjecture></conjecture> tags and the proof in <proof></proof> tags as follows in your response:\n'
             '\n'
             '<conjecture>original or modified conjecture</conjecture>\n'
             '<proof>refined proof of the conjecture above</proof>\n'
             '\n'
             '### Conjecture and Judgement\n'
             '\n'
             f'{conjecture}\n'
             '\n'
             '### Proof\n'
             '\n'
             f'{proof}\n'
             '\n'
             '### Review of the Proof'
             '\n'
             f'{review}'
             + context_instruct
            }
        ]
