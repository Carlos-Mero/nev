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
