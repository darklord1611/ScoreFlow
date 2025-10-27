PYTHON_START = '''import asyncio
from typing import Literal
import ScoreFlow.scripts.GSM8K.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance as create

'''

PYTHON_END = '''

    async def __call__(self):
        TIMEOUT = {time}
        return await asyncio.wait_for(self.run_workflow(), timeout=TIMEOUT)'''


START_PORMPT = '''You objective is to output a workflow graph, based on the following template:

<graph>
class Workflow:
    def __init__(
        self,
        config,
        problem
    ) -> None:
        self.problem = problem
        self.config = create(config)
        self.custom = operator.Custom(self.config, self.problem)
        self.sc_ensemble = operator.ScEnsemble(self.config, self.problem)
        self.programmer = operator.Programmer(self.config, self.problem)
        self.review = operator.Review(self.config, self.problem)

    async def run_workflow(self):
        """
        This is a workflow graph.
        """
        solution = await self.custom(instruction="Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?")

        return solution
</graph>


Here's an introduction to operators you can use: (these are all you can use, do not create new operators)
1. Custom:
Usage: Generates anything based on fixed input problem and modifiable instruction.
Format MUST follow: custom(instruction: str) -> str
You can modify the instruction prompt, such like "Can you break down the problem into smaller steps?", "Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?", "Explain how to solve the problem with clear reasoning for each step", etc. For example:
solution = await self.custom(instruction="Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?")
The output can serve as the input of next operators or the final output.
2. Programmer:
Usage: Automatically writes, executes Python code, and returns the final solution based on the provided problem description and analysis.
Format MUST follow: programmer(analysis: str = 'None') -> str
The input analysis can be output of other operators, for exmaple:
program_solution = await self.programmer(analysis=first_solution)
The output can serve as the input of next operators or the final output.
3. ScEnsemble:
Usage: Evaluate every solutions, then select the best solution in the solution list.
Format MUST follow: sc_ensemble(solutions: List[str]) -> str
You can ensemble few solutions, for example:
ensembled_solution = await self.sc_ensemble(solutions=solution_list)
The output can serve as the input of next operators or the final output.
4. Review:
Usage: Given previous solution, Review operator reviews the previous solution to regenerate the solution.
Format MUST follow: review(pre_solution: str) -> str
pre_solution should be solution from previous operator, for example
rev_solution = await self.review(pre_solution=pre_solution)
The output can serve as the input of next operators or the final output.


We have the problem input as follow. But your output graph can not contain any specific information of the this problem.
Question: '''

END_PROMPT = '''

You need to notice:

**Ensure your graph is based on the given template and is correct to avoid runtime failures.** Do NOT import the modules operator and create, which have already been automatically imported. Ensure that all the prompts required by the current graph are included. Exclude any other prompts. The generated prompt must not contain any placeholders. Do not load the operators not provided.

**Introducing multiple operators at appropriate points can enhance performance.** Consider Python's loops (for, list comprehensions) to generate multiple solutions to ensemble. Consider logical and control flow (IF-ELSE, loops) for a more enhanced graphical representation.

**The graph complexity may corelate with the problem complexity.** The graph complexity must between 3 and 8. Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution.

**As for the instruction prompt for custom operator. Your instruction prompt should focus on encouraging agent to think step by step. Do not ask agent to generate multiple (a few, some, etc) answers in one operator's instruction. Also note that different agents are independent, so do not use prompts like "generate another/alternative/different answer", "generate the first/second answer", etc.**

**Your output graph must be optimized and different from the given template graph.**

**Your output graph can not contain any specific information of the given problem due to project requirement. All the information of this problem will be given as input "problem" (self.problem) and other agents will execute this workflow.**

Only output the optimized graph (remember to add <graph> and </graph>, and the output can not contain any information of the given problem).

Here is the graph without any problem information: '''


TEMP_AVOID = '''class Workflow:
    def __init__(
        self,
        config,
        problem
    ) -> None:
        self.problem = problem
        self.config = create(config)
        self.custom = operator.Custom(self.config, self.problem)
        self.sc_ensemble = operator.ScEnsemble(self.config, self.problem)
        self.programmer = operator.Programmer(self.config, self.problem)
        self.review = operator.Review(self.config, self.problem)

    async def run_workflow(self):
        """
        This is a workflow graph.
        """
        solution = await self.custom(instruction="Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?")

        return solution'''

TEST_PROMPT = "A is 1, B is 2, What's A + B?"

NO_EXCEPTION_LIST = ['''.split(' ')''', '''int(''', '''self.loop.append''']

TIME_LIMIT_TEST = 60
TIME_LIMIT = 120
sim_threshold = 0.8

examples = [
    {
        "query": "April is donating plant pots to a local school for their new garden. They ask for 30 plant pots for the daisies, and twice as many for the roses. April had already bought 100 plant pots from the garden center before she knew how many she needed. How many plant pots does April have left over?",
        "graph": "class Workflow:\n    def __init__(\n        self,\n        name: str,\n        llm_config,\n        dataset: DatasetType,\n    ) -> None:\n        self.name = name\n        self.dataset = dataset\n        self.llm = create_llm_instance(llm_config)\n        self.custom = operator.Custom(self.llm)\n        self.programmer = operator.Programmer(self.llm)\n        self.review = operator.Review(self.llm)\n        self.revise = operator.Revise(self.llm)\n\n    async def __call__(self, problem: str):\n        # Step 1: Summarize the requirements for plant pots needed.\n        summary = await self.custom(input=problem, instruction=\"Summarize the needed plant pots for daisies and roses.\")\n        \n        # Step 2: Execute the calculation of remaining pots.\n        code_problem = \"total_pots = 100; daisies = 30; roses = daisies * 2; needed_pots = daisies + roses; leftover_pots = total_pots - needed_pots; leftover_pots\"\n        calculation = await self.programmer(problem=code_problem)\n        \n        # Step 3: Review the calculated result.\n        review_feedback = await self.review(problem=problem, solution=calculation['output'])\n        \n        # Step 4: Revise the solution if necessary.\n        if not review_feedback['review_result']:\n            revised_solution = await self.revise(problem=problem, solution=calculation['output'], feedback=review_feedback['feedback'])\n            final_solution = revised_solution['solution']\n        else:\n            final_solution = calculation['output']\n\n        return final_solution, self.llm.get_usage_summary()['total_cost']",
        "prompt": "",
        "modification": "The problem type is MATH, focusing on calculating the number of plant pots left. \n    My strategy involves first calculating the total plant pots needed and then determining how many are left after donation. \n    I will use multiple operators: \n    1. A **Custom** operator to summarize the requirements based on the user's query.\n    2. A **Programmer** operator to write a code snippet that calculates the total needed pots and leftover pots.\n    3. A **Review** operator to verify the solution for accuracy before finalizing it.\n    4. A **Revise** operator to amend any inaccuracies identified in the review, if necessary.\n    \n    The logical flow is as follows:\n    - First, summarize the requirements (30 pots for daisies and 60 pots for roses).\n    - Next, execute the calculation in Python code.\n    - Review the result for correctness.\n    - Revise it if needed, then return the final solution and cost.",
    },
]

# now restructure this examples into a prompt so we can use

FEWSHOT_WORKFLOW_EXAMPLES = """
\n
Here is a set of few-shot examples of desired workflow optimizations. Each example includes the user's query, the optimized graph, the custom prompts used (if any), and a detailed reasoning of the modifications made.
"""
for ex in examples:
    FEWSHOT_WORKFLOW_EXAMPLES += f"""
    <example>
        <query>{ex['query']}</query>
        <graph>{ex['graph']}</graph>
    </example>
"""

