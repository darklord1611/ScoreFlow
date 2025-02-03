import re
from typing import Callable, List, Optional, Tuple, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_outputs(self, graph):
        return await graph()

    async def _filter(self, extraction, question, answer):
        return await extraction(question, answer)

    def get_input_text(self, problem):
        input_text = problem["question"]
        return input_text

    async def judge_answer(self, judger, question, model_answer, right_answer):
        return await judger(question, model_answer, right_answer)
    
    def get_graph_input_text(self, problem):
        input_text = problem["question"]
        return input_text

    def get_problem_id(self, problem):
        return problem["id"]

    def convert_to_binary(self, s):
        if s.strip() == "0":
            return 0
        elif s.strip() == "1":
            return 1
        else:
            return 0
    
    async def evaluate_problem(self, problem: dict, extraction: Optional[Callable], judger: Optional[Callable], graph: Callable) -> Tuple[str, str, float, float, float]:
        judger = None
        input_text = self.get_input_text(problem)
        expected_output = self.extract_number(problem["answer"])
        
        try:
            output = await self._generate_outputs(graph)
            if extraction != None:
                output = await self._filter(extraction, input_text, output)
            predicted_number = self.extract_number(output)
            
            
            if judger == None:
                score, extracted_output = self.calculate_score(expected_output, predicted_number)
            else:
                score = await self.judge_answer(judger, input_text, str(predicted_number), str(expected_output))
                score = self.convert_to_binary(score)

            return input_text, predicted_number, expected_output, score
        
        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]