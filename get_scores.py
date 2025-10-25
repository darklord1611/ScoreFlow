# get_scores.py
import re
import os
import argparse
import yaml
import asyncio
import aiofiles
import importlib
import pickle
import ScoreFlow.params
from metagpt.logs import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Monkey-patch để bypass MetaGPT registry
import metagpt.provider.llm_provider_registry as registry_module

def _mock_create_llm_instance(config):
    """Mock function trả về chính config (vì config đã là provider shim)."""
    return config

# Backup function gốc để restore nếu cần
_original_create_llm_instance = getattr(registry_module, 'create_llm_instance', None)
# Override
registry_module.create_llm_instance = _mock_create_llm_instance

# ============================ Executor shim ============================
class APILLMProviderShim:
    """
    Provider gọi API thay vì local vLLM, dùng cho executor.
    Giả làm 'openai_api' type để MetaGPT registry chấp nhận.
    """
    def __init__(self, base_url, api_key, model_name="gpt-3.5-turbo",
                 temperature=0.0, top_p=0.95, max_tokens=512, concurrent=5):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Khởi tạo OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Các thuộc tính để tương thích với MetaGPT
        # Thử dùng 'openai_api' thay vì 'openai'
        self.api_type = "openai_api"
        self.api_base = base_url
        self.model = model_name
        self.llm_provider = self
        
        # Thêm các thuộc tính MetaGPT có thể cần
        self.rpm = 10
        self.max_budget = 10.0
        self.calc_usage = True
        
        self._sem = asyncio.Semaphore(concurrent)
    
    async def aask(self, prompt: str, **kwargs) -> str:
        """Gọi API async."""
        loop = asyncio.get_running_loop()
        
        async with self._sem:
            try:
                # Log prompt ngắn
                short_prompt = prompt[:400].replace("\n", " ")
                logger.info(f"[APIExec] Prompt: {short_prompt}...")
                
                # Gọi API trong executor để không block
                def _call_api():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        timeout=120
                    )
                    return response.choices[0].message.content
                
                resp = await loop.run_in_executor(None, _call_api)
                
                # Log response ngắn
                short_resp = (resp or "")[:400].replace("\n", " ")
                logger.info(f"[APIExec] Response: {short_resp}...")
                return resp
                    
            except Exception as e:
                logger.error(f"[APIExec] Error: {e}")
                return ""

import aiohttp

class VLLMProviderShim:
    """
    Provider cho local vLLM server, giả lập OpenAI API.
    """
    def __init__(self, base_url="http://localhost:8000/v1", model_name="Qwen/Qwen2.5-7B-Instruct",
                 temperature=0.0, top_p=0.95, max_tokens=512, concurrent=5):
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(concurrent)

        # MetaGPT compatibility attributes
        self.api_type = "vllm_api"
        self.api_base = base_url
        self.model = model_name
        self.llm_provider = self
        self.rpm = 10
        self.max_budget = 10.0
        self.calc_usage = True

    async def aask(self, prompt: str, **kwargs) -> str:
        """Gọi local vLLM API."""
        async with self._sem:
            short_prompt = prompt[:400].replace("\n", " ")
            logger.info(f"[vLLMExec] Prompt: {short_prompt}...")

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stop": ["</graph>"]
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.base_url}/completions", json=payload, timeout=120) as resp:
                        result = await resp.json()
                        if "choices" in result and result["choices"]:
                            text = result["choices"][0].get("text", "")
                        else:
                            text = ""
                        short_resp = text[:400].replace("\n", " ")
                        logger.info(f"[vLLMExec] Response: {short_resp}...")
                        return text
            except Exception as e:
                logger.error(f"[vLLMExec] Error: {e}")
                return ""

# ============================ Helpers ============================
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_graph(question_id, graph_id, workflows_path: str):
    workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
    graph_module_name = f"{workflows_path}.graph_{question_id}_{graph_id}"
    try:
        graph_module = __import__(graph_module_name, fromlist=[""])
        graph_class = getattr(graph_module, "Workflow")
        return graph_class
    except ImportError as e:
        logger.info(f"Error loading graph: {e}")
        raise

@retry(stop=stop_after_attempt(1), wait=wait_fixed(1),
       retry=retry_if_exception_type(Exception), reraise=True)
async def _configure_graph(graph, exec_provider, problem=None):
    # MetaGPT các workflow kỳ vọng đối tượng config có .aask(...),
    # shim này cung cấp đúng hàm đó
    return graph(config=exec_provider, problem=problem)

def load_postprocessor(workflows_path: str, name):
    workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
    ext_module_name = f"{workflows_path}.{name}"
    try:
        ext_module = __import__(ext_module_name, fromlist=[""])
        ext_class = getattr(ext_module, "Workflow")
        return ext_class
    except ImportError as e:
        logger.info(f"Error loading graph: {e}")
        raise

async def _configure_postprocessor(extraction, exec_provider):
    # các postprocessor trong repo nhận llm_config; ta truyền shim
    return extraction(llm_config=exec_provider)

# ============================ Core ============================
import os
import re
import pickle
import asyncio
import importlib
from datetime import datetime

async def get_scores(work_dir, data_set, exec_provider, max_concurrent_tasks,
                     i, question_start, question_end, post_dir, use_judger,
                     use_extraction, data, benchmark, temp_path,
                     save_interval=50, output_base="checkpoint", parallel_id="main"):
    vali_num = 1
    prompt_module = importlib.import_module(f"ScoreFlow.scripts.{data_set}.conditions")
    TIME_LIMIT = prompt_module.TIME_LIMIT
    PYTHON_END = prompt_module.PYTHON_END.format(time=TIME_LIMIT)
    PYTHON_START = prompt_module.PYTHON_START
    TEMP_AVOID = prompt_module.TEMP_AVOID  # compatibility

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    graph_scores = []
    tasks = []
    processed_since_last_save = 0

    extraction = load_postprocessor(post_dir, "extraction")
    judger = load_postprocessor(post_dir, "judger")

    configured_judger = await _configure_postprocessor(judger, exec_provider) if use_judger else None
    configured_extraction = await _configure_postprocessor(extraction, exec_provider) if use_extraction else None

    # === checkpoint helpers ===
    def timestamp_str():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _write_checkpoint(filename, data):
        """Write pickle safely (flush + fsync)."""
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

    async def save_checkpoint_async():
        """Async-safe periodic save via thread offloading."""
        ts = timestamp_str()
        checkpoint_file = temp_path + f"{output_base}_scores_{parallel_id}_{ts}.pkl"
        try:
            await asyncio.to_thread(_write_checkpoint, checkpoint_file, graph_scores)
            logger.info(f"💾 Saved checkpoint ({len(graph_scores)} entries) → {checkpoint_file}")
        except Exception as e:
            logger.error(f"⚠️ Error saving checkpoint: {e}")

    # === evaluation loop ===
    for question_id in range(question_start, question_end):
        problem = data[i][0]
        graph = []
        while True:
            graph.append(data[i][1])
            i += 1
            if i == len(data):
                break
            i_id = benchmark.get_problem_id(data[i][0])
            i_last_id = benchmark.get_problem_id(data[i - 1][0])
            if i_id != i_last_id:
                break

        for graph_id, response_text in enumerate(graph):
            graph_content = re.search(r"<graph>(.*?)</graph>", response_text, re.DOTALL)
            if not graph_content:
                logger.info(f"[get_scores] No <graph>...</graph> for q={question_id}, g={graph_id}")
                continue

            python_script = PYTHON_START + graph_content.group(1).strip() + PYTHON_END
            graph_file_path = f"{work_dir}/graph_{question_id}_{graph_id}.py"
            with open(graph_file_path, mode='w', encoding="utf-8") as graph_file:
                graph_file.write(python_script)

            try:
                optimizer_graph = load_graph(question_id, graph_id, work_dir)
                input_text = benchmark.get_input_text(problem)
                configured_graph = await _configure_graph(optimizer_graph, exec_provider, input_text)

                async def sem_evaluate(question_id, graph_id, rep_id, problem,
                                       configured_extraction, configured_judger, configured_graph):
                    nonlocal processed_since_last_save
                    async with semaphore:
                        try:
                            logger.info(f"Start evaluating query id {question_id}, graph_id {graph_id}, rep_id {rep_id}")
                            results = await benchmark.evaluate_problem(
                                problem, configured_extraction, configured_judger, configured_graph
                            )
                            # results: [is_correct, pred, gold, feedback]
                            graph_scores.append([
                                question_id, graph_id, rep_id,
                                results[3], results[0], results[1], results[2]
                            ])
                            processed_since_last_save += 1

                            logger.info(f"Results of query id {question_id}, graph_id {graph_id}, rep_id {rep_id}: {results[3]}")
                            logger.info(f"Finish evaluating query id {question_id}, graph_id {graph_id}, rep_id {rep_id}")

                            # Periodic checkpointing
                            if processed_since_last_save >= save_interval:
                                await save_checkpoint_async()
                                processed_since_last_save = 0

                        except Exception as e:
                            logger.error(
                                f"Errors when evaluating q={question_id}, g={graph_id}, rep={rep_id}, Error: {str(e)}"
                            )

                for rep_id in range(vali_num):
                    tasks.append(
                        sem_evaluate(
                            question_id, graph_id, rep_id, problem,
                            configured_extraction, configured_judger, configured_graph
                        )
                    )

            except Exception as e:
                logger.info(f"Error when loading graph: {e}")

            try:
                os.remove(graph_file_path)
            except Exception:
                pass

        if question_start >= question_end:
            break

    await asyncio.gather(*tasks)

    # === final save ===
    await save_checkpoint_async()
    return graph_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated workflows")
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path tới file .pkl chứa workflow đã sinh")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (ví dụ: GSM8K, MATH, etc.)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task type (optimize hoặc inference)")
    parser.add_argument("--parallel_id", type=int, default=0)
    parser.add_argument("--full", action="store_true",
                    help="Nếu set, chạy toàn bộ file PKL, bỏ phân mảnh theo parallel_id")

    args = parser.parse_args()

    pkl_path = args.pkl_path
    data_set = args.dataset
    task_type = args.task
    parallel_id = args.parallel_id

    # ===== load benchmark + data =====
    bench_dic = ScoreFlow.params.bench_dic
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")

    # Load data trực tiếp từ file pkl truyền vào
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Không tìm thấy file {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"✅ Loaded {len(data)} workflow records from {pkl_path}")

    # ===== chuẩn bị executor (API thay vì vLLM local) =====
    # ===== chọn provider backend =====
    backend_type = os.environ.get("EXECUTOR_BACKEND", "openai").lower()

    if backend_type == "vllm":
        exec_provider = VLLMProviderShim(
            base_url=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
            model_name=os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            temperature=float(os.environ.get("EXECUTOR_TEMPERATURE", 0.0)),
            top_p=0.95,
            max_tokens=512,
            concurrent=int(os.environ.get("EXECUTOR_CONCURRENT", 5))
        )
    else:
        exec_provider = APILLMProviderShim(
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.environ.get("EXECUTOR_TEMPERATURE", 0.0)),
            top_p=0.95,
            max_tokens=512,
            concurrent=int(os.environ.get("EXECUTOR_CONCURRENT", 5))
        )

    logger.info(f"✅ Using executor backend: {backend_type.upper()} ({exec_provider.model})")

    # ===== các tham số mặc định =====
    difference = 3000
    max_concurrent_tasks = 4
    use_extraction = True
    use_judger = (task_type == "inference")
    graph_num = 1 if use_judger else 8

    work_dir = "scoreflow_workspace/temp_eval_workflow_file"
    output_dir = os.path.splitext(pkl_path)[0] + f"_scores_{parallel_id}.pkl"
    temp_path = os.path.splitext(pkl_path)[0]
    ensure_directory_exists(work_dir)

    post_dir = f"ScoreFlow/scripts/{data_set}"

    # ===== xác định range chạy =====
    len_data = int(len(data) / graph_num)
    if args.full:
        start, end = 62, len_data
    else:
        chunk = len_data // 3
        start = parallel_id * chunk
        end = (parallel_id + 1) * chunk if parallel_id < 2 else len_data


    # ===== chạy đánh giá =====
    i = 0
    question_id = 0
    while question_id < start:
        i += 1
        if benchmark.get_problem_id(data[i][0]) != benchmark.get_problem_id(data[i - 1][0]):
            question_id += 1

    score_results = []
    question_start = start
    while question_start <= end:
        question_end = min(question_start + difference, end)
        graph_scores = asyncio.run(
            get_scores(
                work_dir, data_set, exec_provider, max_concurrent_tasks,
                i, question_start, question_end, post_dir,
                use_judger, use_extraction, data, benchmark, temp_path
            )
        )
        score_results.extend(graph_scores)
        question_start += difference

    # ===== lưu kết quả =====
    with open(output_dir, "wb") as f:
        pickle.dump(score_results, f)
    logger.info(f"✅ Saved results to {output_dir}")

if __name__ == "__main__":
    main()