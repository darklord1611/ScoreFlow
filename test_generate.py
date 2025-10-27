import glob
import traceback
import aiofiles
import json
import os
import importlib
import re
import yaml
import asyncio
import pickle
import argparse
from difflib import SequenceMatcher
from openai import OpenAI

import ScoreFlow.params
from termcolor import cprint
from metagpt.logs import logger
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

import random

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
OPENAI_BASE_URL="https://vnu-vinallama--example-vllm-openai-compatible-1-serve.modal.run/v1"
API_KEY="super-secret-key"

# Force reload .env even if it was loaded before
load_dotenv(override=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class APILLMProviderShim:
    """Provider gọi API thay vì local vLLM, dùng cho executor."""
    def __init__(self, base_url, api_key, model_name="gpt-3.5-turbo",
                 temperature=0.0, top_p=0.95, max_tokens=1024, concurrent=5, stop=["</graph>"]):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        
        # Khởi tạo OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Các thuộc tính để tương thích với MetaGPT
        self.api_type = "openai"
        self.api_base = base_url
        self.model = model_name
        self.llm_provider = self
        
        self._sem = asyncio.Semaphore(concurrent)
    
    async def aask(self, prompt: str, **kwargs) -> str:
        """Gọi API async."""
        loop = asyncio.get_running_loop()
        
        async with self._sem:
            try:
                # Log prompt ngắn
                short_prompt = prompt[:100].replace("\n", " ")
                logger.info(f"[APIExec] Prompt: {short_prompt}...")
                
                # Gọi API trong executor để không block
                def _call_api():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        stop=self.stop,
                        timeout=120
                    )
                    return response.choices[0].message.content
                
                resp = await loop.run_in_executor(None, _call_api)
                
                # Log response ngắn
                short_resp = (resp or "")[:100].replace("\n", " ")
                logger.info(f"[APIExec] Response: {short_resp}...")
                return resp
                    
            except Exception as e:
                logger.error(f"[APIExec] Error: {e}")
                return ""


class LocalLLMProviderShim:
    """Provider local vLLM (giữ lại cho generator)."""
    def __init__(self, llm_engine, model_name="local-vllm",
                 temperature=0.0, top_p=0.95, max_tokens=512, concurrent=1):
        self._engine = llm_engine
        self._sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.api_type = "local"
        self.api_base = ""
        self.api_key = ""
        self.model = model_name
        self.temperature = temperature
        self.llm_provider = self
        self._sem = asyncio.Semaphore(concurrent)

    async def aask(self, prompt: str, **kwargs) -> str:
        loop = asyncio.get_running_loop()

        def _gen():
            outs = self._engine.generate([prompt], self._sp)
            if not outs or not outs[0].outputs:
                return ""
            return outs[0].outputs[0].text

        async with self._sem:
            try:
                short_prompt = prompt[:400].replace("\n", " ")
                logger.info(f"[LocalExec] Prompt: {short_prompt}...")
                resp = await loop.run_in_executor(None, _gen)
                short_resp = (resp or "")[:400].replace("\n", " ")
                logger.info(f"[LocalExec] Response: {short_resp}...")
                return resp
            except Exception as e:
                logger.info(f"[LocalExec] Engine error: {e}")
                return ""


# ---------------------------- Helpers ----------------------------
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data(file_path):
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_model(is_first, generator_model, finetuned_model_name, num_gpu):
    model_id = generator_model if is_first else finetuned_model_name
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=num_gpu,
        gpu_memory_utilization=0.90,
        max_model_len=3072,
    )
    tokenizer = AutoTokenizer.from_pretrained(generator_model if is_first else finetuned_model_name)
    return llm, tokenizer


def get_sampling_params(generator_temperature):
    return SamplingParams(
        temperature=generator_temperature,
        top_p=0.95,
        max_tokens=512,
        stop=["</graph>"],
    )


def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


# ---------------------- Runnable check per graph ----------------------
async def test_if_runable(
    num_epoch,
    sub_index_i,
    generated_text,
    problem_type,
    PYTHON_START,
    PYTHON_END,
    sim_threshold,
    TEMP_AVOID,
    TEST_PROMPT,
    NO_EXCEPTION_LIST,
    temp_file_dir,
    provider,
):
    try:
        graph_content = re.search(r"<graph>(.*?)</graph>", generated_text, re.DOTALL)
        if graph_content is None:
            logger.info(f"Error for datapoint {sub_index_i}: Format error")
            return False

        class_script = graph_content.group(1).strip()
        extract_graph_script = re.search(r"async def run_workflow\(self\)(.*?)return", class_script, re.DOTALL)
        if extract_graph_script is None:
            logger.info(f"Error for datapoint {sub_index_i}: Format error")
            return False

        extract_graph_script = extract_graph_script.group(1).strip()
        extract_TEMP_AVOID = re.search(
            r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL
        ).group(1).strip()
        similar_score = similarity_ratio(extract_graph_script, extract_TEMP_AVOID)
        if similar_score >= sim_threshold:
            logger.info(f"Error for datapoint {sub_index_i}: Nearly no modification")
            return False

        python_script = PYTHON_START + class_script + PYTHON_END
        graph_file_path = f"{temp_file_dir}"
        for no_exception_char in NO_EXCEPTION_LIST:
            if no_exception_char in python_script:
                logger.info(
                    f"Error for datapoint {sub_index_i}: Contain no_exception_char {no_exception_char}"
                )

                return False

        file_name = f"/graph_{num_epoch}_{sub_index_i}.py"
        async with aiofiles.open(graph_file_path + file_name, mode="w") as graph_file:
            await graph_file.write(python_script)

        workflows_path = graph_file_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.graph_{num_epoch}_{sub_index_i}"
        graph_module = __import__(graph_module_name, fromlist=[""])
        graph_class = getattr(graph_module, "Workflow")

        backup_create = getattr(graph_module, "create", None)
        setattr(graph_module, "create", lambda cfg: provider)

        if problem_type is None:
            graph_instance = graph_class(config=provider, problem=TEST_PROMPT)
        else:
            graph_instance = graph_class(config=provider, problem=TEST_PROMPT[problem_type])

        await graph_instance()

        if backup_create is not None:
            setattr(graph_module, "create", backup_create)

        os.remove(graph_file_path + file_name)
        return True

    except Exception as e:
        try:
            os.remove(graph_file_path + f"/graph_{num_epoch}_{sub_index_i}.py")
        except Exception:
            pass
        logger.info(f"Error for datapoint {sub_index_i}: {e}")
        return False


def _save_pickle(path, obj):
    """Blocking helper to save pickle file (run in thread)."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------- Worker function ----------------------
async def worker(
    num_epoch: int,
    q: asyncio.Queue,
    results: dict,
    results_lock: asyncio.Lock,
    save_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,           # <- pass a semaphore from caller
    llm_config,
    PYTHON_START,
    PYTHON_END,
    sim_threshold,
    TEMP_AVOID,
    TEST_PROMPT,
    NO_EXCEPTION_LIST,
    temp_file_dir: str,
    provider,
    save_interval: int = 50,   # How often to checkpoint
):
    """
    Robust async worker for validating and checkpointing generated graphs.
    Processes items from the queue until a sentinel (None) is received.
    Uses try/finally to ensure q.task_done() is called exactly once per get().
    """

    processed_count = 0

    while True:
        item = await q.get()
        try:
            # If sentinel received, exit loop (but still call q.task_done() in finally)
            if item is None:
                logger.info("[Worker] Received sentinel -> exiting.")
                break

            i, sub_index_i, generated_text, problem_type = item

            try:
                # Limit concurrency for heavy tasks (file I/O, imports, model exec)
                async with semaphore:
                    ok = await test_if_runable(
                        num_epoch=num_epoch,
                        sub_index_i=sub_index_i,
                        generated_text=generated_text,
                        problem_type=problem_type,
                        PYTHON_START=PYTHON_START,
                        PYTHON_END=PYTHON_END,
                        sim_threshold=sim_threshold,
                        TEMP_AVOID=TEMP_AVOID,
                        TEST_PROMPT=TEST_PROMPT,
                        NO_EXCEPTION_LIST=NO_EXCEPTION_LIST,
                        temp_file_dir=temp_file_dir,
                        provider=provider,
                    )
            except Exception as e:
                logger.error(f"[Worker] Fatal error while testing index {sub_index_i}: {e}")
                logger.debug(traceback.format_exc())
                ok = False

            # Safely update results
            async with results_lock:
                results[i] = ok
                processed_count += 1
                total_done = len(results)

            # Periodic checkpoint saving (guarded by save_lock)
            if total_done % save_interval == 0:
                checkpoint_path = os.path.join(
                    temp_file_dir, f"fail_checkpoint_epoch_{num_epoch}.pkl"
                )

                async with save_lock:
                    try:
                        os.makedirs(temp_file_dir, exist_ok=True)
                        await asyncio.to_thread(_save_pickle, checkpoint_path, results)
                        logger.info(
                            f"[Checkpoint] Saved progress → {checkpoint_path} ({total_done} processed)"
                        )
                    except Exception as e:
                        logger.warning(f"[Checkpoint] Save failed at {total_done}: {e}")

        finally:
            # Ensure exactly one task_done() per q.get()
            q.task_done()

    logger.info(f"[Worker] Finished after processing {processed_count} items.")


# ---------------------- Main controller ----------------------
async def get_fail_list(
    num_epoch,
    sub_index,
    sub_generated_results,
    sub_type_list,
    llm_config,
    max_concurrent_tasks,
    PYTHON_START,
    PYTHON_END,
    sim_threshold,
    TEMP_AVOID,
    TEST_PROMPT,
    NO_EXCEPTION_LIST,
    temp_file_dir,
    provider,
    save_interval: int = 50,
):
    q = asyncio.Queue()
    results = {}
    results_lock = asyncio.Lock()
    save_lock = asyncio.Lock()

    # ✅ Proper checkpoint loading
    checkpoint_path = os.path.join(temp_file_dir, f"fail_checkpoint_epoch_{num_epoch}.pkl")
    results = {}

    if os.path.exists(checkpoint_path):
        try:
            def load_pickle(path):
                with open(path, "rb") as f:
                    return pickle.load(f)

            results = await asyncio.to_thread(load_pickle, checkpoint_path)
            if results:
                last_done = max(results.keys())
                logger.info(f"[Resume] Loaded {len(results)} previous results from {checkpoint_path}")
                logger.info(f"Resuming from around index {last_done + 1}")
            else:
                logger.info("[Resume] Checkpoint found but empty — starting fresh.")
        except Exception as e:
            logger.warning(f"[Resume] Failed to load checkpoint: {e}")
            results = {}

    # ✅ Shared concurrency control
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Populate queue — skip already processed ones
    for i, gen_text in enumerate(sub_generated_results):
        if i in results:
            continue  # already processed
        await q.put((i, sub_index[i], gen_text, sub_type_list[i]))

    # Send sentinel tasks before joining
    for _ in range(max_concurrent_tasks):
        await q.put(None)

    # Spawn workers
    workers = [
        asyncio.create_task(
            worker(
                num_epoch,
                q,
                results,
                results_lock,
                save_lock,
                semaphore,
                llm_config,
                PYTHON_START,
                PYTHON_END,
                sim_threshold,
                TEMP_AVOID,
                TEST_PROMPT,
                NO_EXCEPTION_LIST,
                temp_file_dir,
                provider,
                save_interval,
            )
        )
        for _ in range(max_concurrent_tasks)
    ]

    await q.join()
    await asyncio.gather(*workers)

    # Collect failed indices
    fail_list = [i for i, ok in results.items() if not ok]
    logger.info(f"🧩 {len(fail_list)} tasks failed at epoch {num_epoch}")

    # ✅ Final checkpoint save
    final_path = os.path.join(temp_file_dir, f"fail_checkpoint_epoch_{num_epoch}_final.pkl")
    try:
        os.makedirs(temp_file_dir, exist_ok=True)
        await asyncio.to_thread(_save_pickle, final_path, results)
        logger.info(f"[Final Save] Saved full checkpoint → {final_path}")
    except Exception as e:
        logger.warning(f"[Final Save] Failed to save final checkpoint: {e}")

    return fail_list


# ---------------------- Generate graphs ----------------------
async def generate_graphs(
    llm,
    data,
    sampling_params,
    num_epoch,
    max_concurrent_tasks,
    graph_num,
    benchmark,
    data_set,
    llm_config,
    temp_file_dir,
    provider,  # API provider cho executor,
    chunk_size = 50
):
    prompt_module = importlib.import_module(f"ScoreFlow.scripts.{data_set}.conditions")
    TIME_LIMIT_TEST = prompt_module.TIME_LIMIT_TEST
    PYTHON_END = prompt_module.PYTHON_END.format(time=TIME_LIMIT_TEST)
    sim_threshold = prompt_module.sim_threshold
    PYTHON_START = prompt_module.PYTHON_START
    TEMP_AVOID = prompt_module.TEMP_AVOID
    TEST_PROMPT = prompt_module.TEST_PROMPT
    NO_EXCEPTION_LIST = prompt_module.NO_EXCEPTION_LIST
    START_PORMPT = prompt_module.START_PORMPT
    END_PROMPT = prompt_module.END_PROMPT
    FEW_SHOT = prompt_module.FEW_SHOT_EXAMPLES

    # Build prompts and type list
    prompts = []
    type_list = []

    # select here
    random.seed(42)

    sample = random.sample(data, 1000)

    print(f"We optimizing {len(sample)} samples out of total {len(data)} samples")

    for problem in sample:
        optimize_prompt = START_PORMPT + benchmark.get_graph_input_text(problem) + END_PROMPT + FEW_SHOT
        prompts += [optimize_prompt] * graph_num
        if "problem_type" in problem:
            type_list += [problem["problem_type"]] * graph_num
        else:
            type_list += [None] * graph_num

    # Initial generation
    generated_results = []
    total = len(prompts)
    os.makedirs(temp_file_dir, exist_ok=True)
    generated_results = []

    start_chunk = 0

    # ---------------------- Generate in Chunks ----------------------
    for chunk_start in range(start_chunk * chunk_size, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_prompts = prompts[chunk_start:chunk_end]
        logger.info(f"Generating chunk {chunk_start // chunk_size + 1}: {len(chunk_prompts)} prompts")

        chunk_results = []
        try:
            if hasattr(llm, "generate"):  # sync vLLM-style
                outputs = llm.generate(chunk_prompts, sampling_params)
                for out in outputs:
                    if not out or not getattr(out, "outputs", None) or not out.outputs:
                        chunk_results.append("<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>")
                        continue
                    text = getattr(out.outputs[0], "text", "") or ""
                    if "</graph>" not in text:
                        text += "</graph>"
                    chunk_results.append(text)
            else:  # async OpenAI-style
                semaphore = asyncio.Semaphore(max_concurrent_tasks)

                async def generate_one(prompt):
                    async with semaphore:
                        try:
                            result = await llm.aask(prompt)
                            if not result.strip():
                                return "<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"
                            if "</graph>" not in result:
                                result += "</graph>"
                            return result
                        except Exception as e:
                            logger.warning(f"Error during generation: {e}")
                            return "<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"

                chunk_results = await asyncio.gather(*[generate_one(p) for p in chunk_prompts])

        except Exception as e:
            logger.warning(f"[Generator] Chunk failed: {e}")
            chunk_results = ["<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"] * len(chunk_prompts)

        # ✅ Save after each chunk
        generated_results.extend(chunk_results)
        chunk_idx = chunk_start // chunk_size
        checkpoint_path = os.path.join(temp_file_dir, f"checkpoint_generate_zeroshot_chunk_{chunk_idx}.pkl")

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(generated_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[Checkpoint] Saved chunk {chunk_idx}: {checkpoint_path} ({len(generated_results)} total)")
        except Exception as e:
            logger.warning(f"[Checkpoint] Failed to save chunk {chunk_idx}: {e}")

    logger.info(f"✅ Generation complete. {len(generated_results)} graphs generated.")

    sub_generated_results = generated_results
    sub_type_list = type_list
    sub_index = list(range(len(generated_results)))

    # Iterative refinement with checkpoints
    # while num_epoch >= 0:
    #     semaphore = asyncio.Semaphore(max_concurrent_tasks)
    #     fail_list = await get_fail_list(
    #         num_epoch,
    #         sub_index,
    #         sub_generated_results,
    #         sub_type_list,
    #         llm_config,
    #         max_concurrent_tasks,
    #         PYTHON_START,
    #         PYTHON_END,
    #         sim_threshold,
    #         TEMP_AVOID,
    #         TEST_PROMPT,
    #         NO_EXCEPTION_LIST,
    #         temp_file_dir,
    #         provider,
    #     )

    #     if len(fail_list) == 0:
    #         logger.info(f"✅ All graphs passed at epoch {num_epoch}.")
    #         break

    #     fail_list.sort()
    #     sub_index = [sub_index[i] for i in fail_list]
    #     sub_prompts = [prompts[i] for i in sub_index]
    #     sub_type_list = [type_list[i] for i in sub_index]
    #     logger.info(f"Epoch {num_epoch}: {len(fail_list)} failed graphs. Regenerating...")

    #     sub_generated_results = []

    #     for chunk_start in range(0, len(sub_prompts), chunk_size):
    #         chunk_end = min(chunk_start + chunk_size, len(sub_prompts))
    #         chunk_prompts = sub_prompts[chunk_start:chunk_end]
    #         chunk_results = []
    #         try:
    #             if hasattr(llm, "generate"):  # sync vLLM-style
    #                 outputs = llm.generate(chunk_prompts, sampling_params)
    #                 for out in outputs:
    #                     if not out or not getattr(out, "outputs", None) or not out.outputs:
    #                         chunk_results.append("<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>")
    #                         continue
    #                     text = getattr(out.outputs[0], "text", "") or ""
    #                     if "</graph>" not in text:
    #                         text += "</graph>"
    #                     chunk_results.append(text)
    #             else:  # async OpenAI-style

    #                 async def generate_one(prompt):
    #                     async with semaphore:
    #                         try:
    #                             result = await llm.aask(prompt)
    #                             if not result.strip():
    #                                 return "<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"
    #                             if "</graph>" not in result:
    #                                 result += "</graph>"
    #                             return result
    #                         except Exception as e:
    #                             logger.warning(f"Error during generation: {e}")
    #                             return "<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"

    #                 chunk_results = await asyncio.gather(*[generate_one(p) for p in chunk_prompts])
    #         except Exception as e:
    #             logger.warning(f"[Regenerate] Failed chunk: {e}")
    #             chunk_results = ["<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>"] * len(chunk_prompts)

    #         sub_generated_results.extend(chunk_results)
    #         # ✅ autosave checkpoint for every regeneration chunk
    #         checkpoint_path = os.path.join(temp_file_dir, f"checkpoint_regen_epoch_{num_epoch}_chunk_{chunk_start//chunk_size}.pkl")
    #         with open(checkpoint_path, "wb") as f:
    #             pickle.dump(sub_generated_results, f)
    #         logger.info(f"[Checkpoint] Saved regeneration chunk at {checkpoint_path}")
        
    #     # OK if we reach here then we already regenerated new flock of workflows

    #     for idx in sub_index:
    #         generated_results[idx] = sub_generated_results[sub_index.index(idx)]
    #         type_list[idx] = sub_type_list[sub_index.index(idx)]

    #     # ✅ Save checkpoint after each epoch
    #     try:
    #         os.makedirs(temp_file_dir, exist_ok=True)
    #         checkpoint_path = os.path.join(temp_file_dir, f"checkpoint_epoch_{num_epoch}.pkl")
    #         with open(checkpoint_path, "wb") as f:
    #             pickle.dump(generated_results, f)
    #         logger.info(f"[Checkpoint] Saved intermediate results to {checkpoint_path}")
    #     except Exception as e:
    #         logger.warning(f"[Checkpoint] Failed to save checkpoint at epoch {num_epoch}: {e}")

    #     num_epoch -= 1

    return generated_results


# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="optimize | inference")
    parser.add_argument("--epoch", type=str, required=True)
    args = parser.parse_args()

    data_set = args.dataset
    task_type = args.task
    epoch = int(args.epoch)

    bench_dic = ScoreFlow.params.bench_dic

    if task_type == "optimize":
        data_set_type = "validate"
        output_dir = f"scoreflow_workspace/output_workflow/dataset-{epoch}.pkl"
        output_json_path = f"scoreflow_workspace/output_workflow/dataset-{data_set}-{epoch}.json"
        graph_num = 8
    elif task_type == "inference":
        data_set_type = "test"
        output_dir = f"scoreflow_workspace/output_workflow/dataset-{epoch}-test.pkl"
        output_json_path = f"scoreflow_workspace/output_workflow/dataset-{data_set}-{epoch}-test.json"
        graph_num = 1
    else:
        raise ValueError("task must be 'optimize' hoặc 'inference'")
    
    if data_set == "GSM8K":
        data_set_type = "train"

    file_path = f"data/{bench_dic[data_set]['benchmark_name']}_{data_set_type}.jsonl"
    is_first = (epoch == 0)

    with open("config/config1.yaml", "r") as file:
        config1 = yaml.safe_load(file)

    generator_model = config1["generator"]["model"]
    generator_temperature = config1["generator"]["temperature"]

    os.environ["CUDA_VISIBLE_DEVICES"] = config1["CUDA_VISIBLE_DEVICES"]
    num_gpu = config1["CUDA_VISIBLE_DEVICES"].count(",") + 1
    num_epoch = 100
    max_concurrent_tasks = 10

    temp_file_dir = "scoreflow_workspace/temp_gene_workflow_file"
    finetuned_model_name = f"scoreflow_workspace/finetuned/{epoch}/merged"

    ensure_directory_exists("scoreflow_workspace/output_workflow")
    ensure_directory_exists(temp_file_dir)
    ensure_directory_exists("scoreflow_workspace/finetuned")

    # Benchmark
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")

    # Data
    data = load_data(file_path)

    # Generator: vẫn dùng vLLM local
    # llm, _ = load_model(is_first, generator_model, finetuned_model_name, num_gpu)

    # Generator: API
    llm = APILLMProviderShim(
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        temperature=float(os.environ.get("GENERATOR_TEMPERATURE", 0.0)),
        top_p=0.95,
        max_tokens=1024,
        concurrent=5,
    )
    print(llm.base_url)


    sampling_params = get_sampling_params(generator_temperature)


    # Executor: dùng API
    api_provider = APILLMProviderShim(
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        temperature=float(os.environ.get("EXECUTOR_TEMPERATURE", 0.0)),
        top_p=0.95,
        max_tokens=1024,
        concurrent=5
    )

    print(api_provider.base_url)

    try:
        generated_results = asyncio.run(
            generate_graphs(
                llm=llm,
                data=data,
                sampling_params=sampling_params,
                num_epoch=num_epoch,
                max_concurrent_tasks=max_concurrent_tasks,
                graph_num=graph_num,
                benchmark=benchmark,
                data_set=data_set,
                llm_config=None,
                temp_file_dir=temp_file_dir,
                provider=api_provider,
            )
        )

        # Lưu ra pkl
        final_dataset = []
        for i in range(len(generated_results)):
            final_dataset.append([data[int(i / graph_num)], generated_results[i]])
        with open(output_dir, "wb") as f:
            pickle.dump(final_dataset, f)

        
        # ✅ Save as JSON (safe conversion)
        safe_final_dataset = to_jsonable(final_dataset)
        output_json = os.path.join(output_dir, "final_dataset.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(safe_final_dataset, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved {len(final_dataset)} entries to:\n- {output_dir}\n- {output_json}")

        with open(output_json_path, "w") as f:
            json.dump(final_dataset, f, indent=4)
        
        cprint("Generation Process Done!", color="green")
        
    finally:
        # Đóng API session
        print("Done")

import numpy as np

def to_jsonable(x):
    """Convert common non-JSON types (NumPy, set, bytes, etc.) to safe JSON-compatible ones."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (set, tuple)):
        return list(x)
    elif isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    elif isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


if __name__ == "__main__":
    main()