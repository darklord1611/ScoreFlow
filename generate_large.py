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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class APILLMProviderShim:
    """Provider gọi API thay vì local vLLM, dùng cho cả generator và executor."""
    def __init__(self, base_url, api_key, model_name="gpt-4o-mini",
                 temperature=0.0, top_p=0.95, max_tokens=1024, concurrent=10):
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
                short_prompt = prompt[:400].replace("\n", " ")
                logger.info(f"[API] Prompt: {short_prompt}...")
               
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
                logger.info(f"[API] Response: {short_resp}...")
                return resp
                   
            except Exception as e:
                logger.error(f"[API] Error: {e}")
                return ""
    
    def generate_batch(self, prompts: list, max_tokens=512):
        """
        Batch generation cho generator (thay thế vLLM).
        Gọi API tuần tự cho từng prompt.
        """
        results = []
        total = len(prompts)
        logger.info(f"[API] Generating batch of {total} prompts...")
        
        for idx, prompt in enumerate(prompts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                    timeout=120
                )
                text = response.choices[0].message.content or ""
                
                # Mock format giống vLLM output
                class MockOutput:
                    def __init__(self, text):
                        self.text = text
                
                class MockResult:
                    def __init__(self, text):
                        self.outputs = [MockOutput(text)]
                
                results.append(MockResult(text))
                
                if (idx + 1) % 10 == 0 or idx == 0:
                    logger.info(f"[API] Progress: {idx + 1}/{total} prompts")
                    
            except Exception as e:
                logger.error(f"[API] Error generating prompt {idx}: {e}")
                # Trả về empty result khi lỗi
                class MockOutput:
                    def __init__(self):
                        self.text = ""
                class MockResult:
                    def __init__(self):
                        self.outputs = [MockOutput()]
                results.append(MockResult())
        
        logger.info(f"[API] Batch generation complete: {len(results)}/{total} results")
        return results


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


def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


# ---------------------- Runnable check per graph ----------------------
async def test_if_runable(
    num_epoch,
    sub_index_i,
    i,
    generated_text,
    problem_type,
    fail_list,
    llm_config,
    semaphore,
    PYTHON_START,
    PYTHON_END,
    sim_threshold,
    TEMP_AVOID,
    TEST_PROMPT,
    NO_EXCEPTION_LIST,
    temp_file_dir,
    provider,
):
    async with semaphore:
        try:
            graph_content = re.search(r"<graph>(.*?)</graph>", generated_text, re.DOTALL)
            if graph_content is None:
                logger.info(f"Error for datapoint {sub_index_i}: Format error")
                fail_list.append(i)
                return False

            class_script = graph_content.group(1).strip()
            extract_graph_script = re.search(r"async def run_workflow\(self\)(.*?)return", class_script, re.DOTALL)
            if extract_graph_script is None:
                logger.info(f"Error for datapoint {sub_index_i}: Format error")
                fail_list.append(i)
                return False

            extract_graph_script = extract_graph_script.group(1).strip()
            extract_TEMP_AVOID = re.search(
                r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL
            ).group(1).strip()
            similar_score = similarity_ratio(extract_graph_script, extract_TEMP_AVOID)
            if similar_score >= sim_threshold:
                logger.info(f"Error for datapoint {sub_index_i}: Nearly no modification")
                fail_list.append(i)
                return False

            python_script = PYTHON_START + class_script + PYTHON_END
            graph_file_path = f"{temp_file_dir}"
            for no_exception_char in NO_EXCEPTION_LIST:
                if no_exception_char in python_script:
                    logger.info(
                        f"Error for datapoint {sub_index_i}: Contain no_exception_char {no_exception_char}"
                    )
                    fail_list.append(i)
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
            fail_list.append(i)
            return False


async def get_fail_list(
    num_epoch,
    sub_index,
    sub_generated_results,
    sub_type_list,
    llm_config,
    semaphore,
    PYTHON_START,
    PYTHON_END,
    sim_threshold,
    TEMP_AVOID,
    TEST_PROMPT,
    NO_EXCEPTION_LIST,
    temp_file_dir,
    provider,
):
    tasks = []
    fail_list = []
    for i, generated_text in enumerate(sub_generated_results):
        tasks.append(
            test_if_runable(
                num_epoch,
                sub_index[i],
                i,
                generated_text,
                sub_type_list[i],
                fail_list,
                llm_config,
                semaphore,
                PYTHON_START,
                PYTHON_END,
                sim_threshold,
                TEMP_AVOID,
                TEST_PROMPT,
                NO_EXCEPTION_LIST,
                temp_file_dir,
                provider,
            )
        )
    await asyncio.gather(*tasks)
    return fail_list


# ---------------------- Generate graphs ----------------------
async def generate_graphs(
    api_generator,
    data,
    num_epoch,
    max_concurrent_tasks,
    graph_num,
    benchmark,
    data_set,
    temp_file_dir,
    executor_provider,
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

    prompts = []
    type_list = []
    for problem in data:
        optimize_prompt = START_PORMPT + benchmark.get_graph_input_text(problem) + END_PROMPT
        prompts += [optimize_prompt] * graph_num
        if "problem_type" in problem:
            type_list += [problem["problem_type"]] * graph_num
        else:
            type_list += [None] * graph_num

    generated_results = []
    try:
        # Dùng API batch generation thay vì vLLM
        outputs = api_generator.generate_batch(prompts, max_tokens=512)
        for out in outputs:
            if not out or not getattr(out, "outputs", None) or not out.outputs:
                generated_results.append("<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>")
                continue
            text = getattr(out.outputs[0], "text", "") or ""
            generated_results.append(text + "</graph>")
    except Exception as e:
        logger.info(f"[generator] API generate failed: {e}")
        for _ in prompts:
            generated_results.append("<graph>\nclass Workflow:\n    async def run_workflow(self):\n        return ''\n</graph>")

    sub_generated_results = generated_results
    sub_type_list = type_list
    sub_index = list(range(len(generated_results)))

    while num_epoch >= 0:
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        fail_list = await get_fail_list(
            num_epoch,
            sub_index,
            sub_generated_results,
            sub_type_list,
            None,
            semaphore,
            PYTHON_START,
            PYTHON_END,
            sim_threshold,
            TEMP_AVOID,
            TEST_PROMPT,
            NO_EXCEPTION_LIST,
            temp_file_dir,
            executor_provider,
        )
        if len(fail_list) == 0:
            break

        fail_list.sort()
        sub_index = [sub_index[i] for i in fail_list]
        sub_prompts = [prompts[i] for i in sub_index]
        sub_type_list = [type_list[i] for i in sub_index]
        logger.info(f"epoch: {num_epoch}, fail list: {fail_list}. sub_index: {sub_index}.")

        # Re-generate failed prompts
        outputs = api_generator.generate_batch(sub_prompts, max_tokens=512)
        sub_generated_results = []
        for out in outputs:
            if out and out.outputs and out.outputs[0].text:
                sub_generated_results.append(out.outputs[0].text + "</graph>")
            else:
                sub_generated_results.append("</graph>")

        for idx in sub_index:
            generated_results[idx] = sub_generated_results[sub_index.index(idx)]
            type_list[idx] = sub_type_list[sub_index.index(idx)]

        num_epoch -= 1

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
        graph_num = 8
    elif task_type == "inference":
        data_set_type = "test"
        output_dir = f"scoreflow_workspace/output_workflow/dataset-{epoch}-test.pkl"
        graph_num = 1
    else:
        raise ValueError("task must be 'optimize' hoặc 'inference'")

    file_path = f"data/{bench_dic[data_set]['benchmark_name']}_{data_set_type}.jsonl"

    num_epoch = 500
    max_concurrent_tasks = 20

    temp_file_dir = "scoreflow_workspace/temp_gene_workflow_file"

    ensure_directory_exists("scoreflow_workspace/output_workflow")
    ensure_directory_exists(temp_file_dir)

    # Benchmark
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")

    # Data
    data = load_data(file_path)

    # Generator: Dùng API thay vì vLLM
    api_generator = APILLMProviderShim(
        base_url="https://api.yescale.io/v1",
        api_key="sk-1BRrBUMuTy3oIc74wSQ2Cw9Kv4zzeSylUFyemcGUejKxupwt",
        model_name="gpt-4o-mini",
        temperature=0.7,  # Cao hơn cho generator để đa dạng
        top_p=0.95,
        max_tokens=2048,  # Dài hơn cho generator
        concurrent=5  # Giới hạn concurrent cho generator
    )

    # Executor: Dùng API với config khác
    api_executor = APILLMProviderShim(
        base_url="https://api.yescale.io/v1",
        api_key="sk-1BRrBUMuTy3oIc74wSQ2Cw9Kv4zzeSylUFyemcGUejKxupwt",
        model_name="gpt-4o-mini",
        temperature=0.0,  # Thấp cho executor để ổn định
        top_p=0.95,
        max_tokens=1024,
        concurrent=10  # Cao hơn cho executor vì task nhỏ hơn
    )

    generated_results = asyncio.run(
        generate_graphs(
            api_generator=api_generator,
            data=data,
            num_epoch=num_epoch,
            max_concurrent_tasks=max_concurrent_tasks,
            graph_num=graph_num,
            benchmark=benchmark,
            data_set=data_set,
            temp_file_dir=temp_file_dir,
            executor_provider=api_executor,
        )
    )

    # Lưu ra pkl
    final_dataset = []
    for i in range(len(generated_results)):
        final_dataset.append([data[int(i / graph_num)], generated_results[i]])
    with open(output_dir, "wb") as f:
        pickle.dump(final_dataset, f)
    cprint("Generation Process Done!", color="green")


if __name__ == "__main__":
    main()