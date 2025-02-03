import aiofiles
import json
import os
import importlib
import torch
import vllm
import re
import yaml
import asyncio
import aiofiles
import pickle
import ScoreFlow.params
import argparse
from vllm import LLM, SamplingParams
from termcolor import cprint
from metagpt.logs import logger
from metagpt.configs.models_config import ModelsConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# load dataset
def load_data(file_path):
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data

# load model
def load_model(is_first, generator_model, finetuned_model_name, num_gpu):
    if is_first:
        llm = LLM(
            model = generator_model,
            dtype = "bfloat16",
            tensor_parallel_size = num_gpu,
            gpu_memory_utilization = 0.9,
            max_model_len = 10000
        )
        tokenizer = AutoTokenizer.from_pretrained(generator_model)
    else:
        llm = LLM(
            model = finetuned_model_name,
            dtype = "bfloat16",
            tensor_parallel_size = num_gpu,
            gpu_memory_utilization = 0.9,
            max_model_len = 10000
        )
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
    return llm, tokenizer

def get_sampling_params(generator_temperature):
    sampling_params = SamplingParams(
        temperature = generator_temperature,
        top_p = 0.95,
        max_tokens = 1000,
        stop=["</graph>"]
    )
    return sampling_params

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# simialrity ratio between generated workflow and template workflow
def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# must make sure the generated worklow is executable, and pass certain conditions, including: 1. time limit 2. modification
async def test_if_runable(num_epoch, sub_index_i, i, generated_text, problem_type, fail_list, llm_config, semaphore, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir):
    async with semaphore:
        try:
            # extract python script
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

            # if there is requirement on level of modification (based on template)
            extract_graph_script = extract_graph_script.group(1).strip()
            extract_TEMP_AVOID = re.search(r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL).group(1).strip()
            similar_score = similarity_ratio(extract_graph_script, extract_TEMP_AVOID)
            if similar_score >= sim_threshold:
                logger.info(f"Error for datapoint {sub_index_i}: Nearly no modification")
                fail_list.append(i)
                return False
            
            python_script = PYTHON_START + class_script + PYTHON_END
            graph_file_path = f"{temp_file_dir}"
            
            for no_exception_char in NO_EXCEPTION_LIST:
                if no_exception_char in python_script:
                    logger.info(f"Error for datapoint {sub_index_i}: Contain no_exception_char {no_exception_char}")
                    fail_list.append(i)
                    return False

            # write and load workflow in .py
            async with aiofiles.open(graph_file_path + f"/graph_{num_epoch}_{sub_index_i}.py", mode='w') as graph_file:
                await graph_file.write(python_script)
            workflows_path = graph_file_path.replace("\\", ".").replace("/", ".")
            graph_module_name = f"{workflows_path}.graph_{num_epoch}_{sub_index_i}"
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")

            # test if executable
            if problem_type == None:
                graph_class = graph_class(config=llm_config, problem=TEST_PROMPT)
            else:
                graph_class = graph_class(config=llm_config, problem=TEST_PROMPT[problem_type])
            await graph_class()
            
            os.remove(graph_file_path + f"/graph_{num_epoch}_{sub_index_i}.py")
        
        except Exception as e:
            os.remove(graph_file_path + f"/graph_{num_epoch}_{sub_index_i}.py")
            logger.info(f"Error for datapoint {sub_index_i}: {e}")
            fail_list.append(i)
            return False

async def get_fail_list(num_epoch, sub_index, sub_generated_results, sub_type_list, llm_config, semaphore, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir):
    tasks = []
    fail_list = []
    for i, generated_text in enumerate(sub_generated_results):
        tasks.append(test_if_runable(num_epoch, sub_index[i], i, generated_text, sub_type_list[i], fail_list, llm_config, semaphore, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir))
    await asyncio.gather(*tasks)
    return fail_list

async def generate_graphs(llm, data, sampling_params, num_epoch, max_concurrent_tasks, graph_num, benchmark, data_set, llm_config, temp_file_dir):

    # load prompts and conditions
    prompt_module = importlib.import_module(f"ScoreFlow.scripts.{data_set}.conditions")
    TIME_LIMIT_TEST = prompt_module.TIME_LIMIT_TEST
    PYTHON_END = prompt_module.PYTHON_END.format(time = TIME_LIMIT_TEST)
    sim_threshold = prompt_module.sim_threshold
    PYTHON_START = prompt_module.PYTHON_START
    TEMP_AVOID = prompt_module.TEMP_AVOID
    TEST_PROMPT = prompt_module.TEST_PROMPT
    NO_EXCEPTION_LIST = prompt_module.NO_EXCEPTION_LIST
    START_PORMPT = prompt_module.START_PORMPT
    END_PROMPT = prompt_module.END_PROMPT
    
    # generate prompts
    prompts = []
    type_list = []
    for problem in data:
        optimize_prompt = START_PORMPT + benchmark.get_graph_input_text(problem) + END_PROMPT
        prompts = prompts + [optimize_prompt]*graph_num
        if "problem_type" in problem:
            type_list = type_list + [problem["problem_type"]]*graph_num
        else:
            type_list = type_list + [None]*graph_num
    
    # generate workflows
    outputs = llm.generate(prompts, sampling_params)
    generated_results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_text = generated_text + "</graph>"
        generated_results.append(generated_text)
    
    sub_generated_results = generated_results
    sub_type_list = type_list
    sub_index = [i for i in range(len(generated_results))]
    
    while num_epoch >= 0:
        
        # get failed workflows
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        fail_list = await get_fail_list(num_epoch, sub_index, sub_generated_results, sub_type_list, llm_config, semaphore, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir)
        if len(fail_list) == 0:
            break

        fail_list.sort()
        sub_index = [sub_index[i] for i in fail_list]
        sub_prompts = [prompts[i] for i in sub_index]
        sub_type_list = [type_list[i] for i in sub_index]
        logger.info(f"epoch: {num_epoch}, fail list: {fail_list}. sub_index: {sub_index}.")
        outputs = llm.generate(sub_prompts, sampling_params)
    
        # generate workflows for those failed again
        sub_generated_results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            generated_text = generated_text + "</graph>"
            sub_generated_results.append(generated_text)
    
        # renew the failed ones
        for i in sub_index:
            generated_results[i] = sub_generated_results[sub_index.index(i)]
            type_list[i] = sub_type_list[sub_index.index(i)]

        num_epoch -= 1

    return generated_results



def main():
    # input dataset, task type (optimize/inference), and epoch(0, 1, 2...)
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("--dataset", type=str, required=True, help="Value for Dataset")
    parser.add_argument("--task", type=str, required=True, help="Value for Task")
    parser.add_argument("--epoch", type=str, required=True, help="Value for Epoch")
    args = parser.parse_args()
    data_set = args.dataset
    task_type = args.task
    epoch = int(args.epoch)
    
    bench_dic = ScoreFlow.params.bench_dic
    
    if task_type == "optimize":
        data_set_type = "validate" # dataset type
        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + ".pkl" # output directory for workflow files
        graph_num = 8 # number of workflow to generate
    elif task_type == "inference":
        data_set_type ="test"
        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"
        graph_num = 1
    
    # dataset path
    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"
    
    # if is the first epoch
    if epoch == 0:
        is_first = True
    else:
        is_first = False
    
    # load model configurations
    with open("config/config1.yaml", "r") as file:
        config1 = yaml.safe_load(file)
    
    # config for executor
    llm_config = ModelsConfig.default().get(config1["executor"]["model"])
    llm_config.temperature = config1["executor"]["temperature"]
    # config for generator
    generator_model = config1["generator"]["model"]
    generator_temperature = config1["generator"]["temperature"]

    os.environ["CUDA_VISIBLE_DEVICES"] = config1["CUDA_VISIBLE_DEVICES"]
    num_gpu = config1["CUDA_VISIBLE_DEVICES"].count(",") + 1
    num_epoch = 500
    max_concurrent_tasks = 50
    
    # temp file directory
    temp_file_dir = "scoreflow_workspace/temp_gene_workflow_file"
    # finetuned model directory
    finetuned_model_name = "scoreflow_workspace/finetuned/" + str(epoch) + "/merged"
    
    ensure_directory_exists("scoreflow_workspace/output_workflow")
    ensure_directory_exists("scoreflow_workspace/temp_gene_workflow_file")
    ensure_directory_exists("scoreflow_workspace/finetuned")
    
    # import corresponding benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")

    data = load_data(file_path)

    # load generator
    llm, _ = load_model(is_first, generator_model, finetuned_model_name, num_gpu)
    sampling_params = get_sampling_params(generator_temperature)

    # generate workflow
    generated_results = asyncio.run(generate_graphs(llm, data, sampling_params, num_epoch, max_concurrent_tasks, graph_num, benchmark, data_set, llm_config, temp_file_dir))

    # output generated workflows
    final_dataset = []
    for i in range(len(generated_results)):
        final_dataset.append([data[int(i/graph_num)], generated_results[i]])
    with open(output_dir, "wb") as f:
        pickle.dump(final_dataset, f)
    cprint("Generation Process Done!", color = "green")

if __name__ == "__main__":
    main()
