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
from metagpt.configs.models_config import ModelsConfig
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

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

@retry(stop=stop_after_attempt(1), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
async def _configure_graph(graph, llm_config, problem = None):
    return graph(config=llm_config, problem = problem)

# get postprocessor function
def load_postprocessor(workflows_path: str, name):
    workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
    ext_module_name = f"{workflows_path}." + name
    try:
        ext_module = __import__(ext_module_name, fromlist=[""])
        ext_class = getattr(ext_module, "Workflow")
        return ext_class
    except ImportError as e:
        logger.info(f"Error loading graph: {e}")
        raise

async def _configure_postprocessor(extraction, llm_config):
    return extraction(llm_config=llm_config)

async def get_scores(work_dir, data_set, llm_config, max_concurrent_tasks, i, question_start, question_end, post_dir, use_judger, use_extraction, data, benchmark):
    vali_num = 3
    prompt_module = importlib.import_module(f"ScoreFlow.scripts.{data_set}.conditions")
    TIME_LIMIT = prompt_module.TIME_LIMIT
    PYTHON_END = prompt_module.PYTHON_END.format(time = TIME_LIMIT)
    PYTHON_START = prompt_module.PYTHON_START
    TEMP_AVOID = prompt_module.TEMP_AVOID
    
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    graph_scores = []
    tasks = []
    extraction = load_postprocessor(post_dir, "extraction")
    judger = load_postprocessor(post_dir, "judger")
    if use_judger:
        configured_judger = await _configure_postprocessor(judger, llm_config)
    else:
        configured_judger = None
    if use_extraction:
        configured_extraction = await _configure_postprocessor(extraction, llm_config)
    else:
        configured_extraction = None
    
    for question_id in range(question_start, question_end):
        problem = data[i][0]
        graph = []
        while (1):
            graph.append(data[i][1])
            i += 1
            if i == len(data):
                break
            i_id = benchmark.get_problem_id(data[i][0])
            i_last_id = benchmark.get_problem_id(data[i-1][0])
            if  i_id != i_last_id:
                break

        for graph_id, response_text in enumerate(graph):

            # Extract graph content
            graph_content = re.search(r"<graph>(.*?)</graph>", response_text, re.DOTALL)
            python_script = PYTHON_START + graph_content.group(1).strip() + PYTHON_END
            graph_file_path = f"{work_dir}/graph_{question_id}_{graph_id}.py"
            with open(graph_file_path, mode='w') as graph_file:
                graph_file.write(python_script)
            
            try:
                optimizer_graph = load_graph(question_id, graph_id, work_dir)
                input_text = benchmark.get_input_text(problem)
                configured_graph = await _configure_graph(optimizer_graph, llm_config, input_text)

                async def sem_evaluate(question_id, graph_id, rep_id, problem, configured_extraction, configured_judger, configured_graph):
                    async with semaphore:
                        try:
                            # evaluate workflow
                            results =  await benchmark.evaluate_problem(problem, configured_extraction, configured_judger, configured_graph)
                            graph_scores.append([question_id, graph_id, rep_id, results[3], results[0], results[1], results[2]])
                        except Exception as e:
                            logger.info(f"Error when running: {e}")

                # evaluate vali_num times for each task
                for rep_id in range(vali_num):
                    tasks.append(sem_evaluate(question_id, graph_id, rep_id, problem, configured_extraction, configured_judger, configured_graph))

            except Exception as e:
                logger.info(f"Error when loading graph: {e}")

            os.remove(graph_file_path)

        if i == len(data):
            break
    
    await asyncio.gather(*tasks)

    return graph_scores



def main():

    # input dataset, task type (optimize/inference), and epoch(0, 1, 2...)
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("--dataset", type=str, required=True, help="Value for Dataset")
    parser.add_argument("--task", type=str, required=True, help="Value for Task")
    parser.add_argument("--epoch", type=str, required=True, help="Value for Epoch")
    parser.add_argument("--parallel_id", type=str, required=True, help="Value for parallel_id")
    args = parser.parse_args()
    data_set = args.dataset
    task_type = args.task
    parallel_id = int(args.parallel_id)
    epoch = int(args.epoch)
    
    bench_dic = ScoreFlow.params.bench_dic

    # load model configurations
    with open("config/config1.yaml", "r") as file:
        config1 = yaml.safe_load(file)
    
    # config for executor
    llm_config = ModelsConfig.default().get(config1["executor"]["model"])
    llm_config.temperature = config1["executor"]["temperature"]
    
    difference  = 3000
    
    max_concurrent_tasks = 50
    use_extraction = True

    if task_type == "optimize":
        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + ".pkl" # input directory of workflow files
        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + ".txt" # output directory of evaluation feedback files
        use_judger = False
        graph_num = 8
    elif task_type == "inference":
        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"
        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"
        use_judger = True
        graph_num = 1

    # temp file directory
    work_dir = "scoreflow_workspace/temp_eval_workflow_file"
    
    # import corresponding benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")
    
    post_dir = "ScoreFlow/scripts/" + data_set
    ensure_directory_exists("scoreflow_workspace/temp_eval_workflow_file")
    ensure_directory_exists("scoreflow_workspace/output_evaluation")

    # load workflow data
    with open(input_dir, "rb") as f:
        data = pickle.load(f)

    len_data = int(len(data)/graph_num)
    if parallel_id == 0:
        start = 0
        end = int(len_data/3)
    elif parallel_id == 1:
        start = int(len_data/3)
        end = int(2*len_data/3)
    elif parallel_id == 2:
        start = int(2*len_data/3)
        end = len_data
    
    # get the start position i
    i = 0
    question_id = 0
    while(1):
        if question_id == start:
            break
        while(1):
            i += 1
            i_id = benchmark.get_problem_id(data[i][0])
            i_last_id = benchmark.get_problem_id(data[i-1][0])
            if i_id != i_last_id:
                break
        question_id += 1

    # evaluate workflows to get scores
    score_results = []
    question_start = start
    while question_start <= end:
        question_end = min(question_start + difference, end)
        graph_scores = asyncio.run(get_scores(work_dir, data_set, llm_config, max_concurrent_tasks, i, question_start, question_end, post_dir, use_judger, use_extraction, data, benchmark))
        score_results = score_results + graph_scores
        question_start += difference

    # output score information
    with open(output_dir, 'wb') as f:
        pickle.dump(score_results, f)
    
if __name__ == "__main__":
    main()

