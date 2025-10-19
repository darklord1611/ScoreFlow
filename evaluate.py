import re
import os
import subprocess
import argparse
import importlib
import pickle
import ScoreFlow.params
from termcolor import cprint
from difflib import SequenceMatcher
import numpy as np


# -------------------- Utility Functions --------------------

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def sample_data(p_r_data, N, f):
    prob_list = []
    for i in range(len(p_r_data)):
        w = p_r_data[i]["chosen_score"]
        l = p_r_data[i]["rejected_score"]
        prob_list.append(f(w, l))
    sum_prob = sum(prob_list)
    prob_list = [p / sum_prob for p in prob_list]
    sampled_elements = np.random.choice(p_r_data, size=N, replace=True, p=prob_list)
    return sampled_elements.tolist()


def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


# -------------------- Core Function --------------------

def generate_preference_data(bench_dic, data_set, epoch, task_type, vali_num=1):
    """
    Generate preference data for workflow optimization based on validation results.
    """
    # Load benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="", log_path="")

    # Determine settings based on task type
    if task_type == "inference":
        post = "-test"
        graph_num = 1
    elif task_type == "optimize":
        post = ""
        graph_num = 8
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    # Load data
    with open(f"scoreflow_workspace/output_workflow/dataset-{epoch}{post}.pkl", "rb") as file:
        data = pickle.load(file)

    # Load evaluation scores from multiple validation rounds
    score_list = []
    for i in range(vali_num):
        score_file = f"scoreflow_workspace/output_evaluation/scores-{epoch}-{i}{post}.txt"
        if not os.path.exists(score_file):
            raise FileNotFoundError(f"Missing score file: {score_file}")
        with open(score_file, "rb") as f:
            loaded_list = pickle.load(f)
            score_list += loaded_list

    n_data = int(len(data) / graph_num)

    # Load dataset-specific prompts and thresholds
    prompt_module = importlib.import_module(f"ScoreFlow.scripts.{data_set}.conditions")
    START_PORMPT = prompt_module.START_PORMPT
    END_PROMPT = prompt_module.END_PROMPT
    TEMP_AVOID = prompt_module.TEMP_AVOID
    sim_threshold = prompt_module.sim_threshold

    # Compute averaged scores per graph
    avg_score_list = []
    for i in range(n_data):
        sub_score_list = [s for s in score_list if s[0] == i]
        graph_list = list(set([s[1] for s in sub_score_list]))
        if len(graph_list) != graph_num:
            print("graph_num not right here:", i, len(graph_list))
        for graph in graph_list:
            scores = [s[3] for s in sub_score_list if s[1] == graph]
            if len(scores) != vali_num:
                print("vali_num mismatch:", i, graph, len(scores))
            avg_score = sum(scores) / len(scores)
            avg_score_list.append([i, graph, avg_score])

    # If inference, report solve rate and exit
    if task_type == "inference":
        cprint(f"Average Solve Rate: {sum([s[2] for s in avg_score_list]) / len(avg_score_list):.4f}", color="green")
        return

    # Filter out similar graphs
    list_aviod = []
    question_id = 0
    i = 0
    while question_id < n_data:
        graphs = []
        while True:
            graphs.append(data[i][1])
            i += 1
            if i == len(data):
                break
            if benchmark.get_problem_id(data[i][0]) != benchmark.get_problem_id(data[i - 1][0]):
                break

        for j, graph in enumerate(graphs):
            graph_script = re.search(r"<graph>(.*?)</graph>", graph, re.DOTALL).group(1).strip()
            extract_graph_script = re.search(r"async def run_workflow\(self\)(.*?)return", graph_script, re.DOTALL).group(1).strip()
            extract_TEMP_AVOID = re.search(r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL).group(1).strip()
            sim_score = similarity_ratio(extract_graph_script, extract_TEMP_AVOID)
            if sim_score >= sim_threshold:
                list_aviod.append([question_id, j, extract_graph_script, sim_score])
        question_id += 1

    list_aviod = sorted(list_aviod, key=lambda x: x[3])
    list_id_aviod = [[x[0], x[1]] for x in list_aviod]
    avg_score_list = [x for x in avg_score_list if [x[0], x[1]] not in list_id_aviod]

    # Build preference pairs
    pre_rej_list = []
    for i in range(n_data):
        sub_data = [x for x in avg_score_list if x[0] == i]
        sub_data = sorted(sub_data, key=lambda x: x[2])
        for j in range(len(sub_data)):
            for k in range(j + 1, len(sub_data)):
                w = sub_data[k][2]
                l = sub_data[j][2]
                pre_rej_list.append([i, sub_data[j][1], sub_data[k][1], w, l])

    # Collect preference data
    pre_rej_data = []
    data_id = 0
    for i in range(n_data):
        optimize_prompt = START_PORMPT + benchmark.get_graph_input_text(data[data_id][0]) + END_PROMPT
        graphs = []
        while True:
            graphs.append(data[data_id][1])
            data_id += 1
            if data_id == len(data):
                break
            if benchmark.get_problem_id(data[data_id][0]) != benchmark.get_problem_id(data[data_id - 1][0]):
                break
        sub_data = [x for x in pre_rej_list if x[0] == i]
        for x in sub_data:
            if x[2] >= len(graphs) or x[1] >= len(graphs):
                continue
            chosen = graphs[x[2]]
            rejected = graphs[x[1]]
            if x[3] == x[4]:
                continue
            pre_rej_data.append({
                "prompt": optimize_prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": x[3],
                "rejected_score": x[4],
            })

    # Sample and save
    n_sample = 600 if data_set == "HumanEval" else 2000
    sampled_pre_rej_data = sample_data(pre_rej_data, n_sample, lambda x, y: (x - y) ** 3)

    ensure_directory_exists("scoreflow_workspace/output_preference_data")
    with open(f"scoreflow_workspace/output_preference_data/preference_data-{epoch}.pkl", "wb") as f:
        pickle.dump(sampled_pre_rej_data, f)

    cprint(f"Preference data saved for epoch {epoch}", color="green")


# -------------------- Entry Point --------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate workflow generation results.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--task", type=str, required=True, choices=["inference", "optimize"], help="Task type")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch identifier")
    parser.add_argument("--vali_num", type=int, default=1, help="Number of validation runs (default=1)")
    args = parser.parse_args()

    data_set = args.dataset
    task_type = args.task
    epoch = args.epoch
    vali_num = args.vali_num

    bench_dic = ScoreFlow.params.bench_dic

    # Launch parallel validation runs
    commands = [
        [
            "python",
            "get_scores.py",
            f"--dataset={data_set}",
            f"--task={task_type}",
            f"--epoch={epoch}",
            f"--parallel_id={i}",
        ]
        for i in range(vali_num)
    ]

    processes = [subprocess.Popen(cmd) for cmd in commands]
    for p in processes:
        p.wait()

    cprint("Evaluation Process Done!", color="green")
    generate_preference_data(bench_dic, data_set, epoch, task_type, vali_num=vali_num)


if __name__ == "__main__":
    main()
