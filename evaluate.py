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
    prob_list = [p/sum_prob for p in prob_list]
    sampled_elements = np.random.choice(p_r_data, size=N, replace=True, p = prob_list)
    return sampled_elements.tolist()

def similarity_ratio(str1, str2):
        return SequenceMatcher(None, str1, str2).ratio()

def generate_preference_data(bench_dic, data_set, epoch, task_type):
    
    # load benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="", log_path="")

    if task_type == "inference":
        post = "-test"
        graph_num = 1
    elif task_type == "optimize":
        post = ""
        graph_num = 8

    with open("scoreflow_workspace/output_workflow/dataset-" + epoch + post + ".pkl", 'rb') as file:
        data = pickle.load(file)
    
    with open("scoreflow_workspace/output_evaluation/scores-" + epoch + "-0" + post + ".txt", 'rb') as f:
        loaded_list0 = pickle.load(f)
    with open("scoreflow_workspace/output_evaluation/scores-" + epoch + "-1" + post + ".txt", 'rb') as f:
        loaded_list1 = pickle.load(f)
    with open("scoreflow_workspace/output_evaluation/scores-" + epoch + "-2" + post + ".txt", 'rb') as f:
        loaded_list2 = pickle.load(f)
    score_list = loaded_list0 + loaded_list1 + loaded_list2
    
    vali_num = 3
    n_data = int(len(data)/graph_num)
    
    # load prompts and conditions
    prompt_module = importlib.import_module(f"ScoreFlow.scripts.{data_set}.conditions")
    START_PORMPT = prompt_module.START_PORMPT
    END_PROMPT = prompt_module.END_PROMPT
    TEMP_AVOID = prompt_module.TEMP_AVOID
    sim_threshold = prompt_module.sim_threshold
    
    avg_score_list = []
    for i in range(n_data):
        sub_score_list = [sub_l for sub_l in score_list if sub_l[0] == i]
        graph_list = list(set([sub_l[1] for sub_l in sub_score_list]))
        if len(graph_list) != graph_num:
            print("graph_num not right here: ", i, len(graph_list))
        sub_avg_score_list = []
        for graph in graph_list:
            avg_score = [sub_l[3] for sub_l in sub_score_list if sub_l[1] == graph]
            if len(avg_score) != vali_num:
                print("vali_num not right here:", i, graph)
            avg_score = sum(avg_score)/vali_num
            sub_avg_score_list.append([i, graph, avg_score])
        avg_score_list = avg_score_list + sub_avg_score_list

    if task_type == "inference":
        cprint(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}", color = "green")
        return
    
    # check if requirement holds
    i = 0
    question_id = 0
    list_aviod = []
    while(1):
        if question_id == n_data:
            break
        graphs = []
        while(1):
            graphs.append(data[i][1])
            i += 1
            if i == len(data):
                break
            i_id = benchmark.get_problem_id(data[i][0])
            i_last_id = benchmark.get_problem_id(data[i-1][0])
            if  i_id != i_last_id:
                break
    
        for j in range(len(graphs)):
            graph = graphs[j]
            graph_script = re.search(r"<graph>(.*?)</graph>", graph, re.DOTALL).group(1).strip()
            extract_graph_script = re.search(r"async def run_workflow\(self\)(.*?)return", graph_script, re.DOTALL).group(1).strip()
            extract_TEMP_AVOID = re.search(r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL).group(1).strip()
            similar_score = similarity_ratio(extract_graph_script, extract_TEMP_AVOID)
            if similar_score >= sim_threshold:
                list_aviod.append([question_id, j, extract_graph_script, similar_score])
        question_id += 1
    list_aviod = sorted(list_aviod, key=lambda x: (x[3]))
    list_id_aviod = [[sub_l[0], sub_l[1]] for sub_l in list_aviod]
    avg_score_list = [sub_l for sub_l in avg_score_list if [sub_l[0], sub_l[1]] not in list_id_aviod]

    # organize data structure
    pre_rej_list = []
    for i in range(n_data):
        sub_data = [sub_l for sub_l in avg_score_list if sub_l[0] == i]
        sub_data = sorted(sub_data, key=lambda x: x[2])
        check_list = []
        for j in range(len(sub_data)):
            for k in range(j+1, len(sub_data)):
                w = sub_data[k][2]
                l = sub_data[j][2]
                score_w = w
                score_l = l
                check_list.append([i, sub_data[j][1], sub_data[k][1], score_w, score_l])
        pre_rej_list = pre_rej_list + check_list

    # obtain raw preference data
    i = 0
    data_id = 0
    pre_rej_data = []
    while (i < n_data):
        optimize_prompt = START_PORMPT + benchmark.get_graph_input_text(data[data_id][0]) + END_PROMPT
        graph = []
        while(1):
            graph.append(data[data_id][1])
            data_id += 1
            if data_id == len(data):
                break
            i_id = benchmark.get_problem_id(data[data_id][0])
            i_last_id = benchmark.get_problem_id(data[data_id-1][0])
            if  i_id != i_last_id:
                break
        sub_data = [sub_l for sub_l in pre_rej_list if sub_l[0] == i]
        for sub_l in sub_data:
            if sub_l[2] >= len(graph) or sub_l[1] >= len(graph):
                continue
            chosen = graph[sub_l[2]]
            chosen_score = sub_l[3]
            rejected = graph[sub_l[1]]
            rejected_score = sub_l[4]
            dic = {"prompt": optimize_prompt, "chosen": chosen, "rejected": rejected, "chosen_score": chosen_score, "rejected_score": rejected_score}
            if chosen_score != rejected_score:
                pre_rej_data.append(dic)
        i += 1
    
    # enhance sampling distribution by function d(x, y)
    n_sample = 2000
    if data_set == "HumanEval":
        n_sample = 600
    sampled_pre_rej_data = sample_data(pre_rej_data, n_sample, lambda x, y: (x - y)**3) # d(x, y) = (x - y)^3

    # output preference data
    ensure_directory_exists("scoreflow_workspace/output_preference_data")
    with open("scoreflow_workspace/output_preference_data/preference_data-" + epoch + ".pkl", "wb") as f:
        pickle.dump(sampled_pre_rej_data, f)

def main():

    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("--dataset", type=str, required=True, help="Value for Dataset")
    parser.add_argument("--task", type=str, required=True, help="Value for Task")
    parser.add_argument("--epoch", type=str, required=True, help="Value for Epoch")
    args = parser.parse_args()
    data_set = args.dataset
    task_type = args.task
    epoch = args.epoch

    bench_dic = ScoreFlow.params.bench_dic

    # we use parallelization to speed up the evaluation process.
    commands = [
        ["python", "get_scores.py", f"--dataset={data_set}", f"--task={task_type}", f"--epoch={epoch}", f"--parallel_id=0"],
        ["python", "get_scores.py", f"--dataset={data_set}", f"--task={task_type}", f"--epoch={epoch}", f"--parallel_id=1"],
        ["python", "get_scores.py", f"--dataset={data_set}", f"--task={task_type}", f"--epoch={epoch}", f"--parallel_id=2"]
    ]
    processes = [subprocess.Popen(cmd) for cmd in commands]
    for p in processes:
        p.wait()

    cprint("Evaluation Process Done!", color = "green")
    
    generate_preference_data(bench_dic, data_set, epoch, task_type)


if __name__ == "__main__":
    main()
