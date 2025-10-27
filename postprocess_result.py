#!/usr/bin/env python3

"""
Merges workflow data (problem, level, type, solution) from a PKL or JSON file
into a score file. It builds a lookup table of unique problems and uses
the 'question_id' from the score file as a direct index.
"""

import json
import pickle
import argparse
import os

def load_data(path):
    """
    Helper function to load a PKL or JSON file based on its extension.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: File not found at {path}")

    _, extension = os.path.splitext(path)

    if extension == ".pkl":
        print(f"Detected .pkl file. Loading with pickle...")
        with open(path, "rb") as f:
            return pickle.load(f)
    elif extension == ".json":
        print(f"Detected .json file. Loading with json...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: '{extension}'. Please provide a .pkl or .json file.")


def save_json(data, path):
    """Helper function to save data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(workflow_path, score_path, output_path):
    """
    Main function to load, merge, and save the data.
    """

    # --- 1. Load and De-duplicate Score Data ---
    print(f"Loading score data from: {score_path}")
    # Scores are assumed to be JSON
    with open(score_path, "r", encoding="utf-8") as f:
        score_data = json.load(f)

    merged_scores = []
    seen_keys = set()

    def get_score_key(entry):
        """Creates a unique key for a score entry."""
        return (entry.get("question_id"), entry.get("graph_id"), entry.get("rep_id"))

    for entry in score_data:
        k = get_score_key(entry)
        if k not in seen_keys:
            merged_scores.append(entry)
            seen_keys.add(k)

    original_count = len(score_data)
    merged_count = len(merged_scores)
    print(f"Loaded {original_count} score entries.")
    if original_count > merged_count:
        print(f"Removed {original_count - merged_count} duplicates.")
    print(f"Total unique score entries: {merged_count}\n")


    # --- 2. Load and Re-format Workflow Data (Problem Lookup) ---
    print(f"Loading workflow data from: {workflow_path}")
    workflow_data_raw = load_data(workflow_path) # Uses the loader

    # --- MODIFIED SECTION ---
    # Build a lookup list of unique problems, assuming 8 graphs per problem
    GRAPH_NUM = 8
    problem_lookup = []
    
    if len(workflow_data_raw) % GRAPH_NUM != 0:
        print(f"⚠️ Warning: Workflow data length ({len(workflow_data_raw)}) is not a multiple of {GRAPH_NUM}.")

    # Iterate by problem (hopping by 8), not by graph
    for i in range(0, len(workflow_data_raw), GRAPH_NUM):
        item = workflow_data_raw[i] # Get the first item of the 8-group
        cur = {
            "problem": item[0]["problem"],
            "level": item[0]["level"],
            "type": item[0]["type"],
            "solution": item[0]["solution"],
        }
        problem_lookup.append(cur) # This creates a list of unique problems

    print(f"Loaded {len(workflow_data_raw)} total workflow entries.")
    print(f"Created a lookup table with {len(problem_lookup)} unique problems.\n")
    # --- END MODIFIED SECTION ---


    # --- 3. Merge Data ---
    print("Merging workflow details into score entries...")
    merge_errors = 0

    for item in merged_scores:
        try:
            # --- MODIFIED SECTION ---
            # Use q_id as the direct index into the problem_lookup list
            q_id = item["question_id"]
            index = q_id
            # --- END MODIFIED SECTION ---

            if index >= len(problem_lookup):
                raise IndexError(f"question_id {index} is out of bounds for problem lookup table (len {len(problem_lookup)}).")

            workflow_entry = problem_lookup[index]

            item["problem"] = workflow_entry["problem"]
            item["level"] = workflow_entry["level"]
            item["type"] = workflow_entry["type"]
            item["solution"] = workflow_entry["solution"]

        except (KeyError, TypeError) as e:
            print(f"Error merging item: {item}. Missing 'question_id'. {e}")
            merge_errors += 1
        except IndexError as e:
            print(f"Error merging item: {item}. {e}")
            merge_errors += 1

    if merge_errors == 0:
        print("✅ Merge complete.")
    else:
        print(f"⚠️ Merge complete with {merge_errors} errors.")


    # --- 4. Save Final Merged File ---
    print(f"\nSaving merged file to: {output_path}")
    save_json(merged_scores, output_path)
    print("🎉 Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge ScoreFlow workflow data into a score file."
    )
    parser.add_argument(
        "-w", "--workflow",
        dest="workflow_path",
        required=True,
        help="Path to the workflow PKL or JSON file (e.g., Qwen-Qwen2.5-7B-Instruct_MATH_0.pkl)"
    )
    parser.add_argument(
        "-s", "--score",
        dest="score_path",
        required=True,
        help="Path to the score JSON file (e.g., merged_eval.json)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        required=True,
        help="Path for the final merged output JSON file (e.g., final_scores_with_problems.json)"
    )

    args = parser.parse_args()

    main(args.workflow_path, args.score_path, args.output_path)