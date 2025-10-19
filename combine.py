#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine ScoreFlow `output_workflow/dataset-*.txt` and
`output_evaluation/*_readable.txt` into a single JSON file.

Usage:
  python combine_workflow_and_scores.py \
    --workflow scoreflow_workspace/output_workflow/dataset-0.txt \
    --scores scoreflow_workspace/output_evaluation/scores-0-0_readable.txt \
    --out combined_results.json
"""

import argparse
import re
import json
import ast
from typing import List, Dict, Any


def split_workflow_items(text: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse workflow text file that uses blocks like:
      --- Item 0 ---
      Question / Problem ID:
      { ... }   <-- python dict printed
      Generated Workflow Text:
      <graph> ... </graph>
    Return mapping item_index -> {"problem": dict, "workflow_text": str, "raw_block": str}
    """
    # Split by item header. Keep header using lookahead.
    parts = re.split(r'(?m)^\s*---\s*Item\s+\d+\s*---\s*$', text)
    # But we want the indices, so find all headers
    headers = re.findall(r'(?m)^\s*---\s*Item\s+(\d+)\s*---\s*$', text)
    items = {}
    # If header not found (maybe file doesn't contain headers), try whole file as item 0
    if not headers:
        # Try to extract problem dict and graph from whole text
        d = parse_one_workflow_block(text)
        items[0] = d
        return items

    # parts[0] is text before first header, usually empty; parts[i] corresponds to header i-1
    # So pair header k with parts[k+1]
    for idx, header in enumerate(headers):
        try:
            block = parts[idx + 1]
        except IndexError:
            block = ""
        item_index = int(header)
        parsed = parse_one_workflow_block(block)
        items[item_index] = parsed
    return items


def parse_one_workflow_block(block: str) -> Dict[str, Any]:
    """From a single block attempt to extract `problem` dict and workflow text."""
    result = {"problem": None, "workflow_text": None, "raw_block": block}

    # 1) try to find a Python dict (starts with '{' and ends with matching '}')
    #    We attempt to locate the first '{' that begins a dict and then find its matching closing '}'.
    dict_match = None
    # find first '{'
    start_pos = block.find('{')
    if start_pos != -1:
        # naive bracket matching to find matching '}'
        depth = 0
        end_pos = -1
        for i in range(start_pos, len(block)):
            ch = block[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        if end_pos != -1:
            dict_text = block[start_pos:end_pos + 1]
            # Try ast.literal_eval to convert to dict
            try:
                problem_dict = ast.literal_eval(dict_text)
                result["problem"] = problem_dict
                dict_match = (start_pos, end_pos + 1)
            except Exception:
                # fallback: keep raw text string
                result["problem"] = None

    # 2) extract Generated Workflow Text: prefer content inside <graph>...</graph> if present
    graph_match = re.search(r'(?s)<graph>(.*?)</graph>', block, re.IGNORECASE)
    if graph_match:
        result["workflow_text"] = graph_match.group(0).strip()  # include tags
    else:
        # fallback: try to find "Generated Workflow Text:" then until a long separator or end
        m = re.search(r'Generated Workflow Text:\s*(.*)', block, re.DOTALL | re.IGNORECASE)
        if m:
            # try to trim trailing separator lines of ====
            text_after = m.group(1).strip()
            text_after = re.split(r'\n=+\n', text_after)[0].strip()
            result["workflow_text"] = text_after
        else:
            # final fallback: use whole block
            result["workflow_text"] = block.strip()

    return result


def extract_score_lists_from_text(text: str) -> List[List]:
    """
    Find all Python-list-like occurrences in the scores readable text and
    convert them to Python objects with ast.literal_eval.

    It returns a list of lists; each list is expected to have the format used
    by ScoreFlow evaluation outputs, e.g.:
      [question_id, graph_id, rep_id, score, code, stdout, error_msg]
    """
    results = []
    # pattern to find top-level [...] groups (non-greedy) that contain at least one comma
    # Use a simple parser: find '[' then attempt to find matching ']' using depth.
    i = 0
    L = len(text)
    while i < L:
        if text[i] == '[':
            depth = 0
            j = i
            while j < L:
                if text[j] == '[':
                    depth += 1
                elif text[j] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < L and text[j] == ']':
                candidate = text[i:j + 1]
                # Heuristic: require at least one comma inside
                if ',' in candidate:
                    try:
                        obj = ast.literal_eval(candidate)
                        if isinstance(obj, (list, tuple)):
                            results.append(list(obj))
                    except Exception:
                        # if literal_eval fails (due to trailing chars etc.), ignore
                        pass
                i = j + 1
                continue
        i += 1
    return results


def combine(workflow_path: str, scores_path: str, out_path: str):
    # read workflow file
    with open(workflow_path, 'r', encoding='utf-8', errors='ignore') as f:
        wf_text = f.read()

    workflows = split_workflow_items(wf_text)

    # read scores readable file
    with open(scores_path, 'r', encoding='utf-8', errors='ignore') as f:
        scores_text = f.read()

    score_lists = extract_score_lists_from_text(scores_text)

    # organize score lists by question_id (assume first element is question_id)
    scores_by_q = {}
    for s in score_lists:
        if len(s) == 0:
            continue
        q = s[0]
        try:
            qidx = int(q)
        except Exception:
            # skip non-integer leading
            continue
        scores_by_q.setdefault(qidx, []).append(s)

    # merge: for each workflow item index create entry
    combined = []
    # choose set of indices: union of workflows keys and score_by_q keys
    all_indices = set(workflows.keys()).union(set(scores_by_q.keys()))
    for idx in sorted(all_indices):
        w = workflows.get(idx, {"problem": None, "workflow_text": None, "raw_block": ""})
        entry = {
            "item_index": idx,
            "problem": w.get("problem"),
            "workflow_text": w.get("workflow_text"),
            "raw_block": w.get("raw_block"),
            "scores": scores_by_q.get(idx, [])
        }
        combined.append(entry)

    # write json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Saved combined {len(combined)} items to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine workflow and scores into a JSON file")
    parser.add_argument("--workflow", required=True, help="Path to dataset-*.txt (workflow output)")
    parser.add_argument("--scores", required=True, help="Path to scores readable txt")
    parser.add_argument("--out", default="combined_results.json", help="Output JSON path")
    args = parser.parse_args()

    combine(args.workflow, args.scores, args.out)


if __name__ == "__main__":
    main()
