import argparse
import os
import json
import pickle

FIELD_NAMES = [
    "question_id",
    "graph_id",
    "rep_id",
    "score",
    "prompt",
    "pred_code",
    "log",
    "correct_solution",
]

def load_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")

def convert_list_of_lists(records):
    out = []
    for rec in records:
        # Map thẳng theo vị trí, không check len
        obj = {
            "question_id": rec[0],
            "graph_id": rec[1],
            "rep_id": rec[2],
            "score": float(rec[3]),
            "prompt": rec[4],
            "pred_code": rec[5],
            "log": rec[6],
            # "correct_solution": rec[7],
        }
        out.append(obj)
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert score list-of-lists (PKL/JSON) sang JSON có tên trường.")
    ap.add_argument("--in", dest="inp", required=True, help="Đường dẫn input (.pkl hoặc .json)")
    ap.add_argument("--out", dest="outp", default=None, help="Đường dẫn output .json (mặc định: cùng tên)")
    args = ap.parse_args()

    data = load_any(args.inp)

    # Kỳ vọng data là list các list 8 phần tử; map trực tiếp theo vị trí
    converted = convert_list_of_lists(data)

    out_path = args.outp or (os.path.splitext(args.inp)[0] + "_named.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    main()
