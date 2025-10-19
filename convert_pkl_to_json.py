import os
import json
import pickle
import argparse
from termcolor import cprint

def convert_pkl_to_json(pkl_path, output_path=None, indent=2):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Không tìm thấy file: {pkl_path}")
    
    # Đọc dữ liệu từ file PKL
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Đặt tên file đầu ra nếu không truyền vào
    if output_path is None:
        output_path = os.path.splitext(pkl_path)[0] + ".json"

    # Convert sang JSON (có thể phải xử lý object không JSON-serializable)
    def safe_convert(obj):
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    # Nếu là list, convert từng phần tử
    if isinstance(data, list):
        data = [safe_convert(x) for x in data]
    elif isinstance(data, dict):
        data = {k: safe_convert(v) for k, v in data.items()}
    else:
        data = safe_convert(data)

    # Ghi ra file JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    cprint(f"✅ Converted: {pkl_path} → {output_path}", "green")


def main():
    parser = argparse.ArgumentParser(description="Convert .pkl files to .json")
    parser.add_argument("--pkl_path", required=True, help="Path tới file .pkl hoặc thư mục chứa các file .pkl")
    parser.add_argument("--output", help="Thư mục đầu ra (tùy chọn)")
    args = parser.parse_args()

    input_path = args.pkl_path
    output_dir = args.output or os.path.dirname(os.path.abspath(input_path))

    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith(".pkl")]
        for f in files:
            src = os.path.join(input_path, f)
            dst = os.path.join(output_dir, os.path.splitext(f)[0] + ".json")
            convert_pkl_to_json(src, dst)
    else:
        convert_pkl_to_json(input_path)

if __name__ == "__main__":
    main()
