import pickle
import argparse
import os

def convert_pickle_to_txt(input_path):
    # Đảm bảo file tồn tại
    if not os.path.exists(input_path):
        print(f"[❌] File '{input_path}' không tồn tại.")
        return

    # Đặt tên file output
    output_path = input_path.replace(".pkl", "_readable.txt").replace(".txt", "_readable.txt")

    # Đọc file pickle
    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[⚠️] Lỗi khi đọc file pickle: {e}")
        return

    # Ghi dữ liệu ra file text
    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(data):
            f.write(f"{i}: {row}\n")

    print(f"[✅] Đã chuyển xong: {output_path}")
    print(f"[ℹ️] Tổng số dòng ghi ra: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pickle score file to readable text")
    parser.add_argument("--input", type=str, required=True, help="Path to the pickle file (e.g. scores-1-0.txt or .pkl)")
    args = parser.parse_args()

    convert_pickle_to_txt(args.input)
