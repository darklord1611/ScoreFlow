import pickle
import os

# Đường dẫn file pkl
pkl_path = "/cm/archive/tungnvt5_node3/MASS/ScoreFlow-main/scoreflow_workspace/output_workflow/dataset-0.pkl"

# File txt output
txt_output_path = os.path.splitext(pkl_path)[0] + ".txt"

# Đọc file pkl
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} items from {os.path.basename(pkl_path)}")

# Ghi nội dung sang file txt
with open(txt_output_path, "w", encoding="utf-8") as out:
    out.write(f"Total items: {len(data)}\n\n")
    for i, item in enumerate(data):
        out.write(f"--- Item {i} ---\n")
        out.write(f"Question / Problem ID:\n{item[0]}\n\n")
        out.write("Generated Workflow Text:\n")
        out.write(item[1])
        out.write("\n" + "=" * 100 + "\n\n")

print(f"💾 Saved all items to: {txt_output_path}")

# In thử 3 phần tử đầu ra terminal
for i, item in enumerate(data[:3]):
    print(f"\n--- Item {i} ---")
    print("Question / Problem ID:", item[0])
    print("Generated Workflow Text (truncated):")
    print(item[1][:500], "..." if len(item[1]) > 500 else "")
