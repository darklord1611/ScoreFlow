#!/bin/bash

# ===============================
# ScoreFlow Optimization Script
# ===============================
# Cách chạy:
#   ./run_scoreflow.sh <DATASET> <NUM_EPOCHS>
# Ví dụ:
#   ./run_scoreflow.sh HumanEval 3
# Nếu không truyền tham số, mặc định: DATASET=MATH, NUM_EPOCHS=3
# ===============================

# Nhận tham số từ command line (hoặc dùng mặc định)
DATASET=${1:-"MATH"}
NUM_EPOCHS=${2:-3}

echo "==============================================="
echo "🚀 Starting ScoreFlow optimization"
echo "📊 Dataset: ${DATASET}"
echo "🔁 Epochs: ${NUM_EPOCHS}"
echo "==============================================="

# Lặp qua từng epoch
for ((i=0; i<${NUM_EPOCHS}; i++))
do
    echo ""
    echo "==============================================="
    echo "🔹 Epoch ${i} / ${NUM_EPOCHS}"
    echo "==============================================="

    echo "➡️ Step 1: Generating workflows..."
    python generate.py --dataset=${DATASET} --task=optimize --epoch=${i}
    if [ $? -ne 0 ]; then
        echo "❌ Error in generate.py at epoch ${i}. Exiting."
        exit 1
    fi

    echo ""
    echo "➡️ Step 2: Evaluating workflows..."
    python evaluate.py --dataset=${DATASET} --task=optimize --epoch=${i}
    if [ $? -ne 0 ]; then
        echo "❌ Error in evaluate.py at epoch ${i}. Exiting."
        exit 1
    fi

    echo ""
    echo "➡️ Step 3: Optimizing with Score-DPO..."
    accelerate launch --num_processes=1 optimize.py --epoch=${i}
    if [ $? -ne 0 ]; then
        echo "❌ Error in optimize.py at epoch ${i}. Exiting."
        exit 1
    fi

    echo ""
    echo "✅ Finished Epoch ${i}"
done

echo ""
echo "==============================================="
echo "🏁 All ${NUM_EPOCHS} epochs completed successfully!"
echo "You can now run inference with:"
echo "  python generate.py --dataset=${DATASET} --task=inference --epoch=$((${NUM_EPOCHS}-1))"
echo "  python evaluate.py --dataset=${DATASET} --task=inference --epoch=$((${NUM_EPOCHS}-1))"
echo "==============================================="
