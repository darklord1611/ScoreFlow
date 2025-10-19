#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pipeline.sh <DATASET> <TASK> <EPOCH>
# Examples:
#   bash run_pipeline.sh GSM8K optimize 0
#   bash run_pipeline.sh MATH inference 1
#
# Yêu cầu:
# - config/config1.yaml (field generator.model) phải có sẵn
# - generate.py, get_scores.py, convert_to_json.py, convert_to_json_score.py ở cùng repo hiện tại
# - Python env đã cài các dependency cần thiết

DATASET="${1:?Missing DATASET (e.g., GSM8K)}"
TASK="${2:?Missing TASK (optimize|inference)}"
EPOCH="${3:?Missing EPOCH (e.g., 0)}"

# --- Đọc model name từ config/config1.yaml bằng Python ---
GEN_MODEL="LARGE_MODEL"

# Chuẩn hoá tên model để đặt tên folder (thay /, :, khoảng trắng)
MODEL_SAFE="$(echo "$GEN_MODEL" | sed 's#[/ ]#-#g; s#:#-#g')"

# --- Tạo thư mục output theo format <model>_<dataset>_<epoch> ---
OUTDIR="runs/${MODEL_SAFE}_${DATASET}_${EPOCH}"
mkdir -p "$OUTDIR"

echo "[INFO] Generator model: $GEN_MODEL"
echo "[INFO] Output directory: $OUTDIR"
echo "[INFO] Dataset: $DATASET | Task: $TASK | Epoch: $EPOCH"

# --- Chạy generate.py ---
echo "[STEP] Running generate.py..."
python3 generate_large.py \
  --dataset "$DATASET" \
  --task "$TASK" \
  --epoch "$EPOCH" 2>&1 | tee -a "$OUTDIR/run.log"

# File PKL do generate.py sinh theo TASK
if [[ "$TASK" == "optimize" ]]; then
  GEN_PKL_SRC="scoreflow_workspace/output_workflow/dataset-${EPOCH}.pkl"
else
  GEN_PKL_SRC="scoreflow_workspace/output_workflow/dataset-${EPOCH}-test.pkl"
fi

if [[ ! -f "$GEN_PKL_SRC" ]]; then
  echo "[ERROR] Không tìm thấy file sinh từ generate.py: $GEN_PKL_SRC" | tee -a "$OUTDIR/run.log"
  exit 1
fi

# Đổi tên/di chuyển file pkl vào thư mục đích với tên chuẩn
GEN_PKL_DST="${OUTDIR}/${MODEL_SAFE}_${DATASET}_${EPOCH}.pkl"
mv -f "$GEN_PKL_SRC" "$GEN_PKL_DST"
echo "[INFO] Saved generated PKL -> $GEN_PKL_DST" | tee -a "$OUTDIR/run.log"

# --- Chạy get_scores.py (full file PKL) ---
echo "[STEP] Running get_scores.py..."
python3 get_scores.py \
  --pkl_path "$GEN_PKL_DST" \
  --dataset "$DATASET" \
  --task "$TASK" \
  --full 2>&1 | tee -a "$OUTDIR/run.log"

# get_scores.py đặt tên output dựa trên pkl_path + _scores_0.pkl theo mặc định
SCORES_PKL="${OUTDIR}/${MODEL_SAFE}_${DATASET}_${EPOCH}_scores_0.pkl"
if [[ -f "$SCORES_PKL" ]]; then
  echo "[INFO] Saved scores PKL -> $SCORES_PKL" | tee -a "$OUTDIR/run.log"
else
  # fallback: tìm file *_scores_*.pkl trong OUTDIR
  FOUND="$(ls -1 "${OUTDIR}"/*_scores_*.pkl 2>/dev/null || true)"
  if [[ -n "$FOUND" ]]; then
    echo "[INFO] Saved scores PKL ->" | tee -a "$OUTDIR/run.log"
    echo "$FOUND" | tee -a "$OUTDIR/run.log"
    # Lấy file đầu tiên làm SCORES_PKL
    SCORES_PKL="$(echo "$FOUND" | head -n1)"
  else
    echo "[WARN] Không tìm thấy file scores trong $OUTDIR. Kiểm tra log để biết chi tiết." | tee -a "$OUTDIR/run.log"
    exit 1
  fi
fi

echo "[DONE] Pipeline hoàn tất. Tất cả artifacts nằm trong: $OUTDIR" | tee -a "$OUTDIR/run.log"

# ===================== HẬU XỬ LÝ: convert PKL -> JSON =====================
# 1) Convert scores PKL thành JSON
SCORE_JSON="${SCORES_PKL%.pkl}.json"
echo "[STEP] Converting scores PKL to JSON..."
python3 convert_pkl_to_json.py --pkl_path "$SCORES_PKL" 2>&1 | tee -a "$OUTDIR/run.log"

if [[ -f "$SCORE_JSON" ]]; then
  echo "[INFO] Saved score JSON -> $SCORE_JSON" | tee -a "$OUTDIR/run.log"
else
  # fallback: tìm file .json vừa tạo trong OUTDIR có cùng prefix
  CANDIDATE_JSON="$(ls -1 "${OUTDIR}/"*.json 2>/dev/null | grep -m1 "$(basename "${SCORES_PKL%.pkl}")" || true)"
  if [[ -n "$CANDIDATE_JSON" ]]; then
    SCORE_JSON="$CANDIDATE_JSON"
    echo "[INFO] Saved score JSON -> $SCORE_JSON" | tee -a "$OUTDIR/run.log"
  else
    echo "[ERROR] Không tìm thấy file JSON sau bước convert_to_json.py" | tee -a "$OUTDIR/run.log"
    exit 1
  fi
fi

# 2) Chạy convert_to_json_score.py trên file JSON vừa tạo
echo "[STEP] Post-processing score JSON..."
python3 convert_pkl_to_json_score.py --input "$SCORE_JSON" 2>&1 | tee -a "$OUTDIR/run.log"

echo "[ALL DONE] Đã convert & hậu xử lý điểm. Xem thư mục: $OUTDIR" | tee -a "$OUTDIR/run.log"
