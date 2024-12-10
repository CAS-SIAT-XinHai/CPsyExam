#!/usr/bin/env bash
WORK_DIR=$(dirname $(dirname $(readlink -f $0)))
echo "${WORK_DIR}"

MODEL=$1
MODEL_NAME_OR_PATH=$2
TASK=$3
SPLIT=$4
GPUS=$5
N_SHOT=$6

OUTPUT_DIR="${WORK_DIR}"/output/$(uuidgen)

# Set template based on model
case $MODEL in
  "LLaMA-2")
    TEMPLATE="llama2"
    SCRIPT=evaluate_local.py
    ;;
  "Baichuan")
    TEMPLATE="baichuan"
    SCRIPT=evaluate_local.py
    ;;
  "Baichuan2")
    TEMPLATE="baichuan2"
    SCRIPT=evaluate_local.py
    ;;
  "InternLM")
    TEMPLATE="intern"
    SCRIPT=evaluate_local.py
    ;;
  "Qwen")
    TEMPLATE="qwen"
    SCRIPT=evaluate_local.py
    ;;
  "XVERSE")
    TEMPLATE="xverse"
    SCRIPT=evaluate_local.py
    ;;
  "ChatGLM2")
    TEMPLATE="chatglm2"
    SCRIPT=evaluate_local.py
    ;;
  "ChatGLM3")
    TEMPLATE="chatglm3"
    SCRIPT=evaluate_local.py
    ;;
  "Yi")
    TEMPLATE="yi"
    SCRIPT=evaluate_local.py
    ;;
  # API-based models
  "ERNIE-Bot-turbo"|"chatglm_turbo"|"gpt-3.5-turbo"|"gpt-3.5-turbo-16k"|"gpt-4"|"gpt-4-0125-preview")
    SCRIPT=evaluate_api.py
    # For OpenAI models
    if [[ $MODEL == "gpt-"* ]]; then
      if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable is not set"
        exit 1
      fi
      if [ -z "$OPENAI_API_BASE" ]; then
        OPENAI_API_BASE="https://api.openai.com/v1"
      fi
      MODEL_NAME_OR_PATH=$OPENAI_API_BASE
    fi
    ;;
  "Qwen"*)
    SCRIPT=evaluate_api.py
    ;;
  *)
    echo "$MODEL is not supported"
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/logs.txt
exec &> >(tee -a "$log_file")

# Run evaluation based on script type
if [ "$SCRIPT" == "evaluate_local.py" ]; then
  # For local models
  HF_DATASETS_CACHE=${OUTPUT_DIR}/cache CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src:${WORK_DIR}/src \
    python "${WORK_DIR}"/evaluations/$SCRIPT \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --template $TEMPLATE \
    --task "$TASK" \
    --task_dir "${WORK_DIR}"/evaluations/llmeval \
    --split "$SPLIT" \
    --n_shot "$N_SHOT" \
    --save_dir "$OUTPUT_DIR"
else
  # For API-based models
  HF_DATASETS_CACHE=${OUTPUT_DIR}/cache CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src:${WORK_DIR}/src \
    python "${WORK_DIR}"/evaluations/$SCRIPT \
    --task "$TASK" \
    --task_dir "${WORK_DIR}"/evaluations/llmeval \
    --split "$SPLIT" \
    --n_shot "$N_SHOT" \
    --save_dir "$OUTPUT_DIR" \
    --model_name "$MODEL" \
    --api_base "$MODEL_NAME_OR_PATH"
fi

echo "$log_file"
