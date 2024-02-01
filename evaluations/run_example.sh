#!/usr/bin/env bash
WORK_DIR=$(dirname $(dirname $(readlink -f $0)))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"

MODEL=$1
MODEL_NAME_OR_PATH=$2
TASK=$3
SPLIT=$4
GPUS=$5
N_SHOT=$6

OUTPUT_DIR="${WORK_DIR}"/output/${UUID}

#LLaMA	7B/13B/33B/65B	q_proj,v_proj	-
#LLaMA-2	7B/13B/70B	q_proj,v_proj	llama2
#BLOOM	560M/1.1B/1.7B/3B/7.1B/176B	query_key_value	-
#BLOOMZ	560M/1.1B/1.7B/3B/7.1B/176B	query_key_value	-
#Falcon	7B/40B	query_key_value	-
#Baichuan	7B/13B	W_pack	baichuan
#Baichuan2	7B/13B	W_pack	baichuan2
#InternLM	7B/20B	q_proj,v_proj	intern
#Qwen	7B/14B	c_attn	chatml
#XVERSE	13B	q_proj,v_proj	xverse
#ChatGLM2	6B	query_key_value	chatglm2
#ChatGLM3	6B	query_key_value	chatglm3
#Phi-1.5	1.3B	Wqkv	-
TEMPLATE="default"
LORA_TARGET="q_proj,v_proj"
SCRIPT=evaluate_local.py
if [[ $MODEL == LLaMA-2 ]]; then
  TEMPLATE=llama2
elif [[ $MODEL == BLOOM ]]; then
  LORA_TARGET="query_key_value"
elif [[ $MODEL == BLOOMZ ]]; then
  LORA_TARGET="query_key_value"
elif [[ $MODEL == Falcon ]]; then
  LORA_TARGET="query_key_value"
elif [[ $MODEL == Baichuan ]]; then
  LORA_TARGET="W_pack"
  TEMPLATE="baichuan"
elif [[ $MODEL == Baichuan2 ]]; then
  LORA_TARGET="W_pack"
  TEMPLATE="baichuan2"
elif [[ $MODEL == InternLM ]]; then
  TEMPLATE="intern"
elif [[ $MODEL == Qwen ]]; then
  LORA_TARGET="c_attn"
  TEMPLATE="chatml"
elif [[ $MODEL == XVERSE ]]; then
  TEMPLATE="xverse"
elif [[ $MODEL == ChatGLM2 ]]; then
  LORA_TARGET="query_key_value"
  TEMPLATE="chatglm2"
elif [[ $MODEL == ChatGLM3 ]]; then
  LORA_TARGET="query_key_value"
  TEMPLATE="chatglm3"
elif [[ $MODEL == "Phi-1.5" ]]; then
  LORA_TARGET="Wqkv"
elif [[ $MODEL == "Yi" ]]; then
  TEMPLATE="yi"
elif [[ $MODEL == "ERNIE-Bot-turbo" ]]; then
  SCRIPT=evaluate_api.py
elif [[ $MODEL == "chatglm_turbo" ]]; then
  SCRIPT=evaluate_api.py
elif [[ $MODEL == "gpt-3.5-turbo" ]]; then
  SCRIPT=evaluate_api.py
else
  echo "$MODEL is not supported"
  exit
fi

LANG='zh'

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/logs.txt
exec &> >(tee -a "$log_file")

#    --checkpoint_dir path_to_checkpoint \
if [ "$SCRIPT" == "evaluate_local.py" ]; then
  CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src python "${WORK_DIR}"/evaluations/$SCRIPT \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --finetuning_type full \
    --template $TEMPLATE \
    --task "$TASK" \
    --task_dir "${WORK_DIR}"/evaluations/llmeval \
    --split "$SPLIT" \
    --lang "$LANG" \
    --n_shot "$N_SHOT" \
    --save_dir "$OUTPUT_DIR"/"$TASK" \
    --batch_size 4
else
  # export API_KEY=xxxxx
  CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src python "${WORK_DIR}"/evaluations/$SCRIPT \
    --task "$TASK" \
    --task_dir "${WORK_DIR}"/evaluations/llmeval \
    --split "$SPLIT" \
    --lang "$LANG" \
    --n_shot "$N_SHOT" \
    --save_dir "$OUTPUT_DIR"/"$TASK" \
    --model_name "$MODEL" \
    --api_base "$MODEL_NAME_OR_PATH"
fi

echo "$log_file"
