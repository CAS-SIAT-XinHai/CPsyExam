#!/usr/bin/env bash
WORK_DIR=$(dirname $(dirname $(readlink -f $0)))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"

MODEL=$1
MODEL_NAME_OR_PATH=$2
DATASET=$3
GPUS=$4

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
else
  echo "$MODEL is not supported"
  exit
fi

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/logs.txt
exec &> >(tee -a "$log_file")

CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src python "$WORK_DIR"/sft/train_bash.py \
    --stage sft \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_train \
    --dataset "$DATASET" \
    --template $TEMPLATE \
    --finetuning_type lora \
    --lora_target $LORA_TARGET \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
