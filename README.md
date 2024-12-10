# CPsyExam

## News
- **December 9, 2024**: Official website launched! Visit [http://cpsyexam.xinhai.co](http://cpsyexam.xinhai.co) for more information and updates.
- **November 30, 2024**: Our paper "CPsyExam: Chinese Benchmark for Evaluating Psychology using Examinations" has been accepted by COLING 2025! 🎉
- **Dataset**: Available on Hugging Face at [CAS-SIAT-XinHai/CPsyExam](https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyExam)

![xinhai-cpsyexam](https://github.com/CAS-SIAT-XinHai/CPsyExam/assets/2136700/e2dd98ed-7090-47c7-aeab-cf58dcb23500)



## Leaderboard
Check out our current leaderboard at [http://cpsyexam.xinhai.co/leaderboard](http://cpsyexam.xinhai.co/leaderboard)

## Evaluation
### Usage
```bash
bash evaluations/run_example.sh MODEL MODEL_NAME_OR_PATH TASK SPLIT GPUS N_SHOT
```

### Parameters
- **MODEL**: Model identifier. Supported models:
  - Local Models:
    - LLaMA-2 (template: llama2)
    - Baichuan (template: baichuan)
    - Baichuan2 (template: baichuan2)
    - InternLM (template: intern)
    - Qwen (template: qwen)
    - XVERSE (template: xverse)
    - ChatGLM2 (template: chatglm2)
    - ChatGLM3 (template: chatglm3)
    - Yi (template: yi)
  - API Models:
    - ERNIE-Bot-turbo
    - chatglm_turbo
    - gpt-3.5-turbo
    - gpt-3.5-turbo-16k
    - gpt-4
    - gpt-4-0125-preview
    - Qwen2.5
- **MODEL_NAME_OR_PATH**: 
  - For local models: Path to model files
  - For API models: API base URL (defaults to official endpoints if not specified)
- **TASK**: Name of the evaluation task (e.g., cpsyexam)
- **SPLIT**: Dataset split (e.g., train, validation, test)
- **GPUS**: GPU device IDs (comma-separated, -1 for CPU)
- **N_SHOT**: Number of examples for few-shot learning (0 for zero-shot)

### Examples
#### Local Model Evaluation
```bash
# Evaluate ChatGLM2 model
bash evaluations/run_example.sh ChatGLM2 /path/to/chatglm2-6b cpsyexam test 0 5

# Evaluate LLaMA-2 model with multiple GPUs
bash evaluations/run_example.sh LLaMA-2 /path/to/llama2 cpsyexam test 0,1 3
```

#### API Model Evaluation
```bash
# OpenAI GPT-4
export OPENAI_API_KEY=your_api_key
bash evaluations/run_example.sh gpt-4 https://api.openai.com/v1 cpsyexam test -1 5

# ERNIE-Bot
bash evaluations/run_example.sh ERNIE-Bot-turbo https://your-api-endpoint cpsyexam test -1 5
```

### Output
Results will be saved in `${WORK_DIR}/output/<uuid>/` with the following structure:
- `logs.txt`: Evaluation logs
- `results_${TASK}_${SPLIT}.json`: Evaluation results

### Merging Results
To merge results from multiple evaluation runs:

```bash
python evaluations/merge_answers.py \
--result_dirs results_api results_local \
--output_file merged_results.json 
```

This will combine all result files that follow the pattern `results_cpsyexam_*.json` from the specified directories into a single merged file.

### Submission
To submit your results:
1. Merge your evaluation results using the instructions in the [Merging Results](#merging-results) section
2. Visit our website [http://cpsyexam.xinhai.co](http://cpsyexam.xinhai.co) to submit your results

## SFT
### SFT Data Preparation
```bash
PYTHONPATH=../related_repos/LLaMA-Factory/src:../src python cpsyexam_to_sft.py \
  --task cpsyexam \
  --task_dir <llmeval_path> \
  --split train \
  --save_dir ../data \
  --qa_file <qa_train_path>/cpsyexam_qa.json
```

## Citation
If you find this work helpful, please cite our paper:
```bibtex
@misc{CPsyExam benchmark,
  title={CPsyExam: Chinese Benchmark for Evaluating Psychology using Examinations},
  author={Jiahao Zhao, Jingwei Zhu, Minghuan Tan, Min Yang, Di Yang, 
          Chenhao Zhang, Guancheng Ye, Chengming Li, Xiping Hu},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/CAS-SIAT-XinHai/CPsyExam}}
}
```