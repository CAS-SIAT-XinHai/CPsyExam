
![xinhai-cpsyexam](https://github.com/CAS-SIAT-XinHai/CPsyExam/assets/2136700/e2dd98ed-7090-47c7-aeab-cf58dcb23500)
# CPsyExam
CPsyExam: A Chinese Benchmark for Evaluating Psychology using Examinations
## Leaderboard
The following tables display the performance of models in the CPsyExam-KG and CPsyExam-CA
#### CPsyExam-KG
| Model               | MCQA(Zero-shot) | MRQA(Zero-shot) | MCQA(Five-shot) | MRQA(Five-shot) | Average(Zero-shot) | Average(Five-shot) |
|---------------------|-----------------|-----------------|-----------------|-----------------|--------------------|--------------------|
| Open-sourced Models |
| [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) | 49.89 | 9.86 | 53.81 | 14.85 | 39.81 | 44.00 |
| [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) | 53.51 | 5.63 | 55.75 | 5.51  | 41.46 | 43.10 |
| [YI-6B](https://huggingface.co/01-ai/Yi-6B)             | 33.26 | 0.26 | 25.39 | 14.01 | 24.95 | 22.31 |
| [QWEN-14B](https://huggingface.co/Qwen/Qwen-14B-Chat)   | 24.99 | 1.54 | 38.17 | 13.19 | 19.08 | 31.88 |
| [YI-34B](https://huggingface.co/01-ai/Yi-34B)           | 25.03 | 1.15 | 33.69 | 18.18 | 24.95 | 22.31 |
| Psychology-oriented Models |
| [MeChat-6B](https://huggingface.co/qiuhuachuan/MeChat)  | 50.24 | 4.10 | 51.79 | 11.91 | 38.62 | 41.75 |
| [MindChat-7B](https://huggingface.co/X-D-Lab/MindChat-Qwen-7B-v2)|49.25 | 6.27 | 56.92 | 5.51  | 38.43 | 43.97 |
| [MindChat-8B](https://huggingface.co/X-D-Lab/MindChat-Qwen-1_8B)| 26.50 | 0.00 | 26.50 | 0.13  | 19.83 | 19.86 |
| Ours-SFT-6B | 52.95 | 10.50 | 58.77 | 2.94 | 42.26 | 44.71 |
| Api-based Models |
| [ERNIE-Bot](https://yiyan.baidu.com) | 52.48 | 6.66 | 56.10 | 10.37 | 40.94 | 44.58 |
| [ChatGPT](https://openai.com/chatgpt)| 57.43 | 11.14 | 61.53 | 24.71 | 45.78 | 52.26 |
| [ChatGLM](https://chatglm.cn)        | 63.29 | **26.12** | 73.85 | 42.13 | 53.93 | 65.86 |
| [GPT4](https://openai.com/gpt4)      | **76.56** | 10.76 | **78.63** | **43.79** | **59.99** | **69.85** |

#### CPsyExam-CA
| Model               | MCQA(Zero-shot) | MRQA(Zero-shot) | MCQA(Five-shot) | MRQA(Five-shot) | Average(Zero-shot) | Average(Five-shot) |
|---------------------|-----------------|-----------------|-----------------|-----------------|--------------------|--------------------|
| Open-sourced Models |
| [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) | 52.50 | 16.00 | 48.50 | 20.00 | 43.38 | 41.38 |
| [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) | 47.00 | 17.00 | 47.33 | 13.50 | 39.50 | 38.88 |
| [YI-6B](https://huggingface.co/01-ai/Yi-6B)             | 38.83 | 0.00  | 20.00 | 13.25 | 29.12 | 18.63 |
| [QWEN-14B](https://huggingface.co/Qwen/Qwen-14B-Chat)   | 20.33 | 2.00  | 30.00 | 14.00 | 15.75 | 26.00 |
| [YI-34B](https://huggingface.co/01-ai/Yi-34B)           | 20.50 | 0.50  | 22.33 | 8.00  | 15.50 | 19.39 |
| Psychology-oriented Models |
| [MeChat-6B](https://huggingface.co/qiuhuachuan/MeChat)  | 48.67 | 13.50 | 44.83 | 10.50 | 39.86 | 36.25 |
| [MindChat-7B](https://huggingface.co/X-D-Lab/MindChat-Qwen-7B-v2)|40.83 | 5.00  | 33.83 | 4.50  | 31.88 | 26.50 |
| [MindChat-8B](https://huggingface.co/X-D-Lab/MindChat-Qwen-1_8B)| 34.17 | 0.00  | 34.17 | 0.00  | 25.63 | 25.63 | 
| Ours-SFT-6B | 46.50 | 5.50 | 48.67 | 13.00 |
| Api-based Models |
| [ERNIE-Bot](https://yiyan.baidu.com) | 42.50 | 8.50  | 50.67 | 12.00 | 34.00 | 41.00 |
| [ChatGPT](https://openai.com/chatgpt)| 47.33 | 9.00  | 52.67 | 29.50 | 37.75 | 46.88 |
| [ChatGLM](https://chatglm.cn)        | **69.00** | **20.50** | **65.33** | **42.50** | **56.88** | **59.63** |
| [GPT4](https://openai.com/gpt4)      | 60.33 | 13.00 | 64.17 | 39.50 | 48.50 | 58.00 |




## Evaluation
### Usage
```bash
run_example.sh MODEL MODEL_NAME_OR_PATH/API_URL TASK SPLIT GPUS N_SHOT
```
### Parameters
- **MODEL**:The identifier for the model used for evaluation. This can be a local model name or an identifier for online evaluation.

- **MODEL_NAME_OR_PATH**:The path to the model file or directory for local evaluation, or the base URL for the API when performing online evaluations.

- **TASK**:The name of the task for which the evaluation is being performed.

- **SPLIT**:The dataset split to use for evaluation, e.g., train, validation, test.

- **GPUS**:The GPU device ID(s) to use for the evaluation. Set to -1 if no GPU is used.

- **N_SHOT**:The number of shots to use for few-shot learning evaluations. Set this to 0 to disable few-shot learning.

### Local Evaluation Example
```bash
bash evaluations/run_example.sh ChatGLM2 /data/pretrained_models/THUDM/chatglm2-6b ceval validation 0 0
```
### Online Evaluation Example
```bash
bash evaluations/run_example.sh ERNIE-Bot-turbo https://your-openai-proxy.com ceval validation 0 0
```
### LeaderBoard
If you are interesting in add your score to our leaderboard, please send your test answer to my [email](mailto:cpsyexam.0930b@slmail.me), Thank you.

## SFT
### SFT Data Preparation

```bash
PYTHONPATH=../related_repos/LLaMA-Factory/src:../src python cpsyexam_to_sft.py --task cpsyexam --task_dir <llmeval_path> --split train  --save_dir ../data --qa_file <qa_train_path>/cpsyexam_qa.json
```
