
![xinhai-cpsyexam](https://github.com/CAS-SIAT-XinHai/CPsyExam/assets/2136700/e2dd98ed-7090-47c7-aeab-cf58dcb23500)
# CPsyExam
CPsyExam: A Chinese Benchmark for Evaluating Psychology using Examinations

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
bash evaluations/run_example.sh ERNIE-Bot-turbo https://one-api.chillway.me/v1/ ceval validation 0 0
```