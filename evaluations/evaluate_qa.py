import os
from openai import OpenAI
import json
from tqdm import tqdm
import re
llm_prompt = """
    ## Role
    作为一名心理学领域的资深专家，你应具备以下特质和能力：
    1. 广泛的心理学理论知识：掌握各种心理学流派的理论和实践。
    2. 深刻的人类行为理解：能够解读复杂的行为模式和心理过程。
    3. 分析和判断能力：基于案例细节，快速准确地进行心理分析和诊断。
    4. 临床经验：具有丰富的临床实践经验，能够处理各种心理问题和状况。
    5. 伦理观念：遵循心理学专业的伦理准则，确保患者的隐私和福祉。

    ## Rules
    1. 你是一位经验丰富的心理学专家。
    2. 你的任务是根据提供的信息，使用你的专业知识和分析能力来解答主观题。
    3. 题目将涉及心理学的各个方面，你需要利用你的专业知识来深入分析并提供详细的答案。
    4. 如果题目信息不足以做出充分分析，你需要根据你的专业经验，构建最可能的情景来提供一个深入的答案。

    ## Initialization
    作为角色 <Role>，严格遵守 <Rules>，解答以下关于“{subject}”考试的{question_type}题,根据提供的主观题题干"{question}"，利用您的专业知识深入分析并提供详细的答案。我们依赖您的专业判断，以确保提供的答案既准确又充分体现心理学的专业知识。请注意，您的答案将被用于评分，因此请确保您的答案是准确的。

    答案格式为：“{{您的答案}}”。
"""
eval_prompt = """
## Task
您需要根据提供的标准答案内容，给出一个分数。

## Rule
1. 评分仅基于标准答案的内容，不考虑任何外部信息或GPT-4的预先知识。
2. 分数范围为0到100，100分代表完全符合标准答案，0分代表完全不符合。

## Evaluation
- 请仅根据标准答案的内容进行评分，考虑其清晰度、完整性和相关性。

## Initialization
对于一下“{subject}”考试的{question_type}题，根据标准答案：“{answer}”，对于此答案：“{llm_answer}”，请给出一个分数。

分数：“{{分数}}”。
"""


# 加载测试数据
def load_test_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        questions = json.load(file)
    return questions


# 使用openai库发送请求到LLM API（示例，需要根据实际API调整）
def get_answer_from_llm_api(question, model_name, llm_api_key, llm_api_base):
    llm_client = OpenAI(
        # This is the default and can be omitted
        api_key=llm_api_key,
        base_url=llm_api_base,
    )
    full_prompt = [{
        "role": "user",
        "content": llm_prompt.format(
            subject=question["subject_name"],
            question_type=question["question_type"],
            question=question["question"],
        ),
    }]
    response = llm_client.chat.completions.create(
        messages=full_prompt,
        model=model_name,
        temperature=0.0,
    )
    return json.loads(response.model_dump_json())["choices"][0]["message"]["content"]


# 使用openai库发送请求到GPT-4接口（评分）
def get_score_from_gpt4(question, llm_answer, gpt4_api_key, gpt4_api_base):
    llm_client = OpenAI(
        # This is the default and can be omitted
        api_key=gpt4_api_key,
        base_url=gpt4_api_base,
    )
    full_prompt = [{
        "role": "user",
        "content": eval_prompt.format(
            answer=question["answer"],
            llm_answer=llm_answer,
            subject=question["subject_name"],
            question_type=question["question_type"],
        ),
    }]
    response = llm_client.chat.completions.create(
        messages=full_prompt,
        model="gpt-4-0125-preview",
        temperature=0.0,
    )
    return json.loads(response.model_dump_json())["choices"][0]["message"]["content"]


# 保存结果到文件
def save_results(questions, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)


# 主逻辑
def main():
    test_questions = load_test_questions(
        "evaluations/llmeval/cpsyexam/cpsyexam_qa/test/cpsyexam_qa.json"
    )
    # llm_api_key = os.getenv("LLM_API_KEY")
    # llm_api_base = "https://one-api.chillway.me/v1/"
    llm_api_key = os.getenv("GPT4_API_KEY")
    llm_api_base = "https://kkkc.net/v1/"
    gpt4_api_key = os.getenv("GPT4_API_KEY")
    gpt4_api_base = "https://kkkc.net/v1/"
    model_name = "gpt-3.5-turbo-16k"

    for question in tqdm(test_questions, desc="Evaluating"):
        llm_answer = get_answer_from_llm_api(
            question, model_name, llm_api_key, llm_api_base
        )
        score = get_score_from_gpt4(question, llm_answer, gpt4_api_key, gpt4_api_base)
        pattern = r"\b(100|[1-9]?[0-9])\b"
        score = re.search(pattern, score).group(0)
        question["llm_answer"] = llm_answer
        question["score"] = score
        
        # print(llm_answer, score)

    save_results(
        test_questions,
        "evaluations/llmeval/cpsyexam/cpsyexam_qa/output/" + model_name + "_qa.json",
    )


if __name__ == "__main__":
    main()
