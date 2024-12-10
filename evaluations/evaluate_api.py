import argparse
import json
import os
import re
import string
import uuid
from time import sleep
import numpy as np
from tqdm import trange
import random
import time
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIEvaluator:
    def __init__(self, args):
        self.n_shot = args.n_shot
        self.task_dir = args.task_dir
        self.task = args.task
        self.model_name = args.model_name
        self.api_base = args.api_base
        self.save_dir = args.save_dir
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=self.api_base
        )

        # Load the mapping table
        with open(os.path.join(self.task_dir, self.task, 'mapping.json'), 'r', encoding='utf-8') as f:
            self.task_mapping = json.load(f)

        # Generate a UUID folder to save all results
        self.unique_id = str(uuid.uuid4())
        print(f"Results will be saved in: {self.save_dir}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def load_json(self, dataset_dir, dataset_name):
        """
        Used to load the specified train or test dataset
        """
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def eval(self):
        # Iterate through all categories
        for split in self.task_mapping:
            dataset_name = self.task_mapping[split]["name"]
            logger.info(f"Evaluating split: {split} ({dataset_name})")

            # Load dataset train and test splits
            train_dir = os.path.join(self.task_dir, self.task, 'train')
            test_dir = os.path.join(self.task_dir, self.task, 'test')
            try:
                # Load train and test splits
                train_dataset = self.load_json(train_dir, dataset_name)
                test_dataset = self.load_json(test_dir, dataset_name)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for {dataset_name}: {e}")
                continue
            except FileNotFoundError as e:
                logger.error(f"File not found for {dataset_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading {dataset_name}: {e}")
                continue

            if not train_dataset or not test_dataset:
                logger.warning(f"Empty dataset for {split}, skipping")
                continue

            results = []
            # Iterate through samples in the test set
            for i in trange(len(test_dataset), desc=f"Evaluating {split}", position=1, leave=False):
                question_type = test_dataset[i]['question_type']
                subject_name = test_dataset[i].get('subject_name', 'Unknown Subject')
                
                # Filter support_set based on question_type only and ensure they have answers
                typed_train_dataset = [
                    example for example in train_dataset 
                    if example['question_type'] == question_type
                    and 'answer' in example
                ]
                
                # Get examples with answers
                support_set = random.sample(typed_train_dataset, min(self.n_shot, len(typed_train_dataset)))
                
                # Get choices for the question
                choices = [c for c in string.ascii_uppercase if c in test_dataset[i]["options"]]
                row = test_dataset[i]

                # Prepare query and history
                query, history, system = self.format_example_with_choices(
                    target_data=row,
                    support_set=support_set,
                    subject_name=subject_name,
                    choices=choices
                )

                # Create the complete prompt with chat history
                messages = [{"role": "system", "content": system}]
                for user_msg, assistant_msg in history:
                    messages.extend([
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ])
                messages.append({"role": "user", "content": query})

                # Get LLM response with retry mechanism
                response_str = self.get_llm_response(messages, choices, row['question_type'])

                # Extract answer
                ans_list = self.extract_ans(response_str, choices, row['question_type'])

                # If we still got an empty answer after all retries, log it and continue
                if not ans_list:
                    print(f"Warning: Empty answer for question {row.get('id', i)} in {split}")
                    continue

                results.append({
                    'question_id': row['id'],
                    'question': row['question'],
                    'options': row['options'],
                    'prediction': ''.join(ans_list),
                    'n_shot': self.n_shot
                })

            # Save results
            self._save_results(results, split)

    def format_example_with_choices(self, target_data, support_set, subject_name, choices):
        """Format the question and few-shot support set with chat history format."""
        question_type = target_data['question_type']
        query = self.parse_example_with_choices(target_data, choices, subject_name, question_type)
        
        # Create system prompt
        system = '''## Role
作为一名心理学领域的资深专家，你应具备以下特质和能力：
1. 广泛的心理学理论知识掌握各种心理学流派的理论和实践。
2. 深刻的人类行为理解：能够解读复杂的行为模式和心理过程。
3. 分析和判断能力：基于案例细节，快速准确地进行心理分析和诊断。
4. 临床经验：具有丰富的临床实践经验，能够处理各种心理问题和状况。
5. 伦理观念：遵循心理学专业的伦理准则，确保患者的隐私和福祉。

## Rules
1. 你是一位经验丰富的心理学专家。
2. 你的任务是根据提供的信息，使用你的专业知识和分析能力来解答{subject}考试中的{question_type}题。
3. 题目将涉及心理学的各个方面，你需要利用你的专业知识来选择正确答案。
4. 如果题目信息不足以做出判断，你需要根据你的专业经验，假设最可能的情景来选择一个最合理的答案。

## Initialization
作为角色 <Role>，严格遵守 <Rules>，请解答以下关于"{subject}"考试的{question_type}题。请利用您的专业知识，仔细分析每个选项，并选择最符合心理学原理和临床经验的答案。我们依赖您的专业判断，以确保选择最准确、最客观的答案。只需要给出答案，无需任何分析

答案格式为"{{"ans":"您选择的答案"}}".'''.format(
            subject=subject_name,
            question_type="单选题" if question_type == "single" else "多选题"
        )
        
        # Format history from support set
        history = []
        for example in support_set:
            user_msg = self.parse_example_with_choices(example, choices, subject_name, question_type)
            if 'answer' in example:
                assistant_msg = f"答案：{example['answer']}"
                history.append((user_msg, assistant_msg))
        
        return query.strip(), history, system

    def parse_example_with_choices(self, example, choices, subject_name, question_type):
        """
        Create a formatted string with the question and options.
        """
        candidates = [f"\n{ch}. {example['options'].get(ch, '')}" for ch in choices if example['options'].get(ch)]
        question = f"科目：{subject_name}\n题型：{'单选题' if question_type == 'single' else '多选题'}\n问题：{example['question']}"
        # Add answer if it exists in the example
        if 'answer' in example:
            question = f"{question}{''.join(candidates)}\n答案：{example['answer']}"
        else:
            question = f"{question}{''.join(candidates)}"
        return question

    def get_llm_response(self, messages, choices, question_type, max_retries=5, initial_delay=1, max_delay=16):
        """
        Send the complete prompt to the OpenAI API and return the response.
        Implements exponential backoff for retries and ensures a non-empty answer.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.,  # Add temperature for better sampling
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content

                # Check if the response is valid and non-empty
                if result:
                    # Try to parse as JSON first
                    try:
                        json_result = json.loads(result)
                        if isinstance(json_result, dict) and 'ans' in json_result:
                            ans = json_result['ans'].upper()
                            # For single choice, just validate the answer
                            if question_type == "single" or question_type == "单项选择题":
                                if ans in choices:
                                    return json_result
                            # For multiple choice, ensure proper formatting
                            else:
                                ans_list = [a for a in ans if a in choices]
                                if ans_list:
                                    return {"ans": "".join(sorted(ans_list))}
                    except json.JSONDecodeError:
                        pass

                    # If JSON parsing fails, try to extract answer directly
                    ans_list = self.extract_ans(result, choices, question_type)
                    if ans_list:  # If we got a non-empty answer, return it
                        return {"ans": "".join(ans_list)}
                    elif question_type == "single" or question_type == "单项选择题":
                        # For single-choice questions, retry if we couldn't extract an answer
                        print(f"Attempt {attempt + 1}: Failed to extract valid answer for single-choice question. Retrying...")
                    else:
                        # For multi-choice questions, return the result even if we got a single answer
                        return {"ans": "".join(ans_list)}

                print(f"Attempt {attempt + 1}: Invalid or empty response: {result}")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Request failed: {e}")

            # If we're here, the request failed or returned an invalid/empty response
            if attempt < max_retries - 1:  # don't sleep after the last attempt
                delay = min(initial_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

        print("Max retries reached. Returning empty response.")
        return {"ans": ""}

    def extract_ans(self, response_str, choices, question_type):
        """
        Extract the answer from the LLM response.
        """
        if not response_str:
            logger.warning("Empty answer received")
            return []

        # Try to parse as JSON first
        try:
            json_result = json.loads(response_str) if isinstance(response_str, str) else response_str
            if isinstance(json_result, dict) and 'ans' in json_result:
                ans = json_result['ans'].upper()
                if question_type == "single" or question_type == "单项选择题":
                    if ans in choices:
                        return [ans]
                else:
                    ans_list = [a for a in ans if a in choices]
                    if ans_list:
                        return ans_list
        except json.JSONDecodeError:
            pass

        # If JSON parsing fails, try regular expression patterns
        if question_type == "single" or question_type == "单项选择题":
            cleaned_ans = re.findall(r'\(?([A-Z])\)?', response_str.upper())
            if cleaned_ans and cleaned_ans[0] in choices:
                return [cleaned_ans[0]]
            else:
                return []
        elif question_type == "multi" or question_type == "多项选择题":
            patterns = [
                r'\(?([A-Z])\)?',  # Match single letters, optionally in parentheses
                r'([A-Z])\s*[,，、]\s*',  # Match letters followed by various separators
                r'\b([A-Z])\b'  # Match whole word letters
            ]
            
            for pattern in patterns:
                multiple_ans = re.findall(pattern, response_str.upper())
                multiple_ans = [a for a in multiple_ans if a in choices]
                if multiple_ans:
                    return multiple_ans

        return []

    def _save_results(self, results, split):
        """
        Save results in a flat directory structure.
        """
        # Save results with predictions only
        results_path = os.path.join(self.save_dir, f"results_{self.task}_{split}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} predictions to {results_path}")


def main(args):
    evaluator = APIEvaluator(args)
    evaluator.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shot", type=int, default=5, help="Number of shots (examples)")
    parser.add_argument("--api_base", type=str, help="OpenAI API base URL", 
                        default="https://api.siliconflow.cn/v1")
    parser.add_argument("--model_name", type=str, help="OpenAI model name", 
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--task_dir", type=str, help="Directory of the task data", 
                        default="evaluations/llmeval")
    parser.add_argument("--task", type=str, help="Name of the task", 
                        default="cpsyexam")
    parser.add_argument("--save_dir", type=str, help="Directory to save the results", 
                        default="results")
    parser.add_argument("--split", type=str, help="Dataset split to evaluate", 
                        default="test")
    parser.add_argument("--lang", type=str, help="Language of the task", 
                        default="zh")
    args = parser.parse_args()

    main(args)
