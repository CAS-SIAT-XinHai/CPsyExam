#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Add LLaMA-Factory source to PYTHONPATH
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_FACTORY_PATH = os.path.join(WORK_DIR, "related_repos/LLaMA-Factory/src")
sys.path.append(LLAMA_FACTORY_PATH)
sys.path.append(os.path.join(WORK_DIR, "src"))

import argparse
import json
import re
import string
import random
import torch
import transformers
from transformers import GenerationConfig, HfArgumentParser
from tqdm import trange
from typing import Dict, List, Optional, Any, Tuple

from llmtuner import ChatModel
from llmtuner.data import get_template_and_fix_tokenizer
from llmtuner.extras.misc import get_logits_processor
from llmtuner.hparams import DataArguments, EvaluationArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from llmtuner.model import load_model_and_tokenizer, dispatch_model
from llmtuner.model.parser import _parse_args


def get_eval_args(args: Optional[Dict[str, Any]] = None) -> Tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments, GeneratingArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments, GeneratingArguments))
    model_args, data_args, eval_args, finetuning_args, generating_args = _parse_args(parser, args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    transformers.set_seed(eval_args.seed)
    return model_args, data_args, eval_args, finetuning_args, generating_args


class LocalChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args, self.generating_args = get_eval_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)


class LocalEvaluator:
    def __init__(self, args):
        self.args = vars(args) if args is not None else None
        self.chat_model = LocalChatModel(self.args)
        self.model = self.chat_model.model
        self.tokenizer = self.chat_model.tokenizer
        self.template = self.chat_model.template
        self.generating_args = self.chat_model.generating_args
        
        self.task_dir = args.task_dir
        self.task = args.task
        self.n_shot = args.n_shot
        self.save_dir = args.save_dir

        # Load the mapping table
        with open(os.path.join(self.task_dir, self.task, 'mapping.json'), 'r', encoding='utf-8') as f:
            self.task_mapping = json.load(f)

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def _encode_choices(self, choices) -> List[int]:
        """Encode choices in a consistent way"""
        kwargs = dict(add_special_tokens=False)
        return [self.tokenizer.encode(ch, **kwargs)[-1] for ch in choices]

    @torch.inference_mode()
    def get_model_response(self, messages, choices, question_type):
        """Get response from the model using logits for single choice and generation for multiple choice"""
        query = messages[-1]["content"]
        input_ids, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer,
            query=query,
            resp="答案：",
            history=[(msg["content"], next_msg["content"]) 
                    for msg, next_msg in zip(messages[:-1:2], messages[1::2]) 
                    if msg["role"] == "user" and next_msg["role"] == "assistant"]
        )
        
        model_input = self.tokenizer.pad(
            [{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}],
            return_attention_mask=True,
            return_tensors="pt"
        ).to(self.model.device)

        if question_type == "single" or question_type == "单项选择题":
            logits = self.model(**model_input).logits
            lengths = torch.sum(model_input["attention_mask"], dim=-1)
            word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
            
            # Encode choices and get probabilities
            choice_inputs = []
            for ch in choices:
                # Try different formats for choice encoding
                token_id = self.tokenizer.encode(ch, add_special_tokens=False)[-1]
                choice_inputs.append(token_id)
            
            choice_probs = torch.nn.functional.softmax(word_probs[:, choice_inputs], dim=-1).detach()
            selected_choice = choices[torch.argmax(choice_probs, dim=-1).item()]
            return {"ans": selected_choice}
        else:
            generating_args = self.generating_args.to_dict()
            generating_args.update(dict(
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
                pad_token_id=self.tokenizer.pad_token_id
            ))
            
            outputs = self.model.generate(
                **model_input,
                generation_config=GenerationConfig(**generating_args),
                logits_processor=get_logits_processor()
            )
            
            input_length = model_input['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            ans_list = self.extract_ans(response, choices, question_type)
            if ans_list:
                return {"ans": "".join(sorted(set(ans_list)))}
            
            # Fallback to logits-based selection
            logits = self.model(**model_input).logits
            lengths = torch.sum(model_input["attention_mask"], dim=-1)
            word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
            choice_inputs = self._encode_choices(choices)
            choice_probs = torch.nn.functional.softmax(word_probs[:, choice_inputs], dim=-1).detach()
            _, top_indices = torch.topk(choice_probs[0], k=2)
            selected_choices = [choices[idx] for idx in top_indices]
            return {"ans": "".join(sorted(selected_choices))}

    def eval(self):
        for split in self.task_mapping:
            dataset_name = self.task_mapping[split]["name"]
            
            # Load dataset train and test splits
            train_dir = os.path.join(self.task_dir, self.task, 'train')
            test_dir = os.path.join(self.task_dir, self.task, 'test')
            try:
                train_dataset = self.load_json(train_dir, dataset_name)
                test_dataset = self.load_json(test_dir, dataset_name)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue

            if not train_dataset or not test_dataset:
                print(f"Empty dataset for {split}, skipping")
                continue

            results = []
            for i in trange(len(test_dataset), desc=f"Evaluating {split}", position=1, leave=False):
                question_type = test_dataset[i]['question_type']
                subject_name = test_dataset[i].get('subject_name', 'Unknown Subject')
                
                typed_train_dataset = [
                    example for example in train_dataset 
                    if example['question_type'] == question_type and 'answer' in example
                ]
                
                support_set = random.sample(typed_train_dataset, min(self.n_shot, len(typed_train_dataset)))
                choices = [c for c in string.ascii_uppercase if c in test_dataset[i]["options"]]
                row = test_dataset[i]

                query, history, system = self.format_example_with_choices(
                    target_data=row,
                    support_set=support_set,
                    subject_name=subject_name,
                    choices=choices
                )

                messages = [{"role": "system", "content": system}]
                for user_msg, assistant_msg in history:
                    messages.extend([
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ])
                messages.append({"role": "user", "content": query})

                response = self.get_model_response(messages, choices, row['question_type'])
                
                if question_type == "single" or question_type == "单项选择题":
                    if isinstance(response, dict) and 'ans' in response:
                        answer = response['ans']
                        if answer in choices:
                            results.append({
                                'question_id': row['id'],
                                'question': row['question'],
                                'options': row['options'],
                                'prediction': answer,
                                'n_shot': self.n_shot
                            })
                else:
                    ans_list = self.extract_ans(response, choices, row['question_type'])
                    if ans_list:
                        results.append({
                            'question_id': row['id'],
                            'question': row['question'],
                            'options': row['options'],
                            'prediction': ''.join(sorted(ans_list)),
                            'n_shot': self.n_shot
                        })

            # Save results
            results_path = os.path.join(self.save_dir, f"results_{self.task}_{split}.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    def load_json(self, dataset_dir, dataset_name):
        """Load JSON dataset"""
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def format_example_with_choices(self, target_data, support_set, subject_name, choices):
        """Format the question and few-shot support set"""
        question_type = target_data['question_type']
        query = self.parse_example_with_choices(target_data, choices, subject_name, question_type)
        
        system = '''## Role
作为一名心理学领域的资深专家，你应具备以下特质和能力：
1. 广泛的心理学理论知识：掌握各种心理学流派的理论和实践。
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

答案格式为"答案：{{您选择的答案}}"。'''.format(
            subject=subject_name,
            question_type="单选题" if question_type == "single" else "多选题"
        )
        
        history = []
        for example in support_set:
            user_msg = self.parse_example_with_choices(example, choices, subject_name, question_type)
            if 'answer' in example:
                assistant_msg = f"答案：{example['answer']}"
                history.append((user_msg, assistant_msg))
        
        return query.strip(), history, system

    def parse_example_with_choices(self, example, choices, subject_name, question_type):
        """Create a formatted string with the question and options"""
        candidates = [f"\n{ch}. {example['options'].get(ch, '')}" for ch in choices if example['options'].get(ch)]
        question = f"科目：{subject_name}\n题型：{'单选题' if question_type == 'single' else '多选题'}\n问题：{example['question']}"
        if 'answer' in example:
            question = f"{question}{''.join(candidates)}\n答案：{example['answer']}"
        else:
            question = f"{question}{''.join(candidates)}"
        return question

    def extract_ans(self, response_str, choices, question_type):
        """Extract the answer from the model's text response"""
        if isinstance(response_str, dict) and 'ans' in response_str:
            response_str = response_str['ans']
        
        if not response_str:
            return []

        response_str = str(response_str).upper().strip()
        answer_match = re.search(r'答案[：:]\s*([A-Z,，、\s]+)', response_str)
        if answer_match:
            response_str = answer_match.group(1)
        
        if question_type == "multi" or question_type == "多项选择题":
            ans_list = [a for a in response_str if a in choices]
            if ans_list:
                return sorted(list(set(ans_list)))

            patterns = [
                r'[A-Z]',
                r'[A-Z][,，、\s]+',
                r'\(?([A-Z])\)?',
            ]
            
            for pattern in re.finditer('|'.join(patterns), response_str):
                choice = pattern.group().strip('()（）,，、 ')
                if choice in choices:
                    ans_list.append(choice)
            
            if ans_list:
                return sorted(list(set(ans_list)))

        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--template", type=str, required=True,
                        help="Template name")
    parser.add_argument("--task_dir", type=str, required=True,
                        help="Directory of the task data")
    parser.add_argument("--task", type=str, required=True,
                        help="Name of the task")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate")
    parser.add_argument("--n_shot", type=int, default=5,
                        help="Number of shots (examples)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the results")
    args = parser.parse_args()
    
    evaluator = LocalEvaluator(args)
    evaluator.eval()


if __name__ == "__main__":
    main()
