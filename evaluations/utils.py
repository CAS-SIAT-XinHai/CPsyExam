import json
import os
import re
import string
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Tuple
from typing import List

import numpy as np
from tqdm import tqdm
from transformers.utils import cached_file

from llmtuner.eval.template import register_eval_template, get_eval_template

# from llmtuner.eval.parser import get_eval_args

register_eval_template(
    name="zh",
    system='''
    ## Role
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
    作为角色 <Role>，严格遵守 <Rules>，请解答以下关于“{subject}”考试的{question_type}题。请利用您的专业知识，仔细分析每个选项，并选择最符合心理学原理和临床经验的答案。我们依赖您的专业判断，以确保选择最准确、最客观的答案。

    答案格式为“答案：{{您选择的答案}}”。\n\n
    ''',
    # system="以下是中国关于{subject}考试的{question_type}，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="\n"
)


class Evaluator:
    def __init__(self, lang, task, task_dir, save_dir):
        self.task = task
        self.task_dir = task_dir
        self.save_dir = save_dir
        self.puncs = list(string.punctuation)
        self.eval_template = get_eval_template(lang)
        self.default_question_type = 'multiple choice' if lang == 'en' else '单项选择题'

    def parse_example_with_choices(
            self,
            example: Dict[str, str],
            choices,
            with_answer=False
    ) -> Tuple[str, str]:
        candidates = [self.eval_template.choice.format(choice=ch, content=example[ch]) for ch in choices if
                      ch in example and example[ch]]
        if not with_answer:
            return "".join([example["question"]] + candidates), example['answer']
        else:
            return "".join([example["question"]] + candidates), self.eval_template.answer + example['answer']

    def format_example_with_choices(
            self,
            target_data: Dict[str, str],
            support_set: "Dataset",
            subject_name: str,
            # use_history: bool,
            choices
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example_with_choices(target_data, choices)
        history = [self.parse_example_with_choices(support_set[k], choices, with_answer=True) for k in range(len(support_set))]
        question_type = target_data.get('question_type', self.default_question_type)
        system = self.eval_template.system.format(subject=subject_name, question_type=question_type)
        return query.strip(), resp, system, history

    def eval(self) -> None:
        mapping = cached_file(
            path_or_repo_id=os.path.join(self.task_dir, self.task),
            filename="mapping.json",
            cache_dir=None,
            # token=self.model_args.hf_hub_token,
            # revision=self.model_args.model_revision
        )

        # print(mapping)

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        subjects = set(["Average"])
        for k, v in categorys.items():
            subjects.add(v['category'])

        category_corrects = {subj: np.array([], dtype="bool") for subj in subjects}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            pbar.set_postfix_str(categorys[subject]["name"])
            corrects, outputs = self.eval_subject(subject, categorys[subject]["name"])
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)

    @abstractmethod
    def eval_subject(self, subject, subject_name):
        pass

    def normalize_answer(self, s):

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(self.puncs)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self, pred, target):
        return self.normalize_answer(pred) == self.normalize_answer(target)

    @staticmethod
    def extract_ans(response_str, choices, is_single=True):
        max_option = chr(ord("A") + len(choices) - 1)
        if is_single:
            pattern = [
                rf"^选([A-{max_option}])",
                rf"^选项([A-{max_option}])",
                rf"答案是\s?选?项?\s?([A-{max_option}])",
                rf"答案为\s?选?项?\s?([A-{max_option}])",
                rf"答案应为\s?选?项?\s?([A-{max_option}])",
                rf"答案选\s?选?项?\s?([A-{max_option}])",
                rf"答案是:\s?选?项?\s?([A-{max_option}])",
                rf"答案应该是:\s?选?项?\s?([A-{max_option}])",
                rf"正确的一项是\s?([A-{max_option}])",
                rf"答案为:\s?选?项?\s?([A-{max_option}])",
                rf"答案应为:\s?选?项?\s?([A-{max_option}])",
                rf"答案:\s?选?项?\s?([A-{max_option}])",
                rf"答案是：\s?选?项?\s?([A-{max_option}])",
                rf"答案应该是：\s?选?项?\s?([A-{max_option}])",
                rf"答案为：\s?选?项?\s?([A-{max_option}])",
                rf"答案应为：\s?选?项?\s?([A-{max_option}])",
                rf"答案：\s?选?项?\s?([A-{max_option}])",
            ]
            ans_list = []
            if response_str[0] in choices:
                ans_list.append(response_str[0])
            for p in pattern:
                if len(ans_list) == 0:
                    ans_list = re.findall(p, response_str)
                else:
                    break
            return ans_list
        else:
            res = re.search(r"(答案|正确选项)(?:是|：|为|应该是|应该为)(.*?)(。|\.|$)", response_str, re.S)
            if res:
                return [x for x in res.group(2) if x in choices]
            return [i for i in response_str if i in choices]

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join([
            "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
            for category_name, category_correct in category_corrects.items() if len(category_correct)
        ])
        print(score_info)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=False)
            with open(os.path.join(self.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)
