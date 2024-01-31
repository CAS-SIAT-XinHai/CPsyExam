import json
import os
import re
import string
from abc import abstractmethod
from typing import Dict, Tuple
from typing import List

import numpy as np
from tqdm import tqdm
from transformers.utils import cached_file

from llmtuner.eval.template import register_eval_template, get_eval_template

# from llmtuner.eval.parser import get_eval_args

register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
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

        print(mapping)

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

    def extract_ans(self, response_str):
        pattern = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list = []
        if response_str[0] in ["A", 'B', 'C', 'D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list) == 0:
                ans_list = re.findall(p, response_str)
            else:
                break
        return ans_list

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
