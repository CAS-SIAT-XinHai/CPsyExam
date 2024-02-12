import argparse
import json
import logging
import os
import string
from itertools import chain

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from cpsyexam.utils import Reader


class MCQASFTReader(Reader):

    def parse_dataset(self, opts):
        with open(os.path.join(opts.task_dir, opts.task, "mapping.json"), "r", encoding="utf-8") as f:
            categories = json.load(f)
        subjects = set(["Average"])
        for k, v in categories.items():
            subjects.add(v['category'])

        category_corrects = {subj: np.array([], dtype="bool") for subj in subjects}
        pbar = tqdm(categories.keys(), desc="Processing subjects", position=0)
        for subject in pbar:
            pbar.set_postfix_str(categories[subject]["name"])
            dataset = load_dataset(os.path.join(opts.task_dir, opts.task), subject,
                                   download_mode="force_redownload")
            choices = [c for c in string.ascii_uppercase if c in dataset[opts.split].features]
            category_name = categories[subject]["category"]
            for i, item in enumerate(dataset[opts.split]):
                if i < 5:
                    print(i, json.dumps(item, indent=2, ensure_ascii=False))
                item['question_type'] = item.get('question_type', self.default_question_type)
                yield item, choices, category_name

    def convert(self, opts):

        # Convert grouped DataFrame to a list of JSON entries
        for row, choices, subject_name in self.parse_dataset(opts):
            query, resp, system, history = self.format_example_with_choices(
                target_data=row,
                support_set=[],
                subject_name=row.get('subject_name', subject_name),
                # use_history=True,
                choices=choices
            )

            sft_entry = {
                "system": system,  # content from cMedQA_Q.csv
                "instruction": query,  # content from cMedQA_Q.csv
                "input": "",
                "output": self.eval_template.answer + resp,  # list of content from cMedQA_A.csv
            }
            yield sft_entry


class QASFTReader(Reader):

    def parse_dataset(self, opts):
        with open(opts.qa_file, "r", encoding="utf-8") as f:
            for i, item in enumerate(json.load(f)):
                if i < 5:
                    print(i, json.dumps(item, indent=2, ensure_ascii=False))
                yield item

    def convert(self, opts):
        system = '''
        ## Role
        作为一名心理学领域的资深专家，你应具备以下特质和能力：
        1. 广泛的心理学理论知识：掌握各种心理学流派的理论和实践。
        2. 深刻的人类行为理解：能够解读复杂的行为模式和心理过程。
        3. 分析和判断能力：基于案例细节，快速准确地进行心理分析和诊断。
        4. 临床经验：具有丰富的临床实践经验，能够处理各种心理问题和状况。
        5. 伦理观念：遵循心理学专业的伦理准则，确保患者的隐私和福祉。

        ## Rules
        1. 你是一位经验丰富的心理学专家。
        2. 你的任务是根据提供的信息，使用你的专业知识和分析能力来解答{subject}考试中的问答题。
        3. 题目将涉及心理学的各个方面，你需要利用你的专业知识来选择正确答案。
        4. 如果题目信息不足以做出判断，你需要根据你的专业经验，给出最合理的答案。

        ## Initialization
        作为角色 <Role>，严格遵守 <Rules>，请解答以下关于“{subject}”考试的问答题。请利用您的专业知识，基于心理学原理和临床经验，给出您的专业判断和答案。\n\n
        '''

        # Convert grouped DataFrame to a list of JSON entries
        for row in self.parse_dataset(opts):
            sft_entry = {
                "system": system.format(subject=row['subject_name']),  # content from cMedQA_Q.csv
                "instruction": row['question'],  # content from cMedQA_Q.csv
                "input": "",
                "output": self.eval_template.answer + row['answer'],  # list of content from cMedQA_A.csv
            }
            yield sft_entry


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='CPsyExam SFT', description='')
    parser.add_argument("--task_dir", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--qa_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--output_file", type=str, default="cpsyexam.json")
    # Initialize arguments
    args = parser.parse_args()

    sft_entries = []
    mcqa_reader = MCQASFTReader('zh', args.task, args.task_dir, args.save_dir)
    qa_reader = QASFTReader('zh', args.task, args.task_dir, args.save_dir)
    for item in chain(mcqa_reader.convert(args), qa_reader.convert(args)):
        sft_entries.append(item)

    # Save JSON data to output file
    with open(os.path.join(args.save_dir, args.output_file), 'w', encoding='utf-8') as json_file:
        json.dump(sft_entries, json_file, ensure_ascii=False, indent=2)
