# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import argparse
import os
import string
from time import sleep

import numpy as np
import openai
from datasets import load_dataset
from tqdm import trange

from utils import Evaluator


class APIEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args.lang, args.task, args.task_dir, args.save_dir)
        openai.api_key = os.getenv("API_KEY")
        openai.api_base = args.api_base
        self.split = args.split
        self.n_shot = args.n_shot
        self.task_dir = args.task_dir
        self.task = args.task
        self.model_name = args.model_name

    def eval_subject(self, subject, subject_name):

        dataset = load_dataset(
            path=os.path.join(self.task_dir, self.task),
            name=subject,
            download_mode="force_redownload"
        )

        inputs, outputs, labels = [], [], []
        for i in trange(len(dataset[self.split]), desc="Formatting batches", position=1, leave=False):
            support_set = dataset["train"].shuffle().select(
                range(min(self.n_shot, len(dataset["train"]))))
            choices = [c for c in string.ascii_uppercase if c in dataset[self.split].features]
            row = dataset[self.split][i]

            query, resp, system, history = self.format_example_with_choices(
                target_data=row,
                support_set=support_set,
                subject_name=subject_name,
                # use_history=True,
                choices=choices
            )

            print(query, resp, system, history)

            full_prompt = [
                              {"role": "system", "content": system}
                          ] + [
                              [
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": self.eval_template.answer + assistant}
                              ] for user, assistant in history
                          ] + [
                              {"role": "user", "content": query},
                          ]

            print(full_prompt)

            response = None
            timeout_counter = 0
            while response is None and timeout_counter <= 30:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter += 1
                    print(msg)
                    sleep(5)
                    continue

            if response == None:
                response_str = ""
            else:
                response_str = response['choices'][0]['message']['content']

            ans_list = self.extract_ans(response_str)
            outputs.append(''.join(ans_list))
            labels.append(resp)
        corrects = (np.array(outputs) == np.array(labels))
        return corrects, outputs


def main(args):
    evaluator = APIEvaluator(args)
    evaluator.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shot", "-k", type=int, default=5)
    parser.add_argument("--api_base", type=str, default="xxx")
    # parser.add_argument("--minimax_group_id", type=str, default="xxx")
    # parser.add_argument("--minimax_key", type=str, default="xxx")
    # parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task_dir", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--lang", type=str, default='zh')
    parser.add_argument("--cot", action="store_true")
    # parser.add_argument("--subject", "-s", type=str, default="operating_system")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()
    main(args)
