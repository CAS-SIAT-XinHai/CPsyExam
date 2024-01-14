import json
import os
from typing import Literal, Optional

import fire
from datasets import load_dataset
from tqdm import tqdm

question_types = set()


def test(
        task: Optional[str] = "ceval",
        dataset_dir: Optional[str] = "llmeval",
        split: Optional[Literal["validation", "test"]] = "validation",
):
    with open(os.path.join(dataset_dir, task, "mapping.json"), "r", encoding="utf-8") as f:
        categories = json.load(f)

    pbar = tqdm(categories.keys(), desc="Processing subjects", position=0)
    for subject in pbar:
        pbar.set_postfix_str(categories[subject]["name"])
        dataset = load_dataset(os.path.join(dataset_dir, task), subject,
                               download_mode="force_redownload")
        for i, item in enumerate(dataset[split]):
            if i < 5:
                print(i, json.dumps(item, indent=2, ensure_ascii=False))
            question_types.update([item.get('question_type')])
    print(question_types)


if __name__ == "__main__":
    fire.Fire(test)
