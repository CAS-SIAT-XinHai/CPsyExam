import argparse
import json
import os
from glob import glob
from typing import Dict

import numpy as np
import pandas as pda
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file


def gold_labels(opts):
    mapping = cached_file(
        path_or_repo_id=os.path.join(opts.task_dir, opts.task),
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
        dataset = load_dataset(
            path=os.path.join(opts.task_dir, opts.task),
            name=subject,
            download_mode="force_redownload"
        )

        ans = []
        for i in trange(len(dataset[opts.split]), desc="Formatting batches", position=1, leave=False):
            row = dataset[opts.split][i]
            ans.append(row['answer'])

        # corrects, outputs = self.eval_subject(subject, categorys[subject]["name"])
        # category_name = categorys[subject]["category"]
        # category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
        # category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
        results[subject] = {str(i): ans[i] for i in range(len(ans))}
    return results


def main(opts):
    answer_file = os.path.join(opts.results_dir, f"{opts.split}_answer.json")
    if not os.path.isfile(answer_file):
        ans = gold_labels(opts)
        with open(answer_file, "w", encoding="utf-8") as f:
            json.dump(ans, f, ensure_ascii=False, indent=4)
    else:
        with open(answer_file) as f:
            ans = json.load(f)

    zero_shot_results = {}
    for f in glob(os.path.join(opts.results_dir, 'zero-shot', "*.json")):
        with open(f) as fd:
            zero_shot_results[os.path.basename(f)] = json.load(fd)

    few_shot_results = {}
    for f in glob(os.path.join(opts.results_dir, 'few-shot', "*.json")):
        with open(f) as fd:
            few_shot_results[os.path.basename(f)] = json.load(fd)

    data = []
    for k, v in zero_shot_results.items():
        print(k)
        for s in ans:
            if len(ans[s]) > 0:
                ks = s.split("-")
                if len(ks) == 3:
                    part, subject, t = ks
                    exam = ""
                else:
                    part, exam, subject, t = ks

                data.append({
                    "Model": k,
                    "Shot": 0,
                    "Exam": exam,
                    "Part": part,
                    "Subject": subject,
                    "Type": t,
                    "Correct": sum([int(v[s][q] == a) for q, a in ans[s].items()]),
                    "Total": len(ans[s])
                })

    for k, v in few_shot_results.items():
        print(k)
        for s in ans:
            if len(ans[s]) > 0:
                ks = s.split("-")
                if len(ks) == 3:
                    part, subject, t = ks
                    exam = ""
                else:
                    part, exam, subject, t = ks

                data.append({
                    "Model": k,
                    "Shot": 5,
                    "Exam": exam,
                    "Part": part,
                    "Subject": subject,
                    "Type": t,
                    "Correct": sum([int(v[s][q] == a) for q, a in ans[s].items()]),
                    "Total": len(ans[s])
                })

    df = pda.DataFrame.from_dict(data)
    print(df)
    df.to_csv("detail_results.csv", index=False)

    df_exam = df.loc[(df.Shot == 0) & (df.Type == '单项选择题') & (df.Part == 'KG')][['Correct', 'Total']].groupby(
        [df.Model, df.Exam]).sum().reset_index()
    print(df_exam)
    df_exam['Score'] = df_exam['Correct'] / df_exam['Total']

    exams = ['GEE', 'PCE', 'SSE', 'TQE']
    for e in exams:
        e_df = df_exam.loc[(df_exam.Exam == e)][['Model', 'Score']]
        print(e)
        print("\t".join(e_df.Model.tolist()))
        print("\t".join([str(c) for c in e_df.Score.tolist()]))

    df_ca = df.loc[(df.Shot == 0) & (df.Type == '单项选择题') & (df.Part == 'CA')][['Correct', 'Total']].groupby(
        [df.Model, df.Subject]).sum().reset_index()
    print(df_ca)
    df_ca['Score'] = df_ca['Correct'] / df_ca['Total']

    cas = {
        "心理咨询": 'Treatment',
        "心理理论": 'Method',
        "心理诊断": 'Diagnosis'
    }
    for e in cas:
        e_df = df_ca.loc[(df_ca.Subject == e)][['Model', 'Score']]
        print(e)
        print("\t".join(e_df.Model.tolist()))
        print("\t".join([str(c) for c in e_df.Score.tolist()]))

    df_subject = df.loc[(df.Shot == 5) & (df.Part == 'KG') & (df.Model.isin(["gpt-4.json", "chatglm_turbo.json"]))][
        ['Correct', 'Total']].groupby([df.Model, df.Subject]).sum().reset_index()
    df_subject['Score'] = df_subject['Correct'] / df_subject['Total']
    print(df_subject)
    df_subject.to_csv("subject_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str)
    parser.add_argument("--task", type=str, default='cpsyexam')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--results_dir", type=str)
    args = parser.parse_args()

    main(args)
