# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os

import datasets

_CITATION = """

"""
_DESCRIPTION = """
A dataset for LLM evaluation on psychology using Chinese psychology examinations
"""

_HOMEPAGE = "https://cpsyexam.github.io"

_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"

_URL = "cpsyexam.zip"

task_list = [
    'CA-心理咨询-单项选择题','CA-心理咨询-多项选择题','CA-心理理论-单项选择题','CA-心理理论-多项选择题',
    'CA-心理诊断-单项选择题','CA-心理诊断-多项选择题','KG-GEE-临床与咨询心理学-单项选择题','KG-GEE-临床与咨询心理学-多项选择题',
    'KG-GEE-人格心理学-单项选择题','KG-GEE-人格心理学-多项选择题','KG-GEE-发展心理学-单项选择题','KG-GEE-发展心理学-多项选择题',
    'KG-GEE-变态心理学-单项选择题','KG-GEE-变态心理学-多项选择题','KG-GEE-实验心理学-单项选择题','KG-GEE-实验心理学-多项选择题',
    'KG-GEE-心理统计与测量-单项选择题','KG-GEE-心理统计与测量-多项选择题','KG-GEE-教育心理学-单项选择题','KG-GEE-教育心理学-多项选择题',
    'KG-GEE-普通心理学-单项选择题','KG-GEE-普通心理学-多项选择题','KG-GEE-社会心理学-单项选择题','KG-GEE-社会心理学-多项选择题',
    'KG-GEE-管理心理学-单项选择题','KG-GEE-管理心理学-多项选择题','KG-PCE-心理咨询师一级-单项选择题','KG-PCE-心理咨询师一级-多项选择题',
    'KG-PCE-心理咨询师三级-单项选择题','KG-PCE-心理咨询师三级-多项选择题','KG-PCE-心理咨询师二级-单项选择题','KG-PCE-心理咨询师二级-多项选择题',
    'KG-SSE-农学类农业推广心理基础-单项选择题','KG-SSE-农学类农业推广心理基础-多项选择题','KG-SSE-医学类人际关系心理学-单项选择题',
    'KG-SSE-医学类人际关系心理学-多项选择题','KG-SSE-医学类公关心理学-单项选择题','KG-SSE-医学类公关心理学-多项选择题','KG-SSE-医学类医学心理学-单项选择题',
    'KG-SSE-医学类医学心理学-多项选择题','KG-SSE-医学类学前教育心理学-单项选择题','KG-SSE-医学类学前教育心理学-多项选择题','KG-SSE-医学类学校心理学-单项选择题',
    'KG-SSE-医学类学校心理学-多项选择题','KG-SSE-医学类心理卫生-单项选择题','KG-SSE-医学类心理卫生-多项选择题','KG-SSE-医学类心理卫生与心理辅导-单项选择题',
    'KG-SSE-医学类心理卫生与心理辅导-多项选择题','KG-SSE-医学类心理咨询与辅导-单项选择题','KG-SSE-医学类心理咨询与辅导-多项选择题','KG-SSE-医学类心理学-单项选择题',
    'KG-SSE-医学类心理学-多项选择题','KG-SSE-医学类心理学导论-单项选择题','KG-SSE-医学类心理学导论-多项选择题','KG-SSE-医学类认知心理-单项选择题',
    'KG-SSE-医学类认知心理-多项选择题','KG-SSE-工学类旅游心理学-单项选择题','KG-SSE-工学类旅游心理学-多项选择题','KG-SSE-教育类发展与教育心理学-单项选择题',
    'KG-SSE-教育类发展与教育心理学-多项选择题','KG-SSE-教育类发展心理学-单项选择题','KG-SSE-教育类发展心理学-多项选择题','KG-SSE-教育类实验心理学-单项选择题',
    'KG-SSE-教育类实验心理学-多项选择题','KG-SSE-教育类教育心理学-单项选择题','KG-SSE-教育类教育心理学-多项选择题','KG-SSE-教育类生理心理学-单项选择题',
    'KG-SSE-教育类生理心理学-多项选择题','KG-SSE-文学类广告心理学-单项选择题','KG-SSE-文学类广告心理学-多项选择题','KG-SSE-文学类新闻心理学-单项选择题',
    'KG-SSE-文学类新闻心理学-多项选择题','KG-SSE-管理类社会心理学(二)-单项选择题','KG-SSE-管理类社会心理学(二)-多项选择题','KG-SSE-管理类管理心理学-单项选择题',
    'KG-SSE-管理类管理心理学-多项选择题','KG-SSE-经济类消费心理学-单项选择题','KG-SSE-经济类消费心理学-多项选择题','KG-TQE-中学教师心理学-单项选择题',
    'KG-TQE-中学教师心理学-多项选择题','KG-TQE-小学教师心理学-单项选择题','KG-TQE-小学教师心理学-多项选择题','KG-TQE-高等学校教师心理学-单项选择题',
    'KG-TQE-高等学校教师心理学-多项选择题'

]

question_type_mapping = {
    'single': '单项选择题',
    'multi': '多项选择题',
}


class CPsyExamConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CPsyExamConfig, self).__init__(**kwargs)


class CPsyExam(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CPsyExamConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "subject_name": datasets.Value("string"),
                "question_type": datasets.Value("string"),
                "kind": datasets.Value("string"),
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "E": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "explanation": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "test", f"{task_name}.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "dev", f"{task_name}.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "train", f"{task_name}.json"
                    ),
                },
            )
        ]

    def _generate_examples(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, encoding="utf-8") as f:
                for key, instance in enumerate(json.load(f)):
                    options = instance.pop("options", {})
                    instance['question_type'] = question_type_mapping[instance['question_type']]
                    instance.update(options)
                    yield key, instance
