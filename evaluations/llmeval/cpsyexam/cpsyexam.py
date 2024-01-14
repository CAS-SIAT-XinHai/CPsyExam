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
    'Psychology_for_Primary_School_Teachers-knowledge-single', 'Psychology_for_Primary_School_Teachers-knowledge-multi', 'Psychology_for_Primary_School_Teachers-case-single',
    'Psychology_for_Primary_School_Teachers-case-multi', 'Psychology_for_Middle_School_Teachers-knowledge-single', 'Psychology_for_Middle_School_Teachers-knowledge-multi',
    'Psychology_for_Middle_School_Teachers-case-single', 'Psychology_for_Middle_School_Teachers-case-multi', 'GEE-case-multi',
    'GEE-case-single', 'GEE-knowledge-multi',
    'GEE-knowledge-single', 'Psychological_consultant_level_1-knowledge-multi',
    'Psychological_consultant_level_1-knowledge-single', 'Psychology for Higher Education Teachers-knowledge-multi', 'Psychology for Higher Education Teachers-knowledge-single',
    'Psychological_consultant_level_2-case-multi', 'Psychological_consultant_level_2-case-single', 'Psychological_consultant_level_2-knowledge-multi',
    'Psychological_consultant_level_2-knowledge--single', 'self_taught_examination-case-multi', 'self_taught_examination-case-single',
    'self_taught_examination-knowledge-multi', 'self_taught_examination-knowledge-single', 'Psychological_consultant_level_3-case-multi',
    'Psychological_consultant_level_3-case-single', 'Psychological_consultant_level_3-knowledge-multi', 'Psychological_consultant_level_3-knowledge-single'

]


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
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "question_type": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "E": datasets.Value("string")

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
                        data_dir, "validation", f"{task_name}.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "dev", f"{task_name}.json"
                    ),
                },
            )
        ]

    def _generate_examples(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, encoding="utf-8") as f:
                for key, instance in enumerate(json.load(f)):
                    yield key, instance
