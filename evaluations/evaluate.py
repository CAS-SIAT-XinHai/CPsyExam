# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import json
import os
import string
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from typing import List

import numpy as np
import tiktoken
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers import GenerationConfig
from transformers import HfArgumentParser
from transformers.utils import cached_file

from llmtuner import ChatModel
from llmtuner.data import get_template_and_fix_tokenizer
# from llmtuner.eval.parser import get_eval_args
from llmtuner.model import load_model_and_tokenizer, dispatch_model
from llmtuner.eval.template import get_eval_template, EvalTemplate, register_eval_template
from llmtuner.extras.misc import get_logits_processor
# from llmtuner.extras.misc import dispatch_model, get_logits_processor, parse_args
# from llmtuner.extras.template import get_template_and_fix_tokenizer
from llmtuner.hparams import (
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments, GeneratingArguments
)
from llmtuner.model.parser import _parse_args


# from llmtuner.tuner.core import load_model_and_tokenizer


@dataclass
class MCQAEvalTemplate(EvalTemplate):
    default: str

    def parse_example_with_choices(
            self,
            example: Dict[str, str],
            choices
    ) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in choices if
                      ch in example and example[ch]]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example_with_choices(
            self,
            target_data: Dict[str, str],
            support_set: "Dataset",
            subject_name: str,
            use_history: bool,
            choices
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example_with_choices(target_data, choices)
        history = [self.parse_example_with_choices(support_set[k], choices) for k in range(len(support_set))]

        question_type = target_data.get('question_type', self.default)
        if len(history):
            temp = history.pop(0)
            history.insert(0, (self.system.format(subject=subject_name,
                                                  question_type=question_type) + temp[0], temp[1]))
        else:
            query = self.system.format(subject=subject_name, question_type=question_type) + query

        if not use_history:
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


def get_eval_args(
        args: Optional[Dict[str, Any]] = None
) -> Tuple[
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments,
    GeneratingArguments
]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        EvaluationArguments,
        FinetuningArguments,
        GeneratingArguments
    ))
    model_args, data_args, eval_args, finetuning_args, generating_args = _parse_args(parser, args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args, generating_args


register_eval_template(
    name="en",
    system="The following are {question_type} questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" "
)

register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的{question_type}，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="\n"
)


class MCQAEvaluator(ChatModel):

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args, self.generating_args = get_eval_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)

        template = get_eval_template(self.eval_args.lang)
        self.eval_template = MCQAEvalTemplate(
            default='multiple choice' if self.eval_args.lang == 'en' else '单项选择题',
            system=template.system,
            choice=template.choice,
            answer=template.answer,
            prefix=template.prefix
        )
        self.system_prompt = template.system

    def _encode_choices(self, choices) -> List[int]:
        if isinstance(getattr(self.tokenizer, "tokenizer", None), tiktoken.Encoding):  # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        return [self.tokenizer.encode(self.eval_template.prefix + ch, **kwargs)[-1] for ch in choices]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor], choices) -> List[str]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_inputs = self._encode_choices(choices)
        choice_probs = torch.nn.functional.softmax(word_probs[:, choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self) -> None:
        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
            revision=self.model_args.model_revision
        )
        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        subjects = set(["Average"])
        for k, v in categorys.items():
            subjects.add(v['category'])

        category_corrects = {subj: np.array([], dtype="bool") for subj in subjects}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                download_mode="force_redownload"
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
                support_set = dataset["train"].shuffle().select(
                    range(min(self.eval_args.n_shot, len(dataset["train"]))))
                choices = [c for c in string.ascii_uppercase if c in dataset[self.data_args.split].features]
                query, resp, history = self.eval_template.format_example_with_choices(
                    target_data=dataset[self.data_args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                    use_history=self.template.use_history,
                    choices=choices
                )
                input_ids, _ = self.template.encode_oneturn(
                    tokenizer=self.tokenizer, query=query, resp=resp, history=history
                )
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(resp)

            for i in trange(0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1,
                            leave=False):
                batch_input = self.tokenizer.pad(
                    inputs[i: i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input, choices)
                outputs += preds

            # replace those multiple option MCQA with generation
            pbar.set_postfix_str(categorys[subject]["name"] + '{len(inputs)}')
            for i in trange(0, len(inputs), desc="Predicting batches", position=1, leave=False):
                item = inputs[i]
                target_data = dataset[self.data_args.split][i]
                generating_args = self.generating_args.to_dict()
                generating_args.update(dict(
                    num_return_sequences=1,
                    eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
                    pad_token_id=self.tokenizer.pad_token_id
                ))

                input_ids = torch.tensor([item['input_ids']], device=self.model.device)
                gen_kwargs = dict(
                    inputs=input_ids,
                    generation_config=GenerationConfig(**generating_args),
                    logits_processor=get_logits_processor()
                )
                prompt_length = len(item['input_ids'])
                generate_output = self.model.generate(**gen_kwargs)
                response_ids = generate_output[:, prompt_length:]
                response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)[0]
                # response_length = 0
                # for i in range(len(response_ids)):
                #     eos_index = (response_ids[i] == self.tokenizer.eos_token_id).nonzero()
                #     response_length += eos_index[0].item() if len(eos_index) else len(response_ids[i])

                if target_data['question_type'] != self.eval_template.default:
                    outputs[i] = ''.join([c for c in choices if c in response])
            corrects = (np.array(outputs) == np.array(labels))
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join([
            "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
            for category_name, category_correct in category_corrects.items() if len(category_correct)
        ])
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def main():
    evaluator = MCQAEvaluator()
    evaluator.eval()


if __name__ == "__main__":
    main()
