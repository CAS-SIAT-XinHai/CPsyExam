# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import os
import string
from typing import Any, Dict, Optional, Tuple
from typing import List

import numpy as np
import tiktoken
import torch
import transformers
from datasets import load_dataset
from tqdm import trange
from transformers import GenerationConfig, HfArgumentParser

from llmtuner import ChatModel
from llmtuner.data import get_template_and_fix_tokenizer
from llmtuner.extras.misc import get_logits_processor
from llmtuner.hparams import DataArguments, EvaluationArguments, FinetuningArguments, GeneratingArguments, \
    ModelArguments
from llmtuner.model import load_model_and_tokenizer, dispatch_model
from llmtuner.model.parser import _parse_args
from utils import Evaluator


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


class LocalChatModel(ChatModel):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args, self.generating_args = get_eval_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)


class LocalEvaluator(Evaluator):

    def __init__(self, args) -> None:
        self.chat_model = LocalChatModel(args)
        self.model_args, self.data_args, self.eval_args, self.generating_args = self.chat_model.model_args, self.chat_model.data_args, self.chat_model.eval_args, self.chat_model.generating_args
        super().__init__(self.eval_args.lang, self.eval_args.task, self.eval_args.task_dir, self.eval_args.save_dir)
        self.model = self.chat_model.model
        self.tokenizer = self.chat_model.tokenizer
        self.template = self.chat_model.template

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

    def eval_subject(self, subject, subject_name):

        dataset = load_dataset(
            path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            name=subject,
            download_mode="force_redownload"
        )

        inputs, outputs, labels = [], [], []
        for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
            support_set = dataset["train"].shuffle().select(
                range(min(self.eval_args.n_shot, len(dataset["train"]))))
            choices = [c for c in string.ascii_uppercase if c in dataset[self.data_args.split].features]
            query, resp, system, history = self.format_example_with_choices(
                target_data=dataset[self.data_args.split][i],
                support_set=support_set,
                subject_name=subject_name,
                # use_history=self.template.use_history,
                choices=choices
            )

            if len(history):
                temp_q, temp_r = history.pop(0)
                history.insert(0, (system + temp_q, temp_r))
            else:
                query = system + query
            #
            if not self.template.use_history:
                query = "\n\n".join(["".join(item) for item in history] + [query])
                history = []

            query = query + self.eval_template.answer

            if i == 0:
                print(query, resp, history)

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
        # pbar.set_postfix_str(categorys[subject]["name"] + '{len(inputs)}')
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

            question_type = target_data.get('question_type', self.default_question_type)
            if question_type != self.default_question_type:
                # choices = [c for c in choices if c in response]
                ans_list = self.extract_ans(response)
                outputs[i] = ''.join(ans_list)
        corrects = (np.array(outputs) == np.array(labels))
        return corrects, outputs


def main():
    evaluator = LocalEvaluator(None)
    evaluator.eval()


if __name__ == "__main__":
    main()
