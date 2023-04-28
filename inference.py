import os
import random
import sys

import fire
import numpy as np
import gradio as gr
from tqdm import tqdm
import jsonlines
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, SequentialSampler
import transformers
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from prompt import NONE_TEMPLATE

from utils.callbacks import Iteratorize, Stream
from utils.parser import parse_response
from utils.prompter import Prompter
import ipdb

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def set_seed(args):
    if isinstance(args, int):
        random.seed(args)
        np.random.seed(args)
        torch.manual_seed(args)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    output_dir: str = "",
    batch_size: int = 2,
    data_path: str="",
    cutoff_len: int = 480,
    top_p=1.0,
    top_k=50,
    num_beams=1,
    device_no=0,
):
    set_seed(42)
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir='./cache')
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference

    data = load_dataset(data_path)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=None,
        )
        # if (
        #     result["input_ids"][-1] != tokenizer.eos_token_id
        #     and len(result["input_ids"]) < cutoff_len
        #     and add_eos_token
        # ):
        #     result["input_ids"].append(tokenizer.eos_token_id)
        #     result["attention_mask"].append(1)

        # result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_input_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"]
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt
    
    def collate_fn(batch):
        input_ids = torch.tensor([f["input_ids"] for f in batch], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in batch], dtype=torch.long)
        input = [f["input"] for f in batch]
        labels = [f["output"] for f in batch]
        instruction = [f["instruction"] for f in batch]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "input": input, "instruction": instruction}

    test_dataset = (
        data["test"].map(generate_and_tokenize_input_prompt)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=SequentialSampler(test_dataset)
    )
    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            cache_dir="./cache",
            device_map={'':device_no}
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'':device_no}
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        dataloader,
        top_p,
        top_k,
        num_beams,
        temperature=0.1,
        max_new_tokens=256,
        **kwargs,
    ):
        def eval_f1_score(gt_list, pred_list):
            gt_num, pred_num, correct_num = 0, 0, 0
            
            for (gt, pred) in zip(gt_list, pred_list):
                if gt == None and pred == None:
                    continue
                
                if gt != None:
                    gt = set(gt)
                    gt_num += len(gt)
                if pred != None:
                    pred = set(pred)
                    pred_num += len(pred)
                
                if gt != None and pred != None:
                    correct_num += len(gt.intersection(pred))
                
            recall = correct_num/gt_num if gt_num!=0 else 0
            precision = correct_num/pred_num if pred_num!=0 else 0
            f1 = 2*recall*precision/(recall + precision) if correct_num!=0 else 0
            return recall, precision, f1, gt_num, pred_num, correct_num

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams
        )
        gt_list = []
        pred_list = []
        for batch in tqdm(dataloader):
            # Without streaming
            with torch.no_grad():
                inputs = batch["input_ids"].to("cuda")
                mask = batch["attention_mask"].to("cuda")
            
                generation_output = model.generate(
                    input_ids=inputs,
                    attention_mask=mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens
                )

                labels = batch['labels']
            
                decoded_preds = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)

                pred_responses = [prompter.get_response(pred) for pred in decoded_preds]

                gt = [parse_response(label) if 'no entity' not in label else None for label in labels]
                pred = [parse_response(res) if 'no entity' not in res else None for res in pred_responses]
                print(gt)
                print(pred)
                
                gt_list.extend(gt)
                pred_list.extend(pred)

        recall, precision, f1, gt_num, pred_num, correct_num = eval_f1_score(gt_list, pred_list)
        metrics = {
            "test_f1": f1,  "test_recall": recall, "test_precision": precision,
            "gt_num": gt_num, "pred_num": pred_num, "correct_num": correct_num
        }

        return metrics, gt_list, pred_list

    metrics, gt_list, pred_list = evaluate(test_dataloader, top_k=top_k, top_p=top_p, num_beams=num_beams)
    with jsonlines.open(os.path.join(output_dir, "metric.jsonlines"), 'a') as writer:
        writer.write({
            'batch': metrics,
            'args': {
                "load_8bit": load_8bit,
                "base_model": base_model,
                "lora_weights": lora_weights,
                "prompt_template": prompt_template,
                "output_dir": output_dir,
                "batch_size": batch_size,
                "data_path": data_path,
                "cutoff_len": cutoff_len,
                "top_p": top_p,
                "top_k": top_k,
                "num_beams": num_beams
            }
        })
    
    show_result(test_dataset, gt_list, pred_list, os.path.join(output_dir, 'test_result.txt'))

def show_result(features, gt_list, pred_list, output_path):
    with open(output_path, 'w') as f:
        # assert(len(features)==len(gt_list))
        # assert(len(features)==len(pred_list))

        for feature, gts, preds in zip(features, gt_list, pred_list):
            f.write("--------------------------------------------------------------------\n")
            f.write("Instruction: {}\n".format(feature['instruction']))
            f.write("Input: {}\n\n".format(feature['input']))

            f.write("Gt Output: {}\n".format(' '.join(map(str, set(gts))) if gts else None))
            if preds:
                for pred in set(preds):
                    match = 'Matched' if gts and pred in gts else 'Dismatched' 
                    f.write("Entity {}: {}.\n".format(
                        match, pred)
                    )
            else:
                f.write("Entity Pred: {}.\n".format(None))

if __name__ == "__main__":
    fire.Fire(main)
