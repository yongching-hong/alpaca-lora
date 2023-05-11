import os
import torch
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
from utils.parser import parse_response
from tqdm import tqdm

from utils.prompter import Prompter
from utils.spotasoc_predict_parser import SpotAsocPredictParser

class AlpacaTrainer(Trainer):
    def __init__(self, prompt_template_name=None, output_dir='', test_dataset=None, SSI=False, causal=True, **kwargs):
        super().__init__(compute_metrics=self.compute_metrics, **kwargs)
        self.prompter = Prompter(prompt_template_name)
        self.output_dir = output_dir
        self.test_dataset = test_dataset
        self.best_f1 = 0
        self.SSI = SSI
        self.causal = causal

    def postprocess_SSI_text(self, x_str):
        if not x_str:
            return
        
        # Clean `bos` `eos` `pad` for cleaned text
        to_remove_token_list = list()
        # if tokenizer.bos_token:
        #     to_remove_token_list += [tokenizer.bos_token]
        if self.tokenizer.eos_token:
            to_remove_token_list += [self.tokenizer.eos_token]
        if self.tokenizer.pad_token:
            to_remove_token_list += [self.tokenizer.pad_token]
        
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()

    def compute_metrics(self, eval_output: EvalPrediction):
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
        
        def show_result(metrics, features, gt_list, pred_list, output_path):
            
            with open(output_path, 'w') as f:
                # assert(len(features)==len(gt_list))
                # assert(len(features)==len(pred_list))
                f.write("Eval metrics: {} \n".format(metrics))
                f.write("*******************************************************\n")

                for feature, gts, preds in zip(features, gt_list, pred_list):
                    f.write("--------------------------------------------------------------------\n")
                    f.write("Insturction: {}\n".format(self.tokenizer.decode(feature['input_ids'])))
                    # f.write("Input: {}\n\n".format(feature['input']))

                    f.write("Gt Output: {}\n".format(' '.join(map(str, set(gts))) if gts else None))
                    if preds:
                        for pred in set(preds):
                            match = 'Matched' if gts and pred in gts else 'Dismatched' 
                            f.write("Entity {}: {}.\n".format(
                                match, pred)
                            )
                    else:
                        f.write("Entity Pred: {}.\n".format(None))

        def evaluate_output(output):
            labels = torch.tensor(output.label_ids)
            labels[labels == -100] = self.tokenizer.pad_token_id
            predictions = torch.tensor(output.predictions[0])
            predictions[predictions == -100] = self.tokenizer.pad_token_id

            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=not self.SSI)
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=not self.SSI)

            
            if self.causal:
                decoded_preds = [self.prompter.get_response(pred) for pred in decoded_preds]
                decoded_labels = [self.prompter.get_response(label) for label in decoded_labels]

            if self.SSI:
                decoded_preds = [self.postprocess_SSI_text(x) or '' for x in decoded_preds]
                decoded_labels = [self.postprocess_SSI_text(x) or '' for x in decoded_labels]

                predict_parser = SpotAsocPredictParser()
                well_formed_list, counter = predict_parser.decode(
                    gold_list=decoded_labels, pred_list=decoded_preds
                )
                gt_list = [x['gold_spot'] for x in well_formed_list]
                pred_list = [x['pred_spot'] for x in well_formed_list]
            else:
                gt_list = [parse_response(res) for res in decoded_labels]
                pred_list = [parse_response(res) for res in decoded_preds]
            
            recall, precision, f1, gt_num, pred_num, correct_num = eval_f1_score(gt_list, pred_list)
            
            metrics = {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "gt_num": gt_num,
                "pred_num": pred_num,
                "correct_num": correct_num
            }
            return metrics, gt_list, pred_list
        
        metrics, eval_gt_list, eval_pred_list = evaluate_output(eval_output)
        
        if not self.best_f1 or metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            show_result(metrics, self.eval_dataset, eval_gt_list, eval_pred_list, os.path.join(self.output_dir, 'dev_result.txt'))

            # if self.test_dataset:
            #     test_output = self.predict(self.test_dataset)
            #     test_metrics, test_gt_list, test_pred_list = evaluate_output(test_output)

            #     for key in list(test_metrics.keys()):
            #         if not key.startswith(f"test_"):
            #             test_metrics[f"test_{key}"] = test_metrics.pop(key)

            #     show_result(test_metrics, self.test_dataset, test_gt_list, test_pred_list, os.path.join(self.output_dir, 'test_result.txt'))

            #     metrics.update(test_metrics)
        
        return metrics
