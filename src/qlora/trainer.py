# Standard Library
import logging
import os

# Third Party
import mlflow
import pandas as pd
from omegaconf import DictConfig

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset
import evaluate

# Local Folder
from ..base.base_trainer import BaseTrainer
from .model import QloraLLM
# from ilv2.utils.mlflow import log_artifacts
from .dataset import make_data_module
from .callbacks import SavePeftModelCallback
from .utils import print_trainable_parameters

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        self.setting(cfg)
        
    def setting(self, cfg):
        self.DEFAULT_PAD_TOKEN = "[PAD]"
        
        if torch.cuda.is_available():   
            torch.backends.cuda.matmul.allow_tf32 = True
        
        QLLM = QloraLLM(cfg)
        self.args, self.training_args = QLLM.args, QLLM.training_args
        self.model, self.tokenizer = QLLM.model, QLLM.tokenizer
        self.data_module = make_data_module(tokenizer=self.tokenizer, args=self.args)
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self.training_args,
            **{k: v for k, v in self.data_module.items() if k != 'predict_dataset'},
        )
            
    def load_checkpoint(self, cfg):
        """
        チェックポイントを読み込んでトレーナーを再構築するメソッド
        """
        # モデルとトークナイザーを保存されたチェックポイントからロード
        self.model = AutoModelForCausalLM.from_pretrained(cfg["trainer"]["checkpoint_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["trainer"]["checkpoint_path"])
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self.training_args,
            **{k: v for k, v in self.data_module.items() if k != 'predict_dataset'},
        )
        print(f"Checkpoint loaded from {checkpoint_path}")
        
    def save_model(self):
        save_path = os.path.join(self.args.output_dir, "final_model")
        self.trainer.save_model(save_path)
        print(f"Model saved to {save_path}")
        
    def train(self):
        # Callbacks
        if not self.args.full_finetune:
            self.trainer.add_callback(SavePeftModelCallback)
        if self.args.do_mmlu_eval:
            if self.args.mmlu_dataset == 'mmlu-zs':
                mmlu_dataset = load_dataset("json", data_files={
                    'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                    'test': 'data/mmlu/zero_shot_mmlu_test.json',
                })
                mmlu_dataset = mmlu_dataset.remove_columns('subject')
            # MMLU Five-shot (Eval/Test only)
            elif self.args.mmlu_dataset == 'mmlu' or self.args.mmlu_dataset == 'mmlu-fs':
                mmlu_dataset = load_dataset("json", data_files={
                    'eval': 'data/mmlu/five_shot_mmlu_val.json',
                    'test': 'data/mmlu/five_shot_mmlu_test.json',
                })
                # mmlu_dataset = mmlu_dataset.remove_columns('subject')
            mmlu_dataset = mmlu_dataset[self.args.mmlu_split]
            if self.args.max_mmlu_samples is not None:
                mmlu_dataset = mmlu_dataset.select(range(self.args.max_mmlu_samples))
            abcd_idx = [
                tokenizer("A", add_special_tokens=False).input_ids[0],
                tokenizer("B", add_special_tokens=False).input_ids[0],
                tokenizer("C", add_special_tokens=False).input_ids[0],
                tokenizer("D", add_special_tokens=False).input_ids[0],
            ]
            accuracy = evaluate.load("accuracy")
            class MMLUEvalCallback(transformers.TrainerCallback):
                def on_evaluate(self, args, state, control, model, **kwargs):
                    data_loader = self.trainer.get_eval_dataloader(mmlu_dataset)
                    source_max_len = self.trainer.data_collator.source_max_len
                    self.trainer.data_collator.source_max_len = args.mmlu_source_max_len
                    self.trainer.model.eval()
                    preds, refs = [], []
                    loss_mmlu = 0
                    for batch in tqdm(data_loader, total=len(data_loader)):
                        (loss, logits, labels) = self.trainer.prediction_step(self.trainer.model,batch,prediction_loss_only=False,)
                        # There are two tokens, the output, and eos token.
                        for i, logit in enumerate(logits):
                            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                            preds.append(torch.argmax(logit_abcd).item())
                        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                        refs += [abcd_idx.index(label) for label in labels.tolist()]
                        loss_mmlu += loss.item()
                    # Extract results by subject.
                    results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                    subject = mmlu_dataset['subject']
                    subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                    for s,p,r in zip(subject, preds, refs):
                        subjects[s]['preds'].append(p)
                        subjects[s]['refs'].append(r)
                    subject_scores = []
                    for subject in subjects:
                        subject_score = accuracy.compute(
                            references=subjects[subject]['refs'],
                            predictions=subjects[subject]['preds']
                        )['accuracy']
                        results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                        subject_scores.append(subject_score)
                    results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                    self.trainer.log(results)
                    self.trainer.data_collator.source_max_len = source_max_len

            self.trainer.add_callback(MMLUEvalCallback)
            
        # Verifying the datatypes and parameter counts before training.
        print_trainable_parameters(self.args, self.model)
        dtypes = {}
        for _, p in self.model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes: dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items(): total+= v
        for k, v in dtypes.items():
            print(k, v, v/total)

        self.all_metrics = {"run_name": self.args.run_name}
        
        
        # Training
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        self.all_metrics.update(metrics)
        
        self.save_model()
    
        with open(os.path.join(self.args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(self.all_metrics))
        
    def evaluate(self, cfg):
        # Evaluation
        logger.info("*** Evaluate ***")
        self.load_checkpoint(cfg)
        metrics = self.trainer.evaluate(metric_key_prefix="eval")
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        self.all_metrics.update(metrics)

    def predict(self, cfg):
        # Prediction
        logger.info("*** Predict ***")
        self.load_checkpoint(cfg)
        prediction_output = self.trainer.predict(test_dataset=self.data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(self.args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(self.data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        self.trainer.log_metrics("predict", prediction_metrics)
        self.trainer.save_metrics("predict", prediction_metrics)
        self.all_metrics.update(prediction_metrics)