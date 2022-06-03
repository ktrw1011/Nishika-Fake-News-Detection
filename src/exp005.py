import os
import pickle
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction,
    RobertaPreTrainedModel, RobertaModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

@dataclass
class Config:
    max_length:int = 512
    model_name:str = "xlm-roberta-large"

cfg = Config()


class MultiSampleDropoutHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        logits = sum([self.fc(dropout(x)) for dropout in self.dropouts])/5
        return logits

class CLSConcatHead(nn.Module):
    pass

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.config.output_hidden_states= True

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.regressor = nn.Linear(self.config.hidden_size*4, self.config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = torch.cat([outputs.hidden_states[-1*i][:,0] for i in range(1, 4+1)], dim=1)
        logits = self.regressor(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def preprocess_function(examples, tokenizer, max_length:int):
    result = tokenizer(examples["text"], padding=True, max_length=max_length, truncation=True)
    if "isFake" in examples:
        result["label"] = examples["isFake"]
    return result


def compute_metrics(p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def main():
    set_seed(42)

    output_dir = Path("../output")
    exp_name = Path(__file__).stem

    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    sub_df = pd.read_csv("../input/sample_submission.csv")

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    with open("../input/fold.pkl", "rb") as f:
        cv = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    preprocess_function_func = partial(preprocess_function, tokenizer=tokenizer, max_length=cfg.max_length)

    train_ds = train_ds.map(
        preprocess_function_func,
        batched=True,
        desc="Running tokenizer on dataset"
    )

    test_ds = test_ds.map(
        preprocess_function_func,
        batched=True,
        desc="Running tokenizer on dataset",
    )

    fold_dirs = []
    
    oof_val_preds = np.zeros((len(train_ds), 2), dtype=np.float32)
    oof_test_preds = np.zeros((5, len(test_ds), 2), dtype=np.float32)

    for fold_idx, (trn_idx, val_idx) in enumerate(cv):
        print(f"=== fold {fold_idx} start ===")
        fold_name = f"fold_{fold_idx}"
        fold_dir = output_dir / exp_name / fold_name

        fold_dirs.append(fold_dir)

        trn_ds = train_ds.select(trn_idx)
        val_ds = train_ds.select(val_idx)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            overwrite_output_dir="True",
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            # weight_decay=0.01,
            learning_rate=1e-5,
            num_train_epochs=5,
            warmup_ratio=0.1,
            save_total_limit=1,
            fp16=True,
            logging_steps=100,
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            gradient_checkpointing=True,
        )

        # model = RobertaForSequenceClassification.from_pretrained(cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=trn_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        train_result = trainer.train()
        
        metrics = train_result.metrics
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        val_preds = trainer.predict(val_ds).predictions
        oof_val_preds[val_idx] = val_preds
        val_labels = np.argmax(val_preds, axis=1)

        print(f"fold {fold_idx} acc: {accuracy_score(train_df['isFake'].values[val_idx], val_labels)}")

        test_preds = trainer.predict(test_ds).predictions
        oof_test_preds[fold_idx] = test_preds

    oof_test_preds = np.mean(oof_test_preds, axis=0)
    oof_test_labels = np.argmax(oof_test_preds, axis=1)

    with open(f"../output/{exp_name}/oof_val_preds.pkl", "wb") as f:
        pickle.dump(oof_val_preds, f)

    with open(f"../output/{exp_name}/oof_test_preds.pkl", "wb") as f:
        pickle.dump(oof_test_preds, f)

    print(f"overall acc: {accuracy_score(train_df['isFake'].values, np.argmax(oof_val_preds, axis=1))})")

    sub_df["isFake"] = oof_test_labels
    sub_df.to_csv(f"../output/{exp_name}/sub_{exp_name}.csv", index=False)

if __name__ == "__main__":
    main()