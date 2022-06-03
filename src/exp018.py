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

import datasets
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
    model_name:str = "microsoft/mdeberta-v3-base"

cfg = Config()


def preprocess_function(examples, tokenizer, max_length:int):
    result = tokenizer(examples["text"], padding=True, max_length=max_length, truncation=True)
    if "isFake" in examples:
        result["label"] = examples["isFake"]
    return result


def compute_metrics(p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def load_pseudo_label(path:str, test_df:pd.DataFrame) -> pd.DataFrame:
    with open(path, "rb") as f:
        test_labels = pickle.load(f)
    test_df["label"] = test_labels
    return test_df

def main():
    set_seed(42)

    output_dir = Path("../output")
    exp_name = Path(__file__).stem

    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    sub_df = pd.read_csv("../input/sample_submission.csv")

    # test_df = load_pseudo_label("../output/emsamble_preds_005_009.pkl", test_df)

    train_df["text"] = train_df["text"].map(lambda x: x.replace(" ", "ω"))
    test_df["text"] = test_df["text"].map(lambda x: x.replace(" ", "ω"))

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

        # trn_ds = datasets.concatenate_datasets([trn_ds, test_ds])
        # trn_ds = trn_ds.shuffle(seed=42)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            overwrite_output_dir="True",
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=4,
            warmup_ratio=0.1,
            save_total_limit=1,
            logging_steps=100,
            eval_steps=100,
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            gradient_checkpointing=True,
            eval_delay=500,
        )

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