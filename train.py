import os
import warnings
import logging

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore")

logging.getLogger(
    "huggingface_hub"
).setLevel(logging.ERROR)


import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from config import *

# 데이터셋 가져오기 =====================================================
def load_kptimes_dataset():

    base_url = (
        "https://huggingface.co/datasets/"
        "midas/kptimes/resolve/main/"
    )

    data_files = {
        "train": base_url + "train.jsonl",
        "validation": base_url + "valid.jsonl",
        "test": base_url + "test.jsonl"
    }

    return load_dataset(
        "json",
        data_files=data_files
    )

# 데이터 샘플링 =====================================================
def sample_dataset(dataset):

    train_df = (
        dataset["train"]
        .shuffle(seed=SEED)
        .select(range(TRAIN_SIZE))
        .to_pandas()
    )

    val_df = (
        dataset["validation"]
        .shuffle(seed=SEED)
        .select(range(VAL_SIZE))
        .to_pandas()
    )

    test_df = (
        dataset["test"]
        .shuffle(seed=SEED)
        .select(range(TEST_SIZE))
        .to_pandas()
    )

    return train_df, val_df, test_df

# 데이터 전처리 =====================================================

def preprocess_dataframe(df):

    df["doc_str"] = df["document"].apply(
        lambda x: " ".join(x)
    )

    df.drop_duplicates(
        subset=["doc_str"],
        inplace=True
    )

    df.dropna(inplace=True)

    df.reset_index(
        drop=True,
        inplace=True
    )

    df.drop(
        columns=["doc_str"],
        inplace=True
    )

    return df

# 토큰 분리 및 라벨 정렬 =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CKPT
)

def tokenize_and_align_labels(batch):

    tokenized_inputs = tokenizer(
        batch["document"],
        truncation=True,
        max_length=512,
        is_split_into_words=True
    )

    labels = []

    for i, label in enumerate(batch["doc_bio_tags"]):

        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:

                label_ids.append(-100)

            elif word_idx != previous_word_idx:

                label_ids.append(
                    label2id[label[word_idx]]
                )

            else:

                current_label = label[word_idx]

                if current_label == "B":
                    current_label = "I"

                label_ids.append(
                    label2id[current_label]
                )

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# 성능 평가 지수 정의 =====================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        pred_seq = []
        label_seq = []

        for p, l in zip(
            prediction,
            label
        ):

            if l != -100:

                pred_seq.append(
                    id2label[p]
                )

                label_seq.append(
                    id2label[l]
                )

        true_predictions.append(pred_seq)
        true_labels.append(label_seq)

    return {
        "precision": precision_score(
            true_labels,
            true_predictions
        ),
        "recall": recall_score(
            true_labels,
            true_predictions
        ),
        "f1": f1_score(
            true_labels,
            true_predictions
        )
    }

# Main =====================================================

def main():

    print(f"Device : {DEVICE}")

    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = load_kptimes_dataset()
    train_df, val_df, test_df = sample_dataset(
        dataset
    )
    train_df = preprocess_dataframe(
        train_df
    )
    val_df = preprocess_dataframe(
        val_df
    )
    test_df = preprocess_dataframe(
        test_df
    )

    dataset = DatasetDict({
        "train":
        Dataset.from_pandas(train_df),
        "validation":
        Dataset.from_pandas(val_df),
        "test":
        Dataset.from_pandas(test_df)
    })

    encoded = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset[
            "train"
        ].column_names
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_CKPT,
            num_labels=len(
                LABEL_LIST
            ),
            id2label=id2label,
            label2id=label2id
        )
        .to(DEVICE)
    )

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer
        )
    )

    training_args = TrainingArguments(
        output_dir=str(
            MODEL_DIR
        ),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\n===== TRAIN START =====\n")
    trainer.train()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    history = pd.DataFrame(
        trainer.state.log_history
    )

    epoch_history = history[
        history["eval_f1"].notna()
    ][[
        "epoch",
        "eval_loss",
        "eval_precision",
        "eval_recall",
        "eval_f1"
    ]]

    epoch_file = (
        OUTPUT_DIR
        / f"epoch_metrics_{timestamp}.txt"
    )

    with open(
        epoch_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 60 + "\n"
        )
        f.write(
            "EPOCH METRICS\n"
        )
        f.write(
            "=" * 60 + "\n\n"
        )
        for _, row in (
            epoch_history.iterrows()
        ):
            f.write(
                f"Epoch "
                f"{int(row['epoch'])}\n"
            )
            f.write(
                f"Loss      : "
                f"{row['eval_loss']:.4f}\n"
            )
            f.write(
                f"Precision : "
                f"{row['eval_precision']:.4f}\n"
            )
            f.write(
                f"Recall    : "
                f"{row['eval_recall']:.4f}\n"
            )
            f.write(
                f"F1 Score  : "
                f"{row['eval_f1']:.4f}\n"
            )
            f.write(
                "-" * 60 + "\n"
            )

        best_row = epoch_history.loc[
            epoch_history[
                "eval_f1"
            ].idxmax()
        ]
        f.write("\n")
        f.write(
            "=" * 60 + "\n"
        )
        f.write(
            "BEST EPOCH\n"
        )
        f.write(
            "=" * 60 + "\n"
        )
        f.write(
            f"Epoch     : "
            f"{int(best_row['epoch'])}\n"
        )
        f.write(
            f"F1 Score  : "
            f"{best_row['eval_f1']:.4f}\n"
        )

    print(
        f"Epoch metrics saved -> "
        f"{epoch_file}"
    )

    print(
        "\n===== TEST START =====\n"
    )

    start = time.time()
    result = trainer.predict(
        encoded["test"]
    )

    end = time.time()
    print(
        result.metrics
    )

    print(
        f"\nInference Time : "
        f"{end - start:.4f}s"
    )

    test_file = (
        OUTPUT_DIR
        / f"test_metrics_{timestamp}.txt"
    )

    with open(
        test_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "=" * 60 + "\n"
        )
        f.write(
            "FINAL TEST RESULT\n"
        )
        f.write(
            "=" * 60 + "\n\n"
        )
        f.write(
            f"Loss      : "
            f"{result.metrics['test_loss']:.4f}\n"
        )
        f.write(
            f"Precision : "
            f"{result.metrics['test_precision']:.4f}\n"
        )
        f.write(
            f"Recall    : "
            f"{result.metrics['test_recall']:.4f}\n"
        )
        f.write(
            f"F1 Score  : "
            f"{result.metrics['test_f1']:.4f}\n"
        )
        f.write(
            f"Runtime   : "
            f"{result.metrics['test_runtime']:.2f} sec\n"
        )

    print(
        f"Test metrics saved -> "
        f"{test_file}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )

    print(
        f"\nSaved Model -> "
        f"{MODEL_DIR}"
    )