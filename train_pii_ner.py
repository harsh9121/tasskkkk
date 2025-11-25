import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from evaluate import load as load_metric

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import classification_report

# --------- LABELS ---------
ENTITY_TYPES = [
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
    "CITY",
    "LOCATION",
]

LABEL_LIST = ["O"]
for ent in ENTITY_TYPES:
    LABEL_LIST.append(f"B-{ent}")
    LABEL_LIST.append(f"I-{ent}")

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# --------- LOAD JSONL ---------
def read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items

# --------- CHAR-LEVEL LABELS ---------
def char_labels_for_example(text, entities):
    labels = ["O"] * len(text)
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        lab = ent["label"]
        for i in range(start, end):
            if 0 <= i < len(labels):
                labels[i] = lab
    return labels

def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    texts = examples["text"]
    all_entities = examples["entities"]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    all_labels = []
    for i, text in enumerate(texts):
        entities = all_entities[i]
        char_labels = char_labels_for_example(text, entities)
        offsets = tokenized["offset_mapping"][i]
        labels = []
        prev_ent_type = None

        for (start, end) in offsets:
            if start == end:
                labels.append(-100)
                continue
            span_labels = char_labels[start:end]
            ent_types = [l for l in set(span_labels) if l != "O"]
            if not ent_types:
                labels.append(LABEL2ID["O"])
                prev_ent_type = None
            else:
                ent_type = ent_types[0]
                if prev_ent_type == ent_type:
                    tag = f"I-{ent_type}"
                else:
                    tag = f"B-{ent_type}"
                labels.append(LABEL2ID[tag])
                prev_ent_type = ent_type

        all_labels.append(labels)

    tokenized["labels"] = all_labels
    tokenized.pop("offset_mapping")
    return tokenized

# --------- METRICS ---------
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_labels = []
    true_preds = []
    for pred, lab in zip(predictions, labels):
        curr_true = []
        curr_pred = []
        for p_id, l_id in zip(pred, lab):
            if l_id == -100:
                continue
            curr_true.append(ID2LABEL[l_id])
            curr_pred.append(ID2LABEL[p_id])
        true_labels.append(curr_true)
        true_preds.append(curr_pred)

    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    data_dir = Path("data")
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"

    train_items = read_jsonl(train_path)
    dev_items = read_jsonl(dev_path)

    train_dataset = Dataset.from_list(train_items)
    dev_dataset = Dataset.from_list(dev_items)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_fn(examples):
        return tokenize_and_align_labels(examples, tokenizer, max_length=128)

    train_tokenized = train_dataset.map(preprocess_fn, batched=True)
    dev_tokenized = dev_dataset.map(preprocess_fn, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    

    args = TrainingArguments(
          output_dir="out",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("out")
    tokenizer.save_pretrained("out")

    # Detailed report
    preds, labels, _ = trainer.predict(dev_tokenized)
    preds = np.argmax(preds, axis=-1)

    true_labels = []
    true_preds = []
    for pred, lab in zip(preds, labels):
        curr_true = []
        curr_pred = []
        for p_id, l_id in zip(pred, lab):
            if l_id == -100:
                continue
            curr_true.append(ID2LABEL[l_id])
            curr_pred.append(ID2LABEL[p_id])
        true_labels.append(curr_true)
        true_preds.append(curr_pred)

    print(classification_report(true_labels, true_preds))

if __name__ == "__main__":
    main()
