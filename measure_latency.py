import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Same labels as training
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


def read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    # 1) Load dev data
    data_dir = Path("data")
    dev_path = data_dir / "dev.jsonl"
    dev_items = read_jsonl(dev_path)

    # Just take some texts for latency measurement
    texts = [ex["text"] for ex in dev_items]
    if len(texts) == 0:
        raise ValueError("No dev data found.")
    # Limit to at most 100 examples
    texts = texts[:100]

    # 2) Load model + tokenizer from 'out'
    model_dir = Path("out")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Force CPU
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # 3) Warmup (a few runs not counted in latency)
    for i in range(5):
        text = texts[i % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

    # 4) Measure latency for N runs
    num_runs = 50
    latencies_ms = []

    for i in range(num_runs):
        text = texts[i % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        _ = model(**inputs)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        latencies_ms.append(latency_ms)

    latencies_ms = np.array(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)

    print(f"Number of runs: {num_runs}")
    print(f"Median latency (p50): {p50:.2f} ms")
    print(f"95th percentile latency (p95): {p95:.2f} ms")


if __name__ == "__main__":
    main()
