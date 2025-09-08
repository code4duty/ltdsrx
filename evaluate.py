import json
import argparse

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def accuracy_at_k(preds, golds, k):
    correct = 0
    total = 0
    for g in golds:
        doc_id = g["doc_id"]
        for i, m in enumerate(g["mentions"]):
            gold_id = f"{doc_id}#{i}"
            true_cids = [c["id"] for c in m["concepts"]]
            if gold_id not in preds:
                continue
            topk = preds[gold_id][:k]
            if any(cid in topk for cid in true_cids):
                correct += 1
            total += 1