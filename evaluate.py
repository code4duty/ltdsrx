import json

def load_gold(gold_file):
    gold = {}
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
           # print(f"Reading line: {repr(line)}")   # <--- Place it here for debugging
            if not line:
                continue  # skip empty lines
            entry = json.loads(line)
            docid = entry.get("doc_id")  # or docid depending on your json keys
            if not docid:
                print(f"Warning: missing 'doc_id' in entry: {entry}")
                continue
            mentions = entry.get("mentions", [])
            for idx, mention in enumerate(mentions):
                mid = f"{docid}#{idx}"
                gold[mid] = [concept["id"] for concept in mention.get("concepts", [])]
    return gold



def load_predictions(pred_file):
    preds = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            preds[entry["id"]] = entry["preds"]
    return preds

def accuracy_at_n(preds, gold, n=1):
    """Compute Accuracy@n metric"""
    correct = 0
    total = 0
    for mid, gold_ids in gold.items():
        if mid in preds:
            top_n_preds = preds[mid][:n]
            if any(g in top_n_preds for g in gold_ids):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    gold_file = 'data/dev_fixed.jsonl'       # path to gold data JSONL file
    pred_file = 'outputs/dev_preds.jsonl'   # path to predicted file JSONL

    gold = load_gold(gold_file)
    preds = load_predictions(pred_file)

    for n in [1, 5, 10]:
        acc = accuracy_at_n(preds, gold, n)
        print(f'Accuracy@{n}: {acc:.4f}')
