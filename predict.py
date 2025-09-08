# src/predict.py
import argparse, json
from search import load_meddra, TfidfBackend, BioBERTBackend
from tqdm import tqdm
#suppress warnings from sentence-transformers
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meddra", default="data/meddra.json")
    ap.add_argument("--test", default="data/test_nolabels.jsonl")
    ap.add_argument("--output", default="outputs/predictions.jsonl")  # save as JSONL
    ap.add_argument("--backend", choices=["tfidf", "biobert"], default="tfidf")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    # Load MedDRA terms
    meddra = load_meddra(args.meddra)
    backend = TfidfBackend(meddra) if args.backend == "tfidf" else BioBERTBackend(meddra)

    out = []
    for ex in tqdm(read_jsonl(args.test), desc="Predicting"):
        doc_id = ex["doc_id"]
        for idx, m in enumerate(ex.get("mentions", [])):
            mention_id = f"{doc_id}#{idx}"
            query = m["text"]
            results = backend.rank(query, args.topk)
            preds = [cid for cid, _ in results]
            out.append({"id": mention_id, "preds": preds})

    # ✅ Write JSONL properly
    with open(args.output, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    main()