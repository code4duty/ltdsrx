# src/cli.py
import argparse, json
from search import load_meddra, TfidfBackend, BioBERTBackend

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meddra", default="data/meddra.json")
    ap.add_argument("--backend", choices=["tfidf", "biobert"], default="tfidf")
    ap.add_argument("--query", type=str, help="ADE mention to search")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    meddra = load_meddra(args.meddra)
    backend = TfidfBackend(meddra) if args.backend == "tfidf" else BioBERTBackend(meddra)

    results = backend.rank(args.query, args.topk)
    print("\nQuery:", args.query)
    for cid, score in results:
        print(f"{cid}\t{score:.4f}\t{meddra[cid][0]}")

if __name__ == "__main__":
    main()