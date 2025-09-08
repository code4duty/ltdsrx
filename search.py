# src/search.py
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class TfidfBackend:
    def __init__(self, meddra):
        self.meddra = meddra
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
        self.concept_strings, self.concept_owner = self._prepare_terms()
        self.matrix = self.vectorizer.fit_transform(self.concept_strings)

    def _prepare_terms(self):
        strings, owners = [], []
        for cid, syns in self.meddra.items():
            for s in syns:
                strings.append(s)
                owners.append(cid)
            strings.append(" ; ".join(syns))  # joined variant
            owners.append(cid)
        return strings, owners

    def rank(self, query, topk=10):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        best_by_cid = defaultdict(float)
        for i, cid in enumerate(self.concept_owner):
            best_by_cid[cid] = max(best_by_cid[cid], sims[i])
        return sorted(best_by_cid.items(), key=lambda x: x[1], reverse=True)[:topk]


class BioBERTBackend:
    def __init__(self, meddra):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")
        self.meddra = meddra
        self.concept_strings, self.concept_owner = self._prepare_terms()
        self.emb = self.model.encode(self.concept_strings, show_progress_bar=True)

    def _prepare_terms(self):
        strings, owners = [], []
        for cid, syns in self.meddra.items():
            for s in syns:
                strings.append(s)
                owners.append(cid)
            strings.append(" ; ".join(syns))
            owners.append(cid)
        return strings, owners

    def rank(self, query, topk=10):
        qv = self.model.encode([query])[0]
        sims = np.dot(self.emb, qv) / (np.linalg.norm(self.emb, axis=1) * np.linalg.norm(qv))
        best_by_cid = defaultdict(float)
        for i, cid in enumerate(self.concept_owner):
            best_by_cid[cid] = max(best_by_cid[cid], sims[i])
        return sorted(best_by_cid.items(), key=lambda x: x[1], reverse=True)[:topk]


def load_meddra(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(cid): list(set(v)) for cid, v in data.items()}