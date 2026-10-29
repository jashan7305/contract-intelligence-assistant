import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text.lower())

class Retriever:
    def __init__(self, passages, method="hybrid", bm25_weight=0.55, tfidf_weight=0.45):
        self.passages = passages
        self.method = method
        self.bm25_weight = float(bm25_weight)
        self.tfidf_weight = float(tfidf_weight)
        corpus = [p.text for p in passages]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None
        self.bm25 = BM25Okapi([tokenize(x) for x in corpus]) if corpus else None

    def search(self, query, top_k=10, min_score=0.0):
        if not self.passages or self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.matrix).ravel()
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        # Normalize BM25 to [0, 1] per query so the blend is interpretable.
        bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
        if self.method == "tfidf":
            scores = tfidf_scores
        elif self.method == "bm25":
            scores = bm25_norm
        else:
            scores = self.bm25_weight * bm25_norm + self.tfidf_weight * tfidf_scores
        order = np.argsort(scores)[::-1]
        results = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                continue
            p = self.passages[int(idx)]
            results.append({
                "score": score, "doc_path": p.doc_path,
                "start": p.start, "end": p.end, "text": p.text,
                "chunk_id": p.chunk_id,
            })
        return results
