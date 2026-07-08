"""Faithful reproduction of the legacy retrieval logic in legacy/rag.py (tfidf_first_search),
extended only to return a ranked top-k list instead of the single best chunk.

Scoring, preprocessing, filtering, and TF-IDF parameters are copied verbatim from
rag.py so baseline numbers reflect the real pipeline. Do not "improve" this file —
it exists to measure the status quo.

Reads the artifacts produced by `python legacy/run_ingestion.py`
(legacy/vector_store.faiss + legacy/metadata1.json).
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LegacyRetriever:
    def __init__(self, legacy_dir: Path):
        metadata_file = legacy_dir / "metadata1.json"
        if (legacy_dir / "metadata.json").exists():  # same precedence as rag.py
            metadata_file = legacy_dir / "metadata.json"
        self.metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(legacy_dir / "vector_store.faiss"))
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        all_texts = [self._prepare_text_for_search(c) for c in self.metadata]
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            sublinear_tf=True,
            norm="l2",
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

    @staticmethod
    def _prepare_text_for_search(chunk):
        text = chunk.get("text", "").strip()
        pdf_name = chunk.get("pdf", "").strip()
        return f"{pdf_name}. {text}" if pdf_name else text

    @staticmethod
    def _preprocess_query(query):
        query = " ".join(query.split())
        query = re.sub(
            r"^(what|how|where|when|why|which|who)\s+(is|are|was|were|does|do|did|can|could|should|would)\s*",
            "", query, flags=re.IGNORECASE,
        )
        return query.strip()

    @staticmethod
    def _extract_important_keywords(query):
        stop_words = {
            "the", "of", "and", "are", "is", "in", "to", "for", "a", "an", "with", "by", "from",
            "on", "at", "as", "what", "how", "where", "when", "why", "which", "that", "this",
        }
        query_clean = re.sub(r"[^\w\s]", " ", query.lower())
        words = query_clean.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        seen, unique = set(), []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique

    @staticmethod
    def _calculate_keyword_score(keywords, chunk):
        if not keywords:
            return 0.0
        text = chunk.get("text", "").lower()
        if not text.strip():
            return 0.0
        exact = partial = 0
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                exact += 1
            elif kw in text:
                partial += 1
        return min((exact * 1.0 + partial * 0.5) / len(keywords), 1.0)

    @staticmethod
    def _is_valid_chunk(chunk):
        text = chunk.get("text", "").strip()
        if len(text) < 30:
            return False
        if len(re.findall(r"\b\w+\b", text)) < 8:
            return False
        if len(re.findall(r"[^\w\s]", text)) / len(text) > 0.3:
            return False
        return True

    def search(self, query: str, top_k: int = 10):
        """Same candidate scoring as rag.py tfidf_first_search; returns ranked top_k chunks."""
        query_processed = self._preprocess_query(query)

        query_tfidf = self.tfidf_vectorizer.transform([query_processed])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        query_vector = np.array(self.embedder.encode([query_processed])).astype("float32")
        search_k = min(20 * 4, len(self.metadata))  # rag.py calls with top_k=20
        _, faiss_indices = self.index.search(query_vector, search_k)
        faiss_candidates = set(faiss_indices[0])

        query_keywords = self._extract_important_keywords(query)

        candidates = []
        for idx, chunk in enumerate(self.metadata):
            if not self._is_valid_chunk(chunk):
                continue
            tfidf_score = tfidf_scores[idx] if idx < len(tfidf_scores) else 0.0
            if tfidf_score > 0.01:
                faiss_bonus = 0.1 if idx in faiss_candidates else 0.0
                keyword_bonus = self._calculate_keyword_score(query_keywords, chunk) * 0.05
                candidates.append((tfidf_score + faiss_bonus + keyword_bonus, idx))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [self.metadata[idx] for _, idx in candidates[:top_k]]
