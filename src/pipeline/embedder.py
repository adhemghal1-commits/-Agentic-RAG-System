"""Embedding generation: Local TF-IDF + SVD dense embeddings + optional API embeddings."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    """Protocol for text embedding providers."""

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, queries: list[str]) -> np.ndarray: ...


class LocalHybridEmbedder:
    """Offline dense semantic embedder using TF-IDF + TruncatedSVD (LSA) + L2 normalization.

    Produces compact dense vectors with zero external model downloads,
    fully compatible with Python 3.13 and fast on CPU.
    """

    def __init__(self, n_components: int = 128, max_features: int = 12000) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self.svd: TruncatedSVD | None = None
        self._use_svd = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit TF-IDF + SVD on corpus texts and return L2-normalized dense embeddings."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf_matrix.shape
        effective_components = min(self.n_components, max(1, min(n_samples, n_features) - 1))

        if effective_components >= 2 and n_samples >= 3 and n_features >= 3:
            self.svd = TruncatedSVD(n_components=effective_components, random_state=42)
            dense = self.svd.fit_transform(tfidf_matrix)
            self._use_svd = True
        else:
            self.svd = None
            self._use_svd = False
            dense = tfidf_matrix.toarray()

        dense = normalize(dense, norm="l2", axis=1)
        return dense.astype(np.float32)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Transform query strings into the fitted dense embedding space."""
        if not queries:
            return np.zeros((0, 1), dtype=np.float32)

        tfidf_q = self.vectorizer.transform(queries)
        if self._use_svd and self.svd is not None:
            dense = self.svd.transform(tfidf_q)
        else:
            dense = tfidf_q.toarray()

        dense = normalize(dense, norm="l2", axis=1)
        return dense.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "max_features": self.max_features,
                    "vectorizer": self.vectorizer,
                    "svd": self.svd,
                    "use_svd": self._use_svd,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "LocalHybridEmbedder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_components=state["n_components"], max_features=state["max_features"])
        obj.vectorizer = state["vectorizer"]
        obj.svd = state["svd"]
        obj._use_svd = state["use_svd"]
        return obj


def embed_with_gemini(texts: list[str], api_key: str | None = None, model: str = "text-embedding-004") -> np.ndarray:
    """Optional Gemini API embeddings when user supplies a Gemini API key."""
    from google import genai

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    vectors: list[list[float]] = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.models.embed_content(model=model, contents=batch)
        for emb in resp.embeddings:
            vectors.append(emb.values)
    arr = np.array(vectors, dtype=np.float32)
    return normalize(arr, norm="l2", axis=1).astype(np.float32)

