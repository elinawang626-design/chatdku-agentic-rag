from __future__ import annotations

import math
from collections import Counter, defaultdict

from .models import Chunk, SearchHit
from .utils import dot, expand_query, hashed_embedding, tokenize


class KeywordSearcher:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.term_df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.term_df[term] += 1

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        q_terms = tokenize(expand_query(query))
        if not q_terms:
            return []
        hits: list[SearchHit] = []
        n_docs = max(len(self.chunks), 1)
        for chunk, tokens, doc_len in zip(self.chunks, self.doc_tokens, self.doc_lens):
            tf = Counter(tokens)
            score = 0.0
            for term in q_terms:
                df = self.term_df.get(term, 0)
                if not df or not tf.get(term):
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                numer = tf[term] * (self.k1 + 1)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1))
                score += idf * numer / denom
            if score > 0:
                hits.append(SearchHit(tool="keyword_search", score=score, chunk=chunk))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


class VectorSearcher:
    def __init__(self, chunks: list[Chunk], dims: int = 512) -> None:
        self.chunks = chunks
        self.dims = dims
        self.embeddings = [hashed_embedding(chunk.text, dims=dims) for chunk in chunks]

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        query_vec = hashed_embedding(expand_query(query), dims=self.dims)
        hits: list[SearchHit] = []
        for chunk, vector in zip(self.chunks, self.embeddings):
            score = dot(query_vec, vector)
            if score > 0:
                hits.append(SearchHit(tool="vector_search", score=score, chunk=chunk))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


class HybridSearcher:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.keyword = KeywordSearcher(chunks)
        self.vector = VectorSearcher(chunks)

    def search(self, query: str, limit: int = 6) -> list[SearchHit]:
        merged: dict[str, SearchHit] = {}
        rrf_k = 60

        for rank, hit in enumerate(self.keyword.search(query, limit=limit), start=1):
            merged[hit.chunk.chunk_id] = SearchHit(
                tool="hybrid",
                score=1.0 / (rrf_k + rank),
                chunk=hit.chunk,
            )

        for rank, hit in enumerate(self.vector.search(query, limit=limit), start=1):
            bonus = 1.0 / (rrf_k + rank)
            if hit.chunk.chunk_id in merged:
                merged[hit.chunk.chunk_id].score += bonus
            else:
                merged[hit.chunk.chunk_id] = SearchHit(tool="hybrid", score=bonus, chunk=hit.chunk)
        return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:limit]
