"""
Module triển khai 3 phương pháp retrieval khác nhau cho Day 7:
1. Dense Retrieval (ChromaDB + all-MiniLM-L6-v2 từ Day 6)
2. Lexical Retrieval (BM25 Okapi với chuẩn hóa tiếng Việt)
3. Hybrid Retrieval (Dense + BM25 kết hợp qua Reciprocal Rank Fusion - RRF)
"""

import json
import math
import re
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer


def tokenize_vi(text: str) -> List[str]:
    """Tách từ đơn giản và chuẩn hóa Unicode NFC cho tiếng Việt."""
    normalized = unicodedata.normalize('NFC', text).casefold()
    return re.findall(r'\w+', normalized)


def hit(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    """Kiểm tra chunk có chứa gold evidence đúng source_id và page không."""
    for g in gold:
        if (
            chunk.get('source_id') == g.get('source_id')
            and chunk.get('page') == g.get('page')
            and g.get('evidence', '') in chunk.get('text', '')
        ):
            return True
    return False


def pack_context(results: List[Dict[str, Any]], budget: int = 1500) -> tuple[str, List[str]]:
    """Đóng gói context với deduplication và character budget."""
    blocks = []
    ids = []
    seen = set()
    for c in results:
        cid = c['id']
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source', '')} | page {c.get('page', '')}]\n{c.get('text', '')}"
        current_len = len('\n\n'.join(blocks + [block]))
        if current_len <= budget or not blocks:
            blocks.append(block)
            ids.append(cid)
        else:
            break
    return '\n\n'.join(blocks), ids


class DenseRetriever:
    """Phương pháp 1: Dense Retrieval dùng ChromaDB và model embedding Day 6."""

    def __init__(
        self,
        chroma_path: str = r"c:\Users\Admin\Downloads\K5-L3A-DAY06-TranNguyenHaLan-2A202603619-DataIngestion\starter\chroma_db",
        collection_name: str = "strategy_b_structure",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)
        self.embedder = SentenceTransformer(model_name, local_files_only=True)
        self.name = "dense_chroma"

    def search(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query]).tolist()[0]
        kwargs: Dict[str, Any] = {
            "query_embeddings": [q_emb],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        raw = self.collection.query(**kwargs)
        results = []
        if raw and raw.get("ids") and raw["ids"][0]:
            ids = raw["ids"][0]
            docs = raw["documents"][0]
            metas = raw["metadatas"][0]
            dists = raw["distances"][0] if "distances" in raw and raw["distances"] else [0.0] * len(ids)

            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                results.append({
                    "id": cid,
                    "text": doc,
                    "source_id": meta.get("document_id", "QCDT_2023"),
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 1),
                    "metadata": meta,
                    "score": float(1.0 / (1.0 + dist)),  # Chuẩn hóa distance sang similarity score
                    "distance": float(dist),
                    "retriever": self.name
                })
        return results


class LexicalBM25Retriever:
    """Phương pháp 2: Lexical Retrieval dùng Okapi BM25 tối ưu cho tiếng Việt."""

    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_corpus = [tokenize_vi(c['text']) for c in chunks]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / len(chunks) if chunks else 1.0
        self.nd = len(chunks)
        self.dfs = Counter()
        for doc in self.tokenized_corpus:
            self.dfs.update(set(doc))
        self.idf = {}
        for term, freq in self.dfs.items():
            self.idf[term] = math.log((self.nd - freq + 0.5) / (freq + 0.5) + 1.0)
        self.name = "lexical_bm25"

    def search(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        tokens = tokenize_vi(query)
        scored = []
        for i, (chunk, doc) in enumerate(zip(self.chunks, self.tokenized_corpus)):
            # Áp dụng metadata filter nếu có
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if chunk.get("metadata", {}).get(k) != v and chunk.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = 0.0
            doc_len = self.doc_lens[i]
            for term in tokens:
                if term in self.idf:
                    tf = doc.count(term)
                    if tf > 0:
                        term_score = self.idf[term] * (tf * (self.k1 + 1.0)) / (
                            tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                        )
                        score += term_score

            if score > 0:
                scored.append((score, chunk))

        # Sắp xếp giảm dần theo điểm BM25
        scored.sort(key=lambda x: (-x[0], x[1]['id']))
        results = []
        for s, c in scored[:top_k]:
            results.append({
                **c,
                "score": float(s),
                "retriever": self.name
            })
        return results


class HybridRRFRetriever:
    """Phương pháp 3: Hybrid Retrieval kết hợp Dense + BM25 qua Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: LexicalBM25Retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.name = "hybrid_rrf"

    def search(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        candidate_k = max(top_k * 4, 15)
        dense_results = self.dense.search(query, top_k=candidate_k, metadata_filter=metadata_filter)
        bm25_results = self.bm25.search(query, top_k=candidate_k, metadata_filter=metadata_filter)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Tính RRF score từ Dense: 1 / (k + rank)
        for rank, item in enumerate(dense_results):
            cid = item['id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            chunk_map[cid] = item

        # Tính RRF score từ BM25: 1 / (k + rank)
        for rank, item in enumerate(bm25_results):
            cid = item['id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = item

        # Sắp xếp các ứng viên theo tổng điểm RRF
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: (-rrf_scores[x], x))
        results = []
        for cid in sorted_cids[:top_k]:
            c = chunk_map[cid]
            results.append({
                **c,
                "score": float(rrf_scores[cid]),
                "retriever": self.name
            })
        return results
