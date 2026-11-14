"""
Pipeline Agentic RAG Day 7:
Kết hợp LLMController với các Retriever (Dense, BM25, Hybrid)
Thực thi vòng lặp agent, quản lý context budget, citation verification và trace logging.
"""

import json
from typing import Any, Dict, List, Optional
from baseline import pack_context
from controller import LLMController
from retrieval import DenseRetriever, LexicalBM25Retriever, HybridRRFRetriever


class RetrieverAdapter:
    """Adapter bọc các retriever tuân thủ Protocol của starter/agentic.py."""

    def __init__(self, backend_retriever):
        self.backend = backend_retriever

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.backend.search(query, top_k=top_k, metadata_filter=filters)


def run_agent_loop(
    question: str,
    controller: LLMController,
    retriever: RetrieverAdapter,
    *,
    filters: Optional[Dict[str, Any]] = None,
    max_searches: int = 3,
    top_k: int = 3,
    budget: int = 1500
) -> Dict[str, Any]:
    """
    Thực thi vòng lặp Agentic RAG cho một câu hỏi.
    Trả về: status ('answered', 'abstain', 'error'), answer, citations, events.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError("Tham số câu hỏi, max_searches, top_k hoặc budget không hợp lệ")

    history: List[Dict[str, Any]] = []
    chunks: Dict[str, Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for turn in range(max_searches + 1):
        try:
            action = controller.decide(question, history)
        except Exception as e:
            return {
                "status": "error",
                "answer": f"Lỗi gọi LLM: {str(e)}",
                "citations": [],
                "events": events + [{"event": "provider_error", "error": str(e)}]
            }

        act_type = action.get("action")
        if act_type == "search":
            search_count = sum(e["event"] == "search" for e in events)
            if search_count >= max_searches:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng để trả lời sau nhiều lần tìm kiếm.",
                    "citations": [],
                    "events": events + [{"event": "search_limit"}]
                }

            query = action.get("query", "").strip()
            if not query:
                query = question  # Fallback nếu model trả query rỗng

            results = retriever.search(query, filters=filters, top_k=top_k)
            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append({
                "event": "search",
                "query": query,
                "filter": filters,
                "results": results,
                "context": context,
                "context_ids": ids
            })

            history = [{
                "tool": "search",
                "queries": [e["query"] for e in events if e["event"] == "search"],
                "context": context,
                "context_ids": ids
            }]

        elif act_type == "answer":
            answer = action.get("answer", "")
            citations = action.get("citations", [])

            # Nếu không có citation hợp lệ hoặc câu trả lời rỗng -> abstain
            if not answer.strip() or not citations:
                return {
                    "status": "abstain",
                    "answer": answer if answer.strip() else "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "abstain_no_evidence"}]
                }

            # Kiểm tra xem toàn bộ citation có nằm trong context và có xuất hiện trong câu trả lời không
            valid_citations = [
                cid for cid in citations
                if cid in chunks and f"[{cid}]" in answer
            ]

            if valid_citations:
                return {
                    "status": "answered",
                    "answer": answer,
                    "citations": valid_citations,
                    "events": events + [{"event": "answer"}]
                }
            else:
                return {
                    "status": "abstain",
                    "answer": "Không đủ bằng chứng xác thực trong ngữ cảnh.",
                    "citations": [],
                    "events": events + [{"event": "invalid_or_missing_citations"}]
                }
        else:
            # Action không hợp lệ
            return {
                "status": "error",
                "answer": f"Hành động không hợp lệ từ model: {act_type}",
                "citations": [],
                "events": events + [{"event": "invalid_action", "raw": action}]
            }

    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events
    }
