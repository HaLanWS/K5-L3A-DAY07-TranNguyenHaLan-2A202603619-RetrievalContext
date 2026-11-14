"""
Script benchmark so sánh trực tiếp 3 phương pháp Retrieval trên tập Dev (và Test):
1. Dense Retrieval (ChromaDB)
2. Lexical Retrieval (BM25)
3. Hybrid Retrieval (RRF Dense + BM25)

Đo lường: Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian xử lý (latency).
"""

import json
import statistics
import time
from pathlib import Path
from typing import Dict, Any, List

from retrieval import DenseRetriever, LexicalBM25Retriever, HybridRRFRetriever, hit, pack_context


def evaluate_retriever(retriever, questions: List[Dict[str, Any]], budget: int = 1500, top_k: int = 3):
    traces = []
    latencies = []
    for q in questions:
        # Bỏ qua câu không có đáp án khi tính metrics retrieval
        if not q.get('gold'):
            continue

        start = time.perf_counter()
        results = retriever.search(q['question'], top_k=top_k, metadata_filter=q.get('filter'))
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        context, ids = pack_context(results, budget=budget)
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]

        hit1 = bool(marked and marked[0]['hit'])
        hit3 = any(c['hit'] for c in marked[:3])
        context_hit = any(c['hit'] and c['id'] in ids for c in marked)

        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'retriever': retriever.name,
            'top_k': top_k,
            'budget_chars': budget,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': hit1,
            'hit3': hit3,
            'context_hit': context_hit
        })

    n = len(traces)
    summary = {
        'strategy': retriever.name,
        'n': n,
        'hit1': sum(t['hit1'] for t in traces),
        'hit1_rate': round(sum(t['hit1'] for t in traces) / n, 4) if n else 0,
        'hit3': sum(t['hit3'] for t in traces),
        'hit3_rate': round(sum(t['hit3'] for t in traces) / n, 4) if n else 0,
        'context_hit': sum(t['context_hit'] for t in traces),
        'context_evidence_rate': round(sum(t['context_hit'] for t in traces) / n, 4) if n else 0,
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 2) if n else 0,
        'median_latency_ms': round(statistics.median(latencies), 2) if latencies else 0,
        'mean_latency_ms': round(statistics.mean(latencies), 2) if latencies else 0
    }
    return summary, traces


def main():
    root = Path(__file__).resolve().parent.parent
    chunks_path = root / 'data' / 'chunks.json'
    questions_path = root / 'data' / 'questions.json'
    results_dir = root / 'results' / 'retrieval_comparison'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading chunks & questions...")
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    questions = json.loads(questions_path.read_text(encoding='utf-8'))

    dev_questions = [q for q in questions if q.get('split') == 'dev' and q.get('gold')]
    print(f"Evaluating on {len(dev_questions)} DEV questions with gold evidence...\n")

    # Khởi tạo 3 retriever
    print("Initializing retrievers...")
    dense = DenseRetriever()
    bm25 = LexicalBM25Retriever(chunks)
    hybrid = HybridRRFRetriever(dense, bm25, rrf_k=60)

    retrievers = [bm25, dense, hybrid]
    all_summaries = []

    for r in retrievers:
        print(f"Running evaluation for: {r.name}...")
        summary, traces = evaluate_retriever(r, dev_questions, budget=1500, top_k=3)
        all_summaries.append(summary)

        # Lưu traces từng phương pháp
        trace_file = results_dir / f"{r.name}_dev_traces.json"
        trace_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')

    # Lưu bảng tổng hợp
    summary_file = results_dir / "dev_retrieval_summary.json"
    summary_file.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding='utf-8')

    print("\n" + "=" * 65)
    print("KẾT QUẢ SO SÁNH 3 PHƯƠNG PHÁP RETRIEVAL TRÊN DEV (N = 15):")
    print("=" * 65)
    print(f"{'Phương pháp':<18} | {'Hit@1':<8} | {'Hit@3':<8} | {'Context Rate':<14} | {'Latency':<10}")
    print("-" * 65)
    for s in all_summaries:
        h1 = f"{s['hit1']}/{s['n']} ({s['hit1_rate']*100:.1f}%)"
        h3 = f"{s['hit3']}/{s['n']} ({s['hit3_rate']*100:.1f}%)"
        ctx = f"{s['context_hit']}/{s['n']} ({s['context_evidence_rate']*100:.1f}%)"
        lat = f"{s['median_latency_ms']:.1f} ms"
        print(f"{s['strategy']:<18} | {h1:<8} | {h3:<8} | {ctx:<14} | {lat:<10}")
    print("=" * 65)


if __name__ == '__main__':
    main()
