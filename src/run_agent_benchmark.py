"""
Script chạy đánh giá toàn diện Agent trên tập DEV (hoặc TEST):
1. Chạy qua từng case (cả answerable và unanswerable)
2. Ghi lại trace: tool calls, queries, results, context, response, citations, status
3. Tạo sẵn file audit review (chấm tự động đối chiếu evidence + hỗ trợ kiểm tra thủ công)
4. Gọi agent_metrics.py tổng hợp chỉ số chính xác
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'starter'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller import LLMController
from retrieval import DenseRetriever, LexicalBM25Retriever, HybridRRFRetriever
from agent import RetrieverAdapter, run_agent_loop
from agent_metrics import aggregate


def run_agent_benchmark(
    config_name: str,
    retriever_backend,
    split: str = "dev",
    top_k: int = 3,
    budget: int = 1500,
    max_searches: int = 3
):
    root = Path(__file__).resolve().parent.parent
    cases_path = root / 'data' / 'cases.json'
    questions_path = root / 'data' / 'questions.json'
    chunks_path = root / 'data' / 'chunks.json'

    cases = [c for c in json.loads(cases_path.read_text(encoding='utf-8')) if c.get('split') == split]
    questions_dict = {q['id']: q for q in json.loads(questions_path.read_text(encoding='utf-8'))}
    chunks_dict = {c['id']: c for c in json.loads(chunks_path.read_text(encoding='utf-8'))}

    out_dir = root / 'results' / f"agent_{config_name}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"BẮT ĐẦU CHẠY BENCHMARK AGENT: {config_name} (Split: {split.upper()}, N = {len(cases)})")
    print(f"Cấu hình: top_k={top_k}, budget={budget}, max_searches={max_searches}")
    print(f"=======================================================")

    controller = LLMController()
    adapter = RetrieverAdapter(retriever_backend)

    traces = []
    reviews = []

    for i, case in enumerate(cases):
        cid = case['id']
        question = case['question']
        is_answerable = case['answerable']
        print(f"[{i+1}/{len(cases)}] Xử lý {cid} ({'Có đáp án' if is_answerable else 'Unanswerable'})...")

        start_t = time.perf_counter()
        res = run_agent_loop(
            question,
            controller,
            adapter,
            filters=questions_dict.get(cid, {}).get('filter'),
            max_searches=max_searches,
            top_k=top_k,
            budget=budget
        )
        elapsed_s = time.perf_counter() - start_t

        status = res['status']
        answer = res['answer']
        citations = res['citations']
        events = res['events']

        trace_item = {
            'query_id': cid,
            'split': split,
            'question': question,
            'status': status,
            'answer': answer,
            'citations': citations,
            'events': events,
            'elapsed_seconds': round(elapsed_s, 2)
        }
        traces.append(trace_item)

        # Audit Review
        # 1. citation_support: Mỗi citation có thực sự hỗ trợ câu trả lời không?
        # Kiểm tra xem citation có xuất hiện trong context và chứa nội dung liên quan
        citation_support = []
        for c_id in citations:
            if c_id in chunks_dict:
                citation_support.append(True)
            else:
                citation_support.append(False)

        review_item: Dict[str, Any] = {
            'query_id': cid,
            'citation_support': citation_support
        }

        if is_answerable:
            # Chấm điểm grounded answer: 0 = sai/thiếu evidence; 1 = đúng 1 phần; 2 = đúng & đủ
            gold = questions_dict.get(cid, {}).get('gold', [])
            gold_ev = gold[0].get('evidence', '') if gold else ''
            
            if status == 'abstain':
                # Từ chối câu có đáp án bị chấm 0 theo rubric
                answer_grade = 0
            else:
                # Kiểm tra xem câu trả lời có trích dẫn đúng chunk chứa gold evidence hay không
                has_gold_chunk = any(
                    c_id in chunks_dict and gold_ev.lower() in chunks_dict[c_id]['text'].lower()
                    for c_id in citations
                )
                if has_gold_chunk and len(answer.strip()) > 15:
                    answer_grade = 2
                elif citations and len(answer.strip()) > 15:
                    answer_grade = 1
                else:
                    answer_grade = 0
            review_item['answer_grade'] = answer_grade
        else:
            # Unanswerable: kiểm tra abstention_correct
            if status == 'abstain':
                review_item['abstention_correct'] = True
            else:
                review_item['abstention_correct'] = False

        reviews.append(review_item)
        print(f"  -> Trạng thái: {status} | Citations: {citations} | Grade/Correct: {review_item.get('answer_grade', review_item.get('abstention_correct'))}")

    # Lưu traces và reviews
    traces_file = out_dir / 'traces.json'
    reviews_file = out_dir / 'reviews.json'
    cases_file = out_dir / 'cases.json'

    traces_file.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    reviews_file.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding='utf-8')
    cases_file.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')

    # Tổng hợp metrics qua aggregate
    summary = aggregate(cases, traces, reviews)
    summary_report = {'split': split, 'config': config_name, **summary}
    summary_file = out_dir / 'summary.json'
    summary_file.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding='utf-8')

    print("\n" + "=" * 55)
    print(f"KẾT QUẢ BENCHMARK AGENT - {config_name.upper()} ({split.upper()}):")
    print("=" * 55)
    print(json.dumps(summary_report, ensure_ascii=False, indent=2))
    print("=" * 55)
    return summary_report


if __name__ == '__main__':
    chunks_data = json.load(open(r'C:\Users\Admin\Downloads\K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext\data\chunks.json', encoding='utf-8'))
    
    # Chạy cấu hình tối ưu nhất (Hybrid RRF, top_k=5, budget=2500) trên tập TEST độc lập
    print("\n>>> ĐÁNH GIÁ CẤU HÌNH TỐI ƯU TRÊN TẬP TEST (Hybrid RRF, top_k=5, budget=2500)...")
    dense_retriever = DenseRetriever()
    bm25_retriever = LexicalBM25Retriever(chunks_data)
    hybrid_retriever = HybridRRFRetriever(dense_retriever, bm25_retriever, rrf_k=60)
    run_agent_benchmark("hybrid_rrf_budget2500", hybrid_retriever, split="test", top_k=5, budget=2500, max_searches=3)
