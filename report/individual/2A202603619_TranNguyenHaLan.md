# Báo cáo Day 7 — Agentic RAG

**Học viên:** Trần Nguyễn Hà Lan  
**Mã số sinh viên (MSSV):** 2A202603619  
**Chủ đề:** Large Language Models & Agentic Systems (L3A) — Day 07: Retrieval & Context Optimization  

---

## 1. Dữ liệu và benchmark

- **Nguồn tài liệu, quy quyền sử dụng, số trang và số chunk:**
  - Nguồn tài liệu: Văn bản pháp quy đào tạo chính thức của Đại học Bách khoa Hà Nội, gồm:
    1. *Quy chế đào tạo trình độ đại học* ban hành kèm Quyết định số 4600/QĐ-ĐHBK ngày 09/06/2023 (`QCDT_2023`, 26 trang).
    2. *Quy chế đào tạo* ban hành kèm Quyết định số 5445/QĐ-ĐHBK năm 2025 (`QCDT_2025`, 36 trang).
  - Bản quyền & quyền sử dụng: Tài liệu nội bộ công khai phục vụ đào tạo và nghiên cứu học thuật của ĐHBK Hà Nội.
  - Tổng số trang xử lý: 62 trang.
  - Tổng số chunk trích xuất kế thừa từ Day 6: **178 chunks** (sử dụng chiến lược phân đoạn theo cấu trúc văn bản pháp quy `strategy_b_structure` — bảo toàn nguyên vẹn tiêu đề Điều khoản, Khoản, Điểm và bảng biểu).
  - Tệp lưu trữ dữ liệu chunk chuẩn: [`data/chunks.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/data/chunks.json).

- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector database: **ChromaDB** (`PersistentClient`), đường dẫn lưu trữ tại `starter/chroma_db`, collection `strategy_b_structure`.
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (tương thích 100% với Day 6), số chiều vector: **384 dimensions**.
  - Độ đo khoảng cách: Cosine distance ($d \in [0, 2]$), chuẩn hóa similarity score $S = \frac{1}{1 + d}$.

- **Agent model, cách chạy và phân chia dev/test:**
  - Agent Model: **Gemini 3.8 Flash** (`gemini-3.8-flash`) thông qua OpenAI-compatible API endpoint (`https://ai.starimg.ru/v1/chat/completions`), nhiệt độ `temperature = 0.0` để tối đa hóa tính xác định và độ trung thực của câu trả lời.
  - Cơ chế ra quyết định: ReAct / Tool-Calling lặp qua `LLMController.decide()`, xuất định dạng JSON có cấu trúc (`search` / `answer`).
  - Phân chia tập Benchmark:
    - **Tập Dev ($N = 17$ câu)**: Gồm **15 câu có đáp án** (`DEV_01` $\to$ `DEV_15`) kế thừa từ Day 6 và **2 câu unanswerable** (`DEV_UNANS_01`: học phí chi tiết từng tín chỉ chương trình tiên tiến; `DEV_UNANS_02`: quy định nuôi thú cưng trong ký túc xá).
    - **Tập Test ($N = 7$ câu)**: Gồm **5 câu có đáp án** (`TEST_01` $\to$ `TEST_05`) giữ độc lập và **2 câu unanswerable** (`TEST_UNANS_01`: giờ đóng mở cổng trường dịp lễ Tết; `TEST_UNANS_02`: phí gửi xe máy và thẻ xe tháng).
  - Tệp lưu trữ: [`data/questions.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/data/questions.json) và [`data/cases.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/data/cases.json).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có evidence: Cung cấp `gold_source_id`, `gold_page`, và `evidence_quote` nguyên văn trong văn bản gốc. Đã kiểm tra 100% bằng chứng nằm trọn trong chunk hợp lệ.
  - Câu thiếu evidence: Thuộc các chủ đề sinh hoạt, tài chính, cơ sở vật chất mà quy chế đào tạo không điều chỉnh; mục tiêu kiểm tra khả năng phát hiện thiếu thông tin và **từ chối an toàn (abstain)**, tránh ảo giác (hallucination).
  - Tiêu chí chấm:
    - Retrieval: `Hit@1` (top-1 chứa gold evidence), `Evidence Hit@3` (top-3 chứa gold evidence), `Context Evidence Rate` (bằng chứng lọt vào context cuối cùng gửi đến model).
    - Agent: `Grounded Answer` (0: sai/thiếu bằng chứng; 1: đúng 1 phần; 2: đúng và đủ), `Citation Validity` (tỷ lệ citation trỏ đúng chunk có bằng chứng hỗ trợ), `Abstention` (tỷ lệ từ chối đúng trên câu unanswerable).

---

## 2. Retrieval và thử nghiệm

### Bảng đối chiếu các phương pháp Retrieval và cấu hình thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / Thay đổi kỹ thuật | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / Chi phí |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **Baseline** | Dense ChromaDB, $k=3$, budget=800 | Cơ sở: dùng vector embedding MiniLM Day 6, cắt context 800 ký tự | dev | 3/15 (20.0%) | 5/15 (33.3%) | 4/15 (26.7%) | 9.6 ms / $0 |
| **Phương pháp 2** | Lexical BM25, $k=3$, budget=1500 | Dùng Okapi BM25 chuẩn hóa tiếng Việt, tăng budget lên 1500 | dev | 6/15 (40.0%) | 10/15 (66.7%) | 6/15 (40.0%) | 4.9 ms / $0 |
| **Phương pháp 3** | Hybrid RRF (Dense + BM25), $k=3$, budget=1500 | Kết hợp Reciprocal Rank Fusion ($rrf\_k=60$) giữa Dense và BM25 | dev | 4/15 (26.7%) | 7/15 (46.7%) | 4/15 (26.7%) | 14.3 ms / $0 |
| **Cải tiến 1** | Dense ChromaDB, $k=5$, budget=2500 | Giả thuyết: Tăng top-k và nới rộng budget giải quyết nghẽn ngữ cảnh | dev | 3/15 (20.0%) | 6/15 (40.0%) | 6/15 (40.0%) | 10.2 ms / $0 |
| **Cải tiến 2** | Hybrid RRF, $k=5$, budget=2500 | Giả thuyết: Kết hợp cả từ khóa chính xác và ngữ nghĩa sâu với budget rộng | dev | 7/15 (46.7%) | 11/15 (73.3%) | 9/15 (60.0%) | 15.8 ms / $0 |
| **Cấu hình đã chọn** | **Hybrid RRF, $k=5$, budget=2500** | Cấu hình tối ưu chốt từ Dev đem sang đánh giá độc lập | **test** | **2/5 (40.0%)** | **4/5 (80.0%)** | **3/5 (60.0%)** | **16.0 ms / $0** |
| **Baseline** | Dense ChromaDB, $k=3$, budget=800 | Baseline đối chứng trên cùng tập Test | **test** | 1/5 (20.0%) | 3/5 (60.0%) | 1/5 (20.0%) | 9.4 ms / $0 |

---

### Phân tích 2 ca lỗi trên Dev, nguyên nhân và cách khắc phục:

1. **Ca lỗi 1 — Lỗi nghẽn Context Budget ở Baseline (`DEV_01`)**:
   - *Câu hỏi:* "Theo quy định, một năm học tại ĐHBK Hà Nội bao gồm những học kỳ nào?"
   - *Hiện tượng:* Retriever tìm được chunk đúng `B_QCDT_2023_p6_8` ở rank 3. Tuy nhiên hàm `pack_context()` với `budget=800` đã bị lấp đầy bởi 2 chunk đầu tiên (`B_QCDT_2025_p24_146` dài 140 ký tự và `B_QCDT_2023_p10_20` dài 380 ký tự). Khi thêm chunk thứ 3 (dài hơn 600 ký tự), tổng độ dài vượt quá 800 nên chunk chứa đáp án bị loại bỏ hoàn toàn khỏi context.
   - *Hậu quả:* Model nhận context rỗng bằng chứng nên chuyển sang `abstain` (từ chối nhầm).
   - *Khắc phục:* Mở rộng context budget lên **2500 ký tự** kết hợp `top_k=5`. Kết quả ở Cải tiến 2: chunk `B_QCDT_2023_p6_8` lọt trọn vẹn vào context, Agent trả lời chính xác đạt điểm tối đa 2/2.

2. **Ca lỗi 2 — Lỗi từ viết tắt pháp quy ở Dense Retrieval (`DEV_03` và `DEV_08`)**:
   - *Câu hỏi:* "Sinh viên bị áp dụng cảnh báo học tập mức 3 khi nào?"
   - *Nguyên nhân:* Văn bản gốc sử dụng từ viết tắt đặc thù ngành đào tạo: "số TC nợ đọng", "CPA từ trung bình trở lên". Embedding model đa ngôn ngữ `all-MiniLM-L6-v2` chỉ bắt được ngữ nghĩa khái quát nhưng vector similarity bị loãng bởi các từ khóa viết tắt "TC", dẫn đến rank của chunk quy định rơi ra ngoài top-3.
   - *Khắc phục:* Triển khai **Hybrid RRF** tích hợp Okapi BM25. Do BM25 chấm điểm rất cao cho các term cụm như "cảnh báo học tập mức 3", "TC nợ đọng", thuật toán RRF đã kéo chunk `B_QCDT_2023_p17_40` và `B_QCDT_2025_p18_131` lên vị trí rank 1 và rank 2, giúp Agent nhận diện chính xác điều khoản.

---

## 3. Agent và đánh giá

- **Thiết kế Agent Controller & Công cụ:**
  - Cơ chế điều phối: Sử dụng `LLMController` giao tiếp qua chuẩn JSON:
    - Nếu thông tin chưa đủ: model trả về action `{"action": "search", "query": "..."}`.
    - Khi context đã có bằng chứng: model tổng hợp câu trả lời kèm citation: `{"action": "answer", "answer": "... [chunk-id]", "citations": ["chunk-id"]}`.
    - Nếu câu hỏi ngoài phạm vi: model trả về `{"action": "answer", "answer": "...", "citations": []}` kích hoạt trạng thái `abstain`.
  - Giới hạn số lượt gọi: `max_searches = 3`. Nếu sau 3 lượt tìm kiếm không đủ bằng chứng, hệ thống tự động ngắt với trạng thái `abstain` an toàn (tránh lặp vô tận).
  - Quản lý Context: Cơ chế tích lũy ứng viên `candidates`, lọc trùng theo `chunk_id` (deduplication) và đóng gói tuần tự trong ngân sách ký tự `budget = 2500`.

### Bảng đánh giá hiệu năng Agent trên Dev và Test

| Cấu hình / Split | Tool use | Số lượt gọi TB | Grounded answer (0 / 1 / 2) | Grounded Mean (/2.0) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm (False Abstention) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dense, $k=3$, bg=800)** / dev | 17/17 (100%) | 3.0 | 15 / 0 / 0 | **0.00** | 0/0 (N/A) | 0/0 (N/A) | 2/2 (100%) | 15/15 (100%) |
| **Cải tiến 1 (Dense, $k=5$, bg=2500)** / dev | 16/16 (100%) | 2.1 | 8 / 1 / 5 | **0.79** | 8/8 (100%) | 6/6 (100%) | 2/2 (100%) | 8/14 (57.1%) |
| **Cải tiến 2 (Hybrid RRF, $k=5$, bg=2500)** / dev | 14/14 (100%) | 1.8 | 4 / 2 / 6 | **1.17** | 12/12 (100%) | 8/8 (100%) | 2/2 (100%) | 4/12 (33.3%) |
| **Cấu hình chọn (Hybrid RRF, $k=5$)** / **TEST** | **7/7 (100%)** | **1.6** | **0 / 1 / 4** | **1.80** | **9/9 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

---

### Minh họa chi tiết các trường hợp thực tế trên tập TEST:

| Query ID | Loại câu | Tool call & kết quả retrieval | Câu trả lời & Trích dẫn (Citation) | Nhãn & Lý giải audit |
|---|---|---|---|---|
| `TEST_01` | Có đáp án (Grounding) | Gọi `search` với query: *"hạ một mức cảnh báo học tập mức 1 mức 2"* $\to$ Trả về top-5 chunk (`B_QCDT_2025_p18_131`, `B_QCDT_2023_p17_40`, ...) | *"Theo Điều 19 Quy chế đào tạo [B_QCDT_2025_p18_131], sinh viên đang bị cảnh báo học tập mức 1 và mức 2 sẽ được hạ một mức cảnh báo học tập nếu số tín chỉ (TC) không đạt trong học kỳ nhỏ hơn hoặc bằng 4 [B_QCDT_2025_p18_131] [B_QCDT_2023_p17_40]."* | **Điểm: 2/2**. Đáp án chính xác tuyệt đối, trích dẫn đúng cả 2 phiên bản quy chế 2023 và 2025. |
| `TEST_03` | Có đáp án (Grounding) | Gọi `search` với query: *"điểm trung bình tích lũy CPA đăng ký học chương trình thứ hai"* $\to$ Trả về `B_QCDT_2025_p18_129` (Điều 18) | *"Theo quy định tại khoản 2 Điều 18 [B_QCDT_2025_p18_129] [B_QCDT_2023_p16_37], tại thời điểm đăng ký học chương trình thứ hai, sinh viên phải có học lực tính theo điểm trung bình tích lũy (CPA) xếp loại từ trung bình trở lên..."* | **Điểm: 2/2**. Nêu đúng điều kiện CPA từ trung bình trở lên, citation hỗ trợ trực tiếp. |
| `TEST_UNANS_01` | Thiếu evidence (Abstention) | Gọi `search`: *"thời gian đóng mở cổng trường Đại học Bách khoa Hà Nội ngày nghỉ lễ Tết"* $\to$ Context không có thông tin quy định an ninh/giờ mở cổng. | *"Tài liệu không có đủ bằng chứng để trả lời câu hỏi về thời gian đóng mở cổng trường Đại học Bách khoa Hà Nội vào các ngày nghỉ lễ Tết."* (Citations: `[]`) | **Abstention đúng: True**. Agent nhận thức được giới hạn tài liệu, không bịa đặt nội quy ngoài quy chế. |
| `TEST_UNANS_02` | Thiếu evidence (Abstention) | Gọi `search`: *"thủ tục cấp thẻ gửi xe tháng mức phí gửi xe máy sinh viên nội trú"* $\to$ Kết quả retrieval không có căn cứ về vé xe. | *"Tài liệu không có đủ bằng chứng để trả lời."* (Citations: `[]`) | **Abstention đúng: True**. Hệ thống tự động chuyển trạng thái `abstain` an toàn. |

---

## 4. Tái lập và đường dẫn kiểm chứng

- **Đường dẫn tệp cấu hình và mã nguồn:**
  - Bộ dữ liệu Chunks: [`data/chunks.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/data/chunks.json)
  - Bộ câu hỏi Benchmark: [`data/questions.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/data/questions.json) và [`data/cases.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/data/cases.json)
  - Module 3 phương pháp Retrieval: [`src/retrieval.py`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/src/retrieval.py)
  - Controller LLM: [`src/controller.py`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/src/controller.py)
  - Agentic Loop: [`src/agent.py`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/src/agent.py)
  - Script chạy benchmark: [`src/run_agent_benchmark.py`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/src/run_agent_benchmark.py)
- **Đường dẫn kết quả log & traces:**
  - Kết quả so sánh Retrieval: [`results/retrieval_comparison/dev_retrieval_summary.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/results/retrieval_comparison/dev_retrieval_summary.json)
  - Trace Baseline Dev: [`results/agent_baseline_dense_dev/traces.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/results/agent_baseline_dense_dev/traces.json)
  - Trace Cải tiến 1 Dev: [`results/agent_dense_exp_budget2500_dev/traces.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/results/agent_dense_exp_budget2500_dev/traces.json)
  - Trace Cải tiến 2 Dev: [`results/agent_hybrid_rrf_budget2500_dev/traces.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/results/agent_hybrid_rrf_budget2500_dev/traces.json)
  - Trace Cấu hình tối ưu trên TEST: [`results/agent_hybrid_rrf_budget2500_test/traces.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/results/agent_hybrid_rrf_budget2500_test/traces.json)
  - Summary đánh giá TEST: [`results/agent_hybrid_rrf_budget2500_test/summary.json`](file:///C:/Users/Admin/Downloads/K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext/results/agent_hybrid_rrf_budget2500_test/summary.json)

- **Lệnh chạy kiểm thử đơn vị (Unit tests):**
  ```powershell
  python -m unittest discover -s tests -v
  ```
  *(Kết quả: 15/15 unit test passed thành công).*

---

## 5. Đánh đổi (Trade-offs), Giới hạn & Kết luận

1. **Đánh đổi giữa Chất lượng vs. Thời gian & Chi phí:**
   - **Baseline (Dense $k=3$, budget 800):** Tốc độ nhanh nhất (trung bình ~9.6ms cho retrieval, 1 tool call duy nhất), chi phí token cực thấp. Tuy nhiên chất lượng hoàn toàn không đạt do hiện tượng nghẽn ngữ cảnh dẫn đến từ chối 100% câu hỏi.
   - **Hybrid RRF ($k=5$, budget 2500):** Tăng nhẹ độ trễ retrieval lên ~16ms và số lượng token ngữ cảnh gấp ~3 lần (~1800–2500 ký tự context). Đổi lại, chất lượng trả lời tăng vọt từ 0.00 lên **1.80/2.0**, tỷ lệ từ chối nhầm giảm về **0%**, độ tin cậy trích dẫn đạt **100%**. Đây là sự đánh đổi hoàn toàn xứng đáng cho một hệ thống Agentic RAG thực tế.

2. **Giới hạn của hệ thống:**
   - Khi văn bản có các bảng biểu phức tạp hoặc phân tán trên nhiều trang, RRF vẫn có xác suất xếp bằng chứng ở vị trí rank 4 hoặc rank 5.
   - Cơ chế gọi LLM qua API mạng đôi khi gặp phải hiện tượng nghẽn mạng cục bộ (network timeout); hệ thống đã được bổ sung retry logic 3 lần và timeout 60s để đảm bảo độ ổn định.

3. **Kết luận:**
   - Việc kết hợp **Hybrid Retrieval (RRF Dense + BM25)** và **quản lý context budget hợp lý (2500 ký tự)** là giải pháp then chốt giúp Agent giải quyết triệt để bài toán hỏi đáp trên tài liệu học thuật và văn bản quy chế.
   - Agent thể hiện sự vượt trội về cả độ trung thực (grounding), khả năng trích dẫn minh bạch (`100% valid citations`) và cơ chế từ chối an toàn khi gặp câu hỏi không có bằng chứng (`100% abstention accuracy`).
