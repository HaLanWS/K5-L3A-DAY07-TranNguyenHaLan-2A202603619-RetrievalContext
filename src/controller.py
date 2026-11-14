"""
Triển khai LLM Controller sử dụng OpenAI compatible API (Gemini 3.8 Flash).
Controller ra quyết định:
1. 'search': Tìm kiếm thêm tài liệu nếu chưa có thông tin
2. 'answer': Trả lời dựa trên các chunk đã tìm được kèm citation [chunk-id]
3. 'abstain': Từ chối nếu tài liệu không có thông tin
"""

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional


class LLMController:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://ai.starimg.ru/v1/chat/completions"
        self.model = model
        self.temperature = temperature

    def decide(self, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Quyết định hành động tiếp theo dựa trên câu hỏi và lịch sử retrieval hiện tại.
        """
        system_prompt = (
            "Bạn là một trợ lý thông minh chuyên trả lời câu hỏi dựa trên tài liệu Quy chế Đào tạo của Đại học Bách khoa Hà Nội.\n"
            "NGUYÊN TẮC BẮT BUỘC:\n"
            "1. Nếu chưa có tài liệu hoặc tài liệu trong lịch sử chưa đủ bằng chứng để trả lời chính xác, bạn PHẢI gọi action 'search' với câu truy vấn thích hợp.\n"
            "2. Khi đã có đủ bằng chứng trong ngữ cảnh tài liệu đã tìm (context), bạn trả về action 'answer':\n"
            "   - 'answer': Chuỗi câu trả lời ngắn gọn, chính xác, nêu đúng số liệu, điều khoản.\n"
            "   - BẮT BUỘC chèn mã trích dẫn dạng [chunk-id] ngay sau thông tin được lấy từ chunk đó (Ví dụ: 'Theo Điều 19 [B_QCDT_2023_p17_40]...').\n"
            "   - 'citations': Danh sách các chunk_id thực sự hỗ trợ cho câu trả lời (Ví dụ: ['B_QCDT_2023_p17_40']).\n"
            "3. Nếu đã tìm kiếm nhưng tài liệu hoàn toàn KHÔNG chứa thông tin liên quan (hoặc câu hỏi hỏi về điều khoản ngoài phạm vi quy chế như học phí cụ thể, thú cưng KTX, cổng trường, vé xe), bạn KHÔNG ĐƯỢC BỊA ĐẶT (hallucination). Bạn PHẢI trả về action 'answer' với nội dung nói rõ tài liệu không đủ bằng chứng, nhưng với citations rỗng: [] (điều này sẽ kích hoạt cơ chế từ chối an toàn của hệ thống).\n"
            "4. Định dạng đầu ra BẮT BUỘC là JSON thuần túy (không kèm markdown ngoài json):\n"
            "   - Dạng 1: {\"action\": \"search\", \"query\": \"cụm từ tìm kiếm\"}\n"
            "   - Dạng 2: {\"action\": \"answer\", \"answer\": \"nội dung trả lời [chunk-id]\", \"citations\": [\"chunk-id\"]}\n"
        )

        user_content = f"Câu hỏi của người dùng: {question}\n\n"
        if not history:
            user_content += "Hiện tại bạn chưa thực hiện tìm kiếm nào. Hãy đưa ra truy vấn tìm kiếm đầu tiên."
        else:
            latest = history[-1]
            user_content += f"Các câu truy vấn trước đó: {latest.get('queries', [])}\n\n"
            user_content += f"Tài liệu đã tìm được (Context hiện tại):\n{latest.get('context', '(Trống)')}\n\n"
            user_content += (
                "Dựa vào các tài liệu trên:\n"
                "- Nếu tài liệu đã có bằng chứng rõ ràng, hãy trả lời và trích dẫn [chunk-id].\n"
                "- Nếu tài liệu chưa đủ và bạn nghĩ cần tìm thêm với từ khóa khác, hãy chọn action 'search'.\n"
                "- Nếu sau khi xem xét kỹ mà tài liệu không có thông tin (câu hỏi ngoài phạm vi quy chế), hãy trả về: {\"action\": \"answer\", \"answer\": \"Tài liệu không có đủ bằng chứng để trả lời.\", \"citations\": []}."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(self.base_url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        
        # Thử lại tối đa 3 lần nếu gặp timeout hoặc lỗi mạng tạm thời
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    
                    # Làm sạch markdown nếu có
                    clean_content = content.strip()
                    if clean_content.startswith("```json"):
                        clean_content = clean_content[7:]
                    if clean_content.startswith("```"):
                        clean_content = clean_content[3:]
                    if clean_content.endswith("```"):
                        clean_content = clean_content[:-3]
                    clean_content = clean_content.strip()

                    parsed = json.loads(clean_content)
                    return parsed
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed in LLMController: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                else:
                    raise e
