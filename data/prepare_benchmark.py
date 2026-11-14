import json
import os

# 1. Load Day 6 questions
day6_q_path = r'c:\Users\Admin\Downloads\K5-L3A-DAY06-TranNguyenHaLan-2A202603619-DataIngestion\benchmark\benchmark_questions.json'
with open(day6_q_path, encoding='utf-8') as f:
    day6_qs = json.load(f)

print('Day 6 questions count:', len(day6_qs))

# 2. Format Answerable Questions (for retrieval baseline and agent)
formatted_qs = []
for q in day6_qs:
    qid = q['id']
    split = q['split']
    ev = q.get('evidence_quote', '')
    
    # Clean whitespace for DEV_03
    if qid == 'DEV_03':
        ev = "Áp dụng cản h báo học tập mức 3 đối với sinh viên có số TC nợ đọng từ đầu khóa lớn hơn 24"
        
    formatted_q = {
        'id': qid,
        'split': split,
        'question': q['question'],
        'gold': [
            {
                'source_id': q.get('gold_source_id', 'QCDT_2023'),
                'page': q.get('gold_page', 1),
                'evidence': ev
            }
        ],
        'expected_answer': q.get('expected_answer', ''),
        'filter': q.get('filter')
    }
    formatted_qs.append(formatted_q)

# 3. Create Cases (24 questions: 17 Dev [15 ans + 2 unans], 7 Test [5 ans + 2 unans])
cases = []
for q in formatted_qs:
    cases.append({
        'id': q['id'],
        'split': q['split'],
        'question': q['question'],
        'answerable': True,
        'expected_behavior': 'search_and_answer'
    })

unanswerable_cases = [
    {
        'id': 'DEV_UNANS_01',
        'split': 'dev',
        'question': 'Mức học phí chi tiết theo từng tín chỉ của chương trình tiên tiến năm học 2024-2025 là bao nhiêu triệu đồng?',
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Tài liệu Quy chế đào tạo (QCDT_2023 và QCDT_2025) chỉ quy định về khung chương trình đào tạo, chuẩn đầu ra và cảnh báo học tập, hoàn toàn không chứa biểu phí học phí chi tiết.',
        'answerable': False
    },
    {
        'id': 'DEV_UNANS_02',
        'split': 'dev',
        'question': 'Sinh viên Đại học Bách khoa Hà Nội có được phép nuôi thú cưng (chó, mèo) trong phòng ký túc xá không?',
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Tài liệu là Quy chế đào tạo, không bao gồm nội quy quản lý ký túc xá hay quy định về thú cưng.',
        'answerable': False
    },
    {
        'id': 'TEST_UNANS_01',
        'split': 'test',
        'question': 'Quy định chi tiết về thời gian đóng mở cổng trường Đại học Bách khoa Hà Nội vào các ngày nghỉ lễ Tết như thế nào?',
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Tài liệu là Quy chế đào tạo, không quy định về công tác an ninh trật tự hay giờ đóng mở cổng trường.',
        'answerable': False
    },
    {
        'id': 'TEST_UNANS_02',
        'split': 'test',
        'question': 'Thủ tục xin cấp thẻ gửi xe tháng và mức phí gửi xe máy đối với sinh viên nội trú là bao nhiêu?',
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Quy chế đào tạo không có điều khoản quy định về dịch vụ trông giữ xe hoặc phí vé xe.',
        'answerable': False
    }
]

cases.extend(unanswerable_cases)

out_dir = r'C:\Users\Admin\Downloads\K5-L3A-DAY07-TranNguyenHaLan-2A202603619-RetrievalContext\data'
with open(os.path.join(out_dir, 'questions.json'), 'w', encoding='utf-8') as f:
    json.dump(formatted_qs, f, ensure_ascii=False, indent=2)

with open(os.path.join(out_dir, 'cases.json'), 'w', encoding='utf-8') as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

dev_c = [c for c in cases if c['split'] == 'dev']
test_c = [c for c in cases if c['split'] == 'test']
dev_ans = len([c for c in dev_c if c['answerable']])
dev_unans = len([c for c in dev_c if not c['answerable']])
test_ans = len([c for c in test_c if c['answerable']])
test_unans = len([c for c in test_c if not c['answerable']])
print(f'Done! Saved {len(formatted_qs)} retrieval questions to data/questions.json')
print(f'Saved {len(cases)} cases to data/cases.json:')
print(f'  - Dev: {len(dev_c)} cases ({dev_ans} ans + {dev_unans} unans)')
print(f'  - Test: {len(test_c)} cases ({test_ans} ans + {test_unans} unans)')
