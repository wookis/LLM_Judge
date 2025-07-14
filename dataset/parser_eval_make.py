import csv
import ast
import os
from random import choice, choices

input_file = os.path.join(os.path.dirname(__file__), 'midm-base.csv')
input_file2 = os.path.join(os.path.dirname(__file__), '702-samples-answered-by-midm-base-inst-2.3.2.csv')
output_file = os.path.join(os.path.dirname(__file__), 'eval_dataset_midm_base.csv')

# 1. reader2에서 user_content(질문) → assistant 답변(content) 매핑
response_map = {}
with open(input_file2, 'r', encoding='utf-8') as infile2:
    reader2 = csv.DictReader(infile2)
    for row in reader2:
        try:
            if not row['messages'] or not row['answer']:
                continue
            messages = ast.literal_eval(row['messages'])
            answer = ast.literal_eval(row['answer'])
            user_content2 = ''
            assistant_content = ''
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content2 = msg.get('content', '')
            # answer는 dict이고, answer['choices']는 리스트
            if isinstance(answer, dict) and 'choices' in answer:
                for choice in answer['choices']:
                    message = choice.get('message', {})
                    if message.get('role') == 'assistant':
                        assistant_content = message.get('content', '')
                        break  # 첫 번째 assistant 답변만 사용
            if user_content2:
                response_map[user_content2.strip()] = assistant_content
        except Exception as e:
            continue

# 2. reader에서 매칭되는 질문이 있으면 답변을 가져와서 저장
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['no', 'domain', 'task', 'level', 'prompt', 'response']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for idx, row in enumerate(reader):
        try:
            domain = row.get('domain', '')
            task = row.get('task', '')
            level = row.get('level', '')            
            messages = ast.literal_eval(row['messages'])
            user_content = ''
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                    break
            # reader2에서 같은 질문이 있으면 답변 가져오기
            response = response_map.get(user_content.strip(), '')
            writer.writerow({
                'no': idx+1,
                'domain': domain, 
                'task': task, 
                'level': level, 
                'prompt': user_content,
                'response': response
            })
        except Exception as e:
            print(f"{idx}번째 row 파싱 오류: {e}") 
    print(f"{idx+1}개 샘플을 {output_file}로 저장했습니다.")