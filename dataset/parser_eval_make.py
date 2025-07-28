import csv
import ast
import os

input_file = os.path.join(os.path.dirname(__file__), '1.tldc_data/702-samples-midm-mini.csv')
input_file2 = os.path.join(os.path.dirname(__file__), '2.answer_data/702-samples-answered-by-gpt-4o.csv')
output_file = os.path.join(os.path.dirname(__file__), '3.eval_data/702-samples-eval_gpt-4o.csv')

# 1. reader2에서 user_content(질문) → assistant 답변(content), model, total_tokens 매핑
response_map = {}
model_map = {}
token_map = {}
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
            model = answer.get('model', '')
            usage = answer.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
    
            if isinstance(answer, dict) and 'choices' in answer:
                for choice in answer['choices']:
                    message = choice.get('message', {})
                    if message.get('role') == 'assistant':
                        assistant_content = message.get('content', '')
                        break  # 첫 번째 assistant 답변만 사용
            if user_content2:
                response_map[user_content2.strip()] = assistant_content
                model_map[user_content2.strip()] = model
                token_map[user_content2.strip()] = total_tokens
        except Exception as e:
            continue

# 2. reader에서 매칭되는 질문이 있으면 답변을 가져와서 저장
with open(input_file, 'r', encoding='utf-8') as infile, \
    open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)

    fieldnames = ['no', 'domain', 'task', 'level', 'model', 'token_usage', 'prompt', 'response']
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
            # reader2에서 같은 질문이 있으면 답변, 모델, 토큰 정보 가져오기
            response = response_map.get(user_content.strip(), '')
            model = model_map.get(user_content.strip(), '')
            total_tokens = token_map.get(user_content.strip(), 0)
            writer.writerow({
                'no': idx+1,
                'domain': domain, 
                'task': task, 
                'level': level,
                'model': model,
                'token_usage': total_tokens,
                'prompt': user_content,
                'response': response
            })
        except Exception as e:
            print(f"{idx}번째 row 파싱 오류: {e}") 
    print(f"{idx+1}개 샘플을 {output_file}로 저장했습니다.")