import csv
import ast
import os



#input_file = os.path.join(os.path.dirname(__file__), '701-samples-gpt-4.1-mini-without-error-onehot_s.csv')
input_file = os.path.join(os.path.dirname(__file__), '702-samples-answered-by-midm-pro-inst-2.3.csv')
output_file = os.path.join(os.path.dirname(__file__), 'eval_dataset_midm_pro_702.csv')

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['no', 'category', 'domain', 'task', 'level', 'prompt', 'response']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for idx, row in enumerate(reader):
        try:
            if not row['metadata'] or not row['messages']:
                print(f"{idx}번째 row: metadata 또는 messages가 비어있음, 건너뜀")
                continue
            metadata = ast.literal_eval(row['metadata'])
            category = metadata.get('category', '')
            domain = row.get('domain', '')
            task = row.get('task', '')
            level = row.get('level', '')            
            messages = ast.literal_eval(row['messages'])
            # answers = ast.literal_eval(row['answer'])
            user_content = ''
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                    break
            # user_assistant = ''
            # for msg in answers:
            #     if msg.get('role') == 'assistant':
            #         user_assistant = msg.get('content', '')
            #         break
            writer.writerow({
                'no': idx+1,
                'category': category, 
                'domain': domain, 
                'task': task, 
                'level': level, 
                'prompt': user_content,
                # 'answer': user_assistant
            })
            
        except Exception as e:
            print(f"{idx}번째 row 파싱 오류: {e}") 
            
    print(f"{idx+1}개 샘플을 {output_file}로 저장했습니다.")