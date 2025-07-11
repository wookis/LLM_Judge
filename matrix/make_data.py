import json
import random

domains = [f'D{i}' for i in range(1, 11)]
tasks = [f'T{i}' for i in range(1, 12)]
models = ['gpt-4o', 'midm2.x', 'GPT K']
levels = ['상', '중', '하']
feedbacks = [
    "정확하고 완성도 높음", "대체로 무난함", "명확한 답변", "일부 부족함", "간결하지만 부족",
    "충분히 설명함", "설명이 명확함", "조금 더 보완 필요", "매우 우수함", "설명이 부족함",
    "보통 수준", "간결함", "정확함", "명확한 설명"
]

data = []
for i in range(1, 101):
    domain = ""
    task = ""
    for model in models:
        if domain=="":
            domain = random.choice(domains)
            task = random.choice(tasks)
        data.append({
            "no": i,
            #"prompt": f"예시 프롬프트 {i}", 
            "domain": domain,
            "task": task,
            #"level": random.choice(levels),
            "model": model,
            "token": random.randint(123, 256),    
            "score": round(random.uniform(0, 1), 3),
            #"feedback": random.choice(feedbacks)
        } 
    )

with open("dataset/eval_results_3.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)