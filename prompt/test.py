import yaml

data = {
    'text': '''\n당신은 대규모 언어 모델이 작성한 '요약  사용자 입력 반영도: '''
}

# 기본 dump - \n 그대로 출력
print(yaml.dump(data))

# literal style 적용하려면
def str_presenter(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

print(yaml.dump(data['text'], allow_unicode=True))
# print(yaml.Dumper.yaml_representers)
# print(type(data['text']))

with open('.test.yaml', 'w', newline='\n', encoding='utf-8') as f:
    yaml.dump(data, f, default_flow_style=False, 
                allow_unicode=True, sort_keys=False)