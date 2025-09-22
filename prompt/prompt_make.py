#!/usr/bin/env python3
"""
평가용 프롬프트 생성기
기본 공통 프롬프트를 참고하여 템플릿 지정과 평가기준을 추가하여 최종 프롬프트를 생성합니다.
"""

from shlex import join
import yaml
import os
import sys
import codecs
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from jinja2 import Template

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger


@dataclass
class EvaluationCriteria:
    """평가 기준을 정의하는 데이터 클래스"""
    name: str
    description: str
    max_score: float = 10.0
    weight: float = 1.0

@dataclass
class PromptTemplate:
    """프롬프트 템플릿을 정의하는 데이터 클래스"""
    name: str
    template: str
    description: str

class PromptMaker:
    """평가용 프롬프트를 생성하는 클래스"""
    
    def __init__(self, config_path: str = "config/judge_template.yaml"):
        self.config_path = config_path
        self.base_prompts = self._load_base_prompts()
        self.templates = self._create_templates()
        self.evaluation_criteria: List[EvaluationCriteria] = []
    
    def _load_base_prompts(self) -> Dict[str, str]:
        """기본 프롬프트 설정을 로드합니다."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"프롬프트 설정 파일 로드 완료: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML 파싱 오류: {e}")
            return {}
    
    def _create_templates(self) -> Dict[str, PromptTemplate]:
        """기본 템플릿들을 생성합니다."""
        templates = {}
        
        # 동적 평가 기준 생성 템플릿
        if 'Dinamic_criteria_prompt' in self.base_prompts:
            templates['dynamic_criteria'] = PromptTemplate(
                name="동적 평가 기준 생성",
                template=self.base_prompts['Dinamic_criteria_prompt'],
                description="평가 기준을 동적으로 생성하는 프롬프트"
            )
  
        if 'Knowledge_template' in self.base_prompts:
            templates['Knowledge_template'] = PromptTemplate(
                name="지식형",
                template=self.base_prompts['Knowledge_template'],
                description="사실 정확성을 평가하는 프롬프트"
            )

        if 'Reason_template' in self.base_prompts:
            templates['Reason_template'] = PromptTemplate(
                name="설명형",
                template=self.base_prompts['Reason_template'],
                description="단계별 설명의 품질을 평가하는 프롬프트"
            )

        if 'Creative_template' in self.base_prompts:
            templates['Creative_template'] = PromptTemplate(
                name="창의형",
                template=self.base_prompts['Creative_template'],
                description="창작/서술형 콘텐츠의 품질을 평가하는 프롬프트"
            )

        if 'Summary_template' in self.base_prompts:
            templates['Summary_template'] = PromptTemplate(
                name="요약/요약형",
                template=self.base_prompts['Summary_template'],
                description="요약/요약형 평가 프롬프트"
            )

        if 'Compare_template' in self.base_prompts:
            templates['Compare_template'] = PromptTemplate(
                name="비교형",
                template=self.base_prompts['Compare_template'],
                description="비교/분석 답변의 품질을 평가하는 프롬프트"
            )

        if 'meeting_agent_criteria_prompt' in self.base_prompts:
            templates['meeting_agent_criteria_prompt'] = PromptTemplate(
                name="기타 평가 기준 (한국어)",
                template=self.base_prompts['meeting_agent_criteria_prompt'],
                description="회의록 에이전트 평가 프롬프트"
            )
        
        logger.info(f"템플릿 생성 완료: {len(templates)}개")
        return templates
    
    def add_criteria(self, criteria: EvaluationCriteria):
        """평가 기준을 추가합니다."""
        self.evaluation_criteria.append(criteria)
        logger.info(f"평가 기준 추가: {criteria.name}")
    
    def add_criteria_list(self, criteria_list: List[EvaluationCriteria]):
        """평가 기준 리스트를 추가합니다."""
        self.evaluation_criteria.extend(criteria_list)
        logger.info(f"평가 기준 {len(criteria_list)}개 추가")
    
    def clear_criteria(self):
        """평가 기준을 모두 제거합니다."""
        self.evaluation_criteria.clear()
        logger.info("평가 기준 모두 제거")
    
    def generate_prompt(self, 
                       template_name: str,
                       prompt: str,
                       response: str,
                       reference_answer: Optional[str] = None,
                       custom_variables: Optional[Dict[str, Any]] = None) -> str:
        """지정된 템플릿을 사용하여 프롬프트를 생성합니다."""
        
        if template_name not in self.templates:
            raise ValueError(f"템플릿을 찾을 수 없습니다: {template_name}")
        
        template_str = self.templates[template_name].template
        template = Template(template_str)
        
        # 기본 변수 설정
        variables = {
            'prompt': prompt,
            'response': response,
            'criteria_list': self.evaluation_criteria
        }
        
        # 참조 답변이 있는 경우 추가
        if reference_answer:
            variables['reference_answer'] = reference_answer
        
        # 사용자 정의 변수 추가
        if custom_variables:
            variables.update(custom_variables)
        
        try:
            generated_prompt = template.render(**variables)
            
            # \n을 실제 줄바꿈으로 변환
            generated_prompt = generated_prompt.replace('\\n', '\n')
            #logger.info(f"generated_prompt: {generated_prompt}")
            #generated_prompt = generated_prompt.encode().decode('unicode_escape')
            #generated_prompt = codecs.decode(generated_prompt, 'unicode_escape')
            
            logger.info(f"프롬프트 생성 완료: {template_name}")
            return generated_prompt
        except Exception as e:
            logger.error(f"프롬프트 생성 중 오류: {e}")
            raise
    
    def save_prompt_to_yaml(self, 
                           prompt: str, 
                           output_path: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """생성된 프롬프트를 YAML 파일로 저장합니다."""
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 저장할 데이터 구성
        output_data = {
            'prompt': prompt
            # 'metadata': metadata or {
            #     'criteria_count': len(self.evaluation_criteria),
            #     'criteria_names': [c.name for c in self.evaluation_criteria]
            # }
        }
        
        # YAML에서 멀티라인 문자열을 깔끔하게 표시하기 위한 커스텀 representer
        def str_presenter(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        
        # str 타입에 대한 커스텀 representer 등록
        yaml.add_representer(str, str_presenter)

        print("------------------dusan--------------")
        print(output_data)
        # print(yaml.Dumper.yaml_representers)
        #print(yaml.dump(output_data['prompt'].replace(':',"dusan").replace('{','dusan').replace('}','dusan'),allow_unicode=True))
        #print(yaml.dump(output_data['prompt'],allow_unicode=True))
        #print(type(output_data['prompt']))
        with open('test-dusan.yaml', 'w', encoding='utf-8') as f:
            f.write(output_data['prompt'])
        print("------------------dusan--------------")
        
        # 프롬프트에서 \n을 실제 줄바꿈으로 변환 (추가 처리)
        if isinstance(output_data['prompt'], str):
            output_data['prompt'] = output_data['prompt'].replace('\\n', '\n')
        
        with open('eval_prompt2.yaml', 'w', encoding='utf-8') as f:
            f.write(output_data['prompt'])

        # print(output_data['prompt'])
        # # 미리보기 출력 (가독성 향상)
        # print("=== 생성된 YAML 미리보기 ===")
        # print(yaml.dump(output_data, allow_unicode=True, default_flow_style=False))
        # print("==========================")
        try:
            with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
                yaml.dump(output_data, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            logger.info(f"프롬프트가 저장되었습니다: {output_path}")
        except Exception as e:
            logger.error(f"파일 저장 중 오류 발생: {e}")
            raise


    def list_available_templates(self) -> List[str]:
        """사용 가능한 템플릿 목록을 반환합니다."""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[PromptTemplate]:
        """템플릿 정보를 반환합니다."""
        return self.templates.get(template_name)
    
    def get_criteria_summary(self) -> str:
        """추가 설정된 평가 기준 요약을 반환합니다."""
        if not self.evaluation_criteria:
            return "설정된 평가 기준이 없습니다."
        
        summary = "추가 설정된 평가 기준:\n"
        for i, criteria in enumerate(self.evaluation_criteria, 1):
            summary += f"{i}. {criteria.name}: {criteria.description} (최대 점수: {criteria.max_score}, 가중치: {criteria.weight})\n"
        return summary

def create_default_criteria() -> List[EvaluationCriteria]:
    """기본 평가 기준을 생성합니다."""
    return [
        EvaluationCriteria(
            name="의미 왜곡 여부",
            description="스크립트의 발언이나 결정을 잘못 해석하거나 왜곡하지 않았는가?",
            max_score=3.0,
            weight=1.5
        ),
        EvaluationCriteria(
            name="사용자 입력 반영도",
            description="‘회의 목적’, ‘요약 수준’ 등 사용자 요구사항을 충실히 반영했는가?",
            max_score=2.0,
            weight=1.0
        )
    ]


def main():

    prompt_maker = PromptMaker()
    
    # 사용 가능한 템플릿 목록 출력
    print("사용 가능한 템플릿:")
    for template_name in prompt_maker.list_available_templates():
        template_info = prompt_maker.get_template_info(template_name)
        print(f"  - {template_name}: {template_info.description}")

    template_name='Knowledge_template'
    # - Knowledge_template: 사실 정확성을 평가하는 프롬프트
    # - Reason_template: 단계별 설명의 품질을 평가하는 프롬프트
    # - Creative_template: 창작/서술형 콘텐츠의 품질을 평가하는 프롬프트
    # - Summary_template: 요약/요약형 평가 프롬프트
    # - Compare_template: 비교/분석 답변의 품질을 평가하는 프롬프트
        
    # 기본 평가 기준 추가
    #default_criteria = create_default_criteria()
    #prompt_maker.add_criteria_list(default_criteria)
    
    # 평가 기준 요약 출력
    print("\n" + prompt_maker.get_criteria_summary())
    
    # 예시 프롬프트와 응답
    target_prompt = "{{prompt}}"
    target_response = "{{response}}"
    
    # 동적 평가 기준 프롬프트 생성
    print("\n=== 동적 평가 기준 프롬프트 생성 ===")
    try:
        eval_prompt = prompt_maker.generate_prompt(
            template_name,
            prompt=target_prompt,
            response=target_response
        )
        
        # 프롬프트를 YAML 파일로 저장
        prompt_maker.save_prompt_to_yaml(
            prompt=eval_prompt,
            output_path=f"config/{template_name}2.yaml",
            metadata={
                'template_used': 'LLM_criteria_prompt',
                'generated_at': '2024-01-01 v1.0'
            }
        )
        
        print("프롬프트 생성이 완료되었습니다!")
        print(f"생성된 프롬프트 길이: {len(eval_prompt)} 문자")
        
    except Exception as e:
        print(f"프롬프트 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

## - !include config/Judge_template.yaml
