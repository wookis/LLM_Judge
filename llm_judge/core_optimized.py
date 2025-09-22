import os
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from unittest import result
import numpy as np
from datetime import datetime, timedelta
import csv
from tqdm import tqdm
import re
import pandas as pd
import yaml
from jinja2 import Template
from utils.logger import logger
import gc  # 가비지 컬렉션을 위한 import 추가

PROMPT_CONFIG_PATH = 'config/Judge_template.yaml'

def load_prompt_templates():
    with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

PROMPT_TEMPLATES = load_prompt_templates()

@dataclass
class EvaluationCriteria:
    """평가 기준을 정의하는 클래스"""
    name: str
    description: str
    max_score: float = 10.0
    weight: float = 1.0

@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 클래스"""
    model_name: str
    criteria_name: str
    score: float
    feedback: str
    prompt: str
    response: str
    timestamp: str = datetime.now().isoformat()

class LLMInterface(ABC):
    """LLM과의 상호작용을 위한 추상 기본 클래스"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> Any:
        """프롬프트에 대한 응답을 생성"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        pass

class LLM_MODEL:
    def __init__(self):
        self.models: Dict[str, LLMInterface] = {}
        
    def add_model(self, model: LLMInterface):
        self.models[model.get_model_name()] = model

class LLMJudgeOptimized:
    """메모리 최적화된 LLM Judge 클래스"""
    
    def __init__(self, chunk_size: int = 50):
        self.evaluation_criteria: List[EvaluationCriteria] = []
        self.results: List[EvaluationResult] = []
        self.models: Dict[str, LLMInterface] = {}
        self.eval_start_time: Optional[datetime] = None
        self.eval_end_time: Optional[datetime] = None
        self.prompt_scores: Dict[str, Dict[str, float]] = {}
        self.dataset_filepath: Optional[str] = None
        self.chunk_size = chunk_size  # 청크 크기 설정
    
    def add_criteria(self, criteria: EvaluationCriteria):
        """평가 기준 추가"""
        self.evaluation_criteria.append(criteria)
    
    def add_model(self, model: LLMInterface):
        """평가할 모델 추가"""
        self.models[model.get_model_name()] = model
    
    def load_dataset_chunked(self, filepath: str) -> Iterator[List[Dict[str, str]]]:
        """청크 단위로 데이터셋을 로드하는 제너레이터"""
        self.dataset_filepath = filepath
        
        try:
            # 청크 단위로 CSV 파일 읽기
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, keep_default_na=False):
                dataset_chunk = []
                for _, row in chunk.iterrows():
                    dataset_chunk.append(row.to_dict())
                yield dataset_chunk
        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {filepath}")
            return
        except Exception as e:
            logger.error(f"데이터셋 로드 중 오류 발생: {e}")
            return

    def evaluate_response_optimized(
        self, 
        no: str, 
        domain: str, 
        task: str, 
        level: str, 
        model_name: str, 
        token_usage: int,
        prompt: str,
        response: str,
        reference_data: Optional[str] = None,
        judge_model: Optional[LLMInterface] = None
    ) -> str:
        """메모리 최적화된 응답 평가"""
        
        if not judge_model:
            return "Error: Judge model not provided"

        # task별 동적 평가 프롬프트 생성
        evaluation_prompt = self._create_multi_criteria_prompt(
            task, prompt, response, reference_data, self.evaluation_criteria
        )
        
        # 프롬프트를 임시 파일에 저장 (메모리 절약)
        prompt_data = {
            'no': no,
            'task': task,
            'domain': domain,
            'level': level,
            'prompt': prompt,
            'response': response,
            'reference_data': reference_data or '',
            'evaluation_prompt': evaluation_prompt
        }
        
        # 임시 파일에 프롬프트 저장
        self._save_prompt_to_temp_file(prompt_data)
        
        # LLM 호출
        judge_response = judge_model.generate_response(evaluation_prompt)
        
        # 응답 처리 및 메모리 정리
        score_feedback = self._extract_response_content(judge_response)
        
        # 메모리 정리
        del evaluation_prompt
        del prompt_data
        gc.collect()
        
        return judge_response

    def _save_prompt_to_temp_file(self, prompt_data: Dict[str, Any]):
        """프롬프트를 임시 파일에 저장"""
        output_prompt_file = "dataset/4.eval_result_data/250911-102500-prompt_df3.csv"
        
        # DataFrame 생성 및 저장
        prompt_df = pd.DataFrame([prompt_data])
        prompt_df.to_csv(output_prompt_file, index=False, encoding='utf-8-sig', mode='a', header=False)

    def _extract_response_content(self, judge_response) -> str:
        """응답 객체에서 content 추출"""
        if judge_response is None:
            return "Error: Judge model returned None"
            
        try:
            if hasattr(judge_response, 'choices') and judge_response.choices and hasattr(judge_response.choices[0], 'message'):
                return judge_response.choices[0].message.content or ""
            elif hasattr(judge_response, 'content') and judge_response.content:
                if isinstance(judge_response.content, list) and len(judge_response.content) > 0:
                    return judge_response.content[0].text
                else:
                    return str(judge_response.content)
            else:
                return str(judge_response)
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return str(judge_response)

    def run_evaluation_on_dataset_optimized(self, 
                                          filepath: str, 
                                          judge_model: Optional[LLMInterface] = None):
        """메모리 최적화된 데이터셋 평가 실행"""
        self.eval_start_time = datetime.now()
        
        output_file = 'dataset/4.eval_result_data/702-multi_judge.csv'
        
        # 결과 파일 초기화
        self._initialize_result_file(output_file)
        
        processed_count = 0
        
        # 청크 단위로 처리
        for chunk_idx, dataset_chunk in enumerate(self.load_dataset_chunked(filepath)):
            logger.info(f"청크 {chunk_idx + 1} 처리 중... ({len(dataset_chunk)}개 항목)")
            
            chunk_results = []
            
            for item in tqdm(dataset_chunk, desc=f"청크 {chunk_idx + 1} 평가"):
                no = item.get("no")
                domain = item.get("domain")
                task = item.get("task")
                level = item.get("level")
                model_name = item.get("model")
                token_usage = item.get("token_usage")
                prompt = item.get("prompt")
                
                if not prompt:
                    continue
                
                reference_data = item.get("reference_data")
                response = item.get("response")
                
                # task 리스트 처리
                task_list = str(task).split(",")
                for task_item in task_list:
                    task_item = task_item.strip()
                    if not task_item:
                        continue
                    
                    eval_response = self.evaluate_response_optimized(
                        no=str(no) if no is not None else "",
                        domain=str(domain) if domain is not None else "",
                        task=task_item,
                        level=str(level) if level is not None else "",
                        model_name=model_name,
                        token_usage=token_usage,
                        prompt=prompt,
                        response=response,
                        reference_data=reference_data,
                        judge_model=judge_model
                    )
                    
                    # 결과 저장
                    result_data = {
                        'no': no,
                        'domain': domain,
                        'task': task_item,
                        'level': level,
                        'model': model_name,
                        'token_usage': token_usage,
                        'prompt': prompt,
                        'response': response,
                        'reference_data': reference_data,
                        f"{model_name}_result": eval_response
                    }
                    
                    chunk_results.append(result_data)
                    processed_count += 1
                    
                    # 메모리 정리
                    del eval_response
                    gc.collect()
            
            # 청크 결과를 파일에 저장
            if chunk_results:
                self._save_chunk_results(chunk_results, output_file)
                logger.info(f"청크 {chunk_idx + 1} 결과 저장 완료")
            
            # 청크 처리 후 메모리 정리
            del chunk_results
            gc.collect()
            
            logger.info(f"총 처리된 항목: {processed_count}개")
        
        self.eval_end_time = datetime.now()
        logger.info(f"평가 완료. 총 처리 시간: {self.eval_end_time - self.eval_start_time}")

    def _initialize_result_file(self, output_file: str):
        """결과 파일 초기화"""
        # 결과 파일이 존재하면 삭제하고 새로 생성
        if os.path.exists(output_file):
            os.remove(output_file)

    def _save_chunk_results(self, chunk_results: List[Dict[str, Any]], output_file: str):
        """청크 결과를 파일에 저장"""
        df_chunk = pd.DataFrame(chunk_results)
        
        # 파일이 존재하지 않으면 헤더와 함께 저장
        if not os.path.exists(output_file):
            df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            # 기존 파일에 추가
            df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=False)

    def _create_multi_criteria_prompt(self, task: str, prompt: str, response: str, reference_data: Optional[str], criteria_list: List[EvaluationCriteria]) -> str:
        """여러 평가 기준을 한 번에 묻는 프롬프트 생성"""
        if task == 'T10'or task == 'T9':
            template_str = PROMPT_TEMPLATES['Knowledge_template']
        elif task == 'T3' or task == 'T4' or task == 'T8':
            template_str = PROMPT_TEMPLATES['Reason_template']
        elif task == 'T5' or task == 'T6' or task == 'T7':
            template_str = PROMPT_TEMPLATES['Creative_template']
        elif task == 'T2':
            template_str = PROMPT_TEMPLATES['Summary_template']
        elif task == 'T1' or task == 'T11' or task == 'T12':
            template_str = PROMPT_TEMPLATES['Reason_template']
        else:
            template_str = PROMPT_TEMPLATES['Reason_template']   
        
        template = Template(template_str)
        return template.render(prompt=prompt, response=response, reference_answer=reference_data, criteria_list=criteria_list)

    def get_memory_usage_info(self) -> Dict[str, Any]:
        """메모리 사용량 정보 반환"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # 실제 메모리 사용량 (MB)
            "vms_mb": memory_info.vms / 1024 / 1024,  # 가상 메모리 사용량 (MB)
            "chunk_size": self.chunk_size,
            "processed_items": len(self.results) if hasattr(self, 'results') else 0
        }




