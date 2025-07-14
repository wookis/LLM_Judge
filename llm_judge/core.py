import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import numpy as np
from datetime import datetime, timedelta
import csv
from tqdm import tqdm
import re
import pandas as pd
import yaml
from jinja2 import Template

PROMPT_CONFIG_PATH = 'config/eval_prompt.yaml'

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
    def generate_response(self, prompt: str) -> str:
        """프롬프트에 대한 응답을 생성"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        pass

class LLMJudge:
    """LLM의 성능을 평가하는 메인 클래스"""
    
    def __init__(self):
        self.evaluation_criteria: List[EvaluationCriteria] = []
        self.results: List[EvaluationResult] = []
        self.models: Dict[str, LLMInterface] = {}
        self.eval_start_time: Optional[datetime] = None
        self.eval_end_time: Optional[datetime] = None
        self.prompt_scores: Dict[str, Dict[str, float]] = {}
        self.dataset_filepath: Optional[str] = None
    
    def add_criteria(self, criteria: EvaluationCriteria):
        """평가 기준 추가"""
        self.evaluation_criteria.append(criteria)
    
    def add_model(self, model: LLMInterface):
        """평가할 모델 추가"""
        self.models[model.get_model_name()] = model
    
    def evaluate_response(
        self, 
        model_name: str, 
        prompt: str,
        response: str,
        reference_answer: Optional[str] = None,
        judge_model: Optional[LLMInterface] = None
    ) -> List[EvaluationResult]:
        """특정 프롬프트에 대한 모델의 응답을 평가 (여러 기준을 한 번에 심판 모델에 묻는 구조)"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        results = []
        
        if judge_model:
            # 여러 평가 기준을 한 번에 묻는 프롬프트 생성
            evaluation_prompt = self._create_multi_criteria_prompt(
                prompt, response, reference_answer, self.evaluation_criteria
            )
            score_feedback = judge_model.generate_response(evaluation_prompt)
            print('\n', "***********************score_feedback : ", score_feedback, '\n')
            print('\n', "***********************evaluation_prompt : ", evaluation_prompt, '\n')
            
            try:
                criteria_results = self._parse_multi_criteria_judge_response(score_feedback, self.evaluation_criteria)
            except Exception as e:
                # 파싱 실패 시 -1점 처리
                criteria_results = [
                    (criteria.name, -1.0, f"파싱 실패: {e}") for criteria in self.evaluation_criteria
                ]
            for criteria_name, score, feedback in criteria_results:
                result = EvaluationResult(
                    model_name=model_name,
                    criteria_name=criteria_name,
                    score=score,
                    feedback=feedback,
                    prompt=prompt,
                    response=response
                )
                results.append(result)
                self.results.append(result)
        else:
            # 기존 기본 평가 로직
            for criteria in self.evaluation_criteria:
                score, feedback = self._basic_evaluation(
                    response, reference_answer, criteria
                )
                result = EvaluationResult(
                    model_name=model_name,
                    criteria_name=criteria.name,
                    score=score,
                    feedback=feedback,
                    prompt=prompt,
                    response=response
                )
                results.append(result)
                self.results.append(result)

        # 평가 결과 저장 후 prompt별 가중 총점 계산
        prompt_key = prompt
        model_results_for_prompt = [r for r in self.results if r.prompt == prompt_key and r.model_name == model_name]
        
        # 모든 criteria에 대한 점수를 모음
        scores_by_criteria = {c.name: [] for c in self.evaluation_criteria}
        for r in model_results_for_prompt:
             if r.criteria_name in scores_by_criteria:
                 scores_by_criteria[r.criteria_name].append(r.score)

        final_scores = []
        weights = []
        for c in self.evaluation_criteria:
            # 가장 최근 점수 사용
            score = scores_by_criteria[c.name][-1] if scores_by_criteria[c.name] else -1.0
            final_scores.append(score)
            weights.append(c.weight)

        if sum(weights) > 0:
            valid_scores = [(s, w) for s, w in zip(final_scores, weights) if s != -1.0]
            if valid_scores:
                scores, ws = zip(*valid_scores)
                weighted_score = float(np.average(scores, weights=ws))
            else:
                weighted_score = -1.0
        else:
            weighted_score = float(np.mean([s for s in final_scores if s != -1.0])) if any(s != -1.0 for s in final_scores) else -1.0

        if prompt_key not in self.prompt_scores:
            self.prompt_scores[prompt_key] = {}
        self.prompt_scores[prompt_key][model_name] = weighted_score
        
        return results

    def _create_multi_criteria_prompt(self, prompt: str, response: str, reference_answer: Optional[str], criteria_list: List[EvaluationCriteria]) -> str:
        """여러 평가 기준을 한 번에 묻는 프롬프트 생성"""
        template_str = PROMPT_TEMPLATES['multi_criteria_prompt']
        template = Template(template_str)
        return template.render(prompt=prompt, response=response, reference_answer=reference_answer, criteria_list=criteria_list)

    def _parse_multi_criteria_judge_response(self, response: str, criteria_list: List[EvaluationCriteria]) -> List[tuple]:
        """심판 모델의 표/JSON 응답에서 각 기준별 점수와 피드백을 파싱 (JSON도 지원)"""
        results = []
        found_names = set()
        fail_reason = None
        # 1. JSON 배열 형태 시도
        try:
            json_array = json.loads(response)
            if isinstance(json_array, list):
                for obj in json_array:
                    name = obj.get('기준') or obj.get('criteria_name') or obj.get('name')
                    score = obj.get('점수') or obj.get('score')
                    feedback = obj.get('피드백') or obj.get('feedback')
                    try:
                        score = float(score)
                    except:
                        score = -1.0
                    results.append((name, score, feedback))
                    found_names.add(name)
        except Exception as e:
            fail_reason = f"JSON 배열 파싱 실패: {e}"
        # 2. 여러 개의 JSON 오브젝트가 줄바꿈으로 구분된 경우
        if not results:
            try:
                json_objs = re.findall(r'\{[^\{\}]+\}', response)
                for obj_str in json_objs:
                    try:
                        #print('\n', "***********************obj_str : ", json_objs, '\n')
                        obj = json.loads(obj_str.replace("'", '"'))
                        name = obj.get('기준') or obj.get('criteria_name') or obj.get('name')
                        score = obj.get('점수') or obj.get('score')
                        feedback = obj.get('피드백') or obj.get('feedback')
                        try:
                            print('\n', "score count", score)
                            score = float(score)
                        except:
                            score = -1.0
                            print('\n', "score error", score)
                        results.append((name, score, feedback))
                        found_names.add(name)
                    except Exception as e2:
                        fail_reason = f"JSON 오브젝트 파싱 실패: {e2, obj_str}"
                        print('\n', name)
                        print('\n', score)
                        print('\n', feedback)
                        #print("****************e2 :", e2)
                        continue
            except Exception as e:
                fail_reason = f"JSON 오브젝트 전체 파싱 실패: {e}"
        # 3. 표 형태 파싱 (기존 방식)
        if not results:
            try:
                lines = [l.strip() for l in response.split('\n') if l.strip()]
                start = None
                for i, l in enumerate(lines):
                    if l.startswith('|') and '기준' in l and '점수' in l and '피드백' in l:
                        start = i + 1
                        break
                if start is not None:
                    for l in lines[start:]:
                        if not l.startswith('|'):
                            break
                        parts = [p.strip() for p in l.strip('|').split('|')]
                        if len(parts) < 3:
                            continue
                        name, score, feedback = parts[0], parts[1], parts[2]
                        try:
                            score = float(score)
                        except:
                            score = -1.0
                        results.append((name, score, feedback))
                        found_names.add(name)
            except Exception as e:
                fail_reason = f"표 파싱 실패: {e}"
        # 4. 모든 기준에 대해 결과가 없으면 -1점/실패 원인 메시지로 추가
        name_set = set(found_names)
        for c in criteria_list:
            if c.name not in name_set:
                msg = fail_reason or "심판 응답 파싱 실패 또는 기준 누락"
                results.append((c.name, -1.0, msg))
        # 5. 만약 결과가 아예 없으면 모든 기준 -1점 처리
        if not results:
            msg = fail_reason or "심판 응답 파싱 실패"
            results = [(c.name, -1.0, msg) for c in criteria_list]
        return results

    def _create_evaluation_prompt(self, 
                                prompt: str, 
                                response: str, 
                                reference_answer: Optional[str],
                                criteria: EvaluationCriteria) -> str:
        """평가를 위한 프롬프트 생성"""
        template_str = PROMPT_TEMPLATES['single_criteria_prompt']
        template = Template(template_str)
        return template.render(prompt=prompt, response=response, reference_answer=reference_answer, criteria=criteria)
    
    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """판단 모델의 응답을 파싱"""
        try:
            # 간단한 파싱 로직 - 실제 구현시 더 강건한 파싱 필요
            score_line = [l for l in response.split('\n') if l.startswith('점수:')][0]
            feedback_line = [l for l in response.split('\n') if l.startswith('피드백:')][0]
            
            score = float(score_line.split(':')[1].strip())
            feedback = feedback_line.split(':')[1].strip()
            
            return score, feedback
        except:
            return 0.0, "응답 파싱 실패"
    
    def _basic_evaluation(self, 
                         response: str, 
                         reference_answer: Optional[str],
                         criteria: EvaluationCriteria) -> tuple[float, str]:
        """기본적인 평가 로직"""
        if not reference_answer:
            return 0.0, "참조 답변 없이는 기본 평가를 수행할 수 없습니다."
        
        # 간단한 문자열 유사도 기반 평가 (실제 구현시 더 복잡한 평가 로직 필요)
        similarity = len(set(response.split()) & set(reference_answer.split())) / \
                    len(set(response.split()) | set(reference_answer.split()))
        
        score = similarity * criteria.max_score
        feedback = f"참조 답변과의 유사도: {similarity:.2f}"
        
        return score, feedback
    
    def get_results_summary(self) -> Dict[str, Any]:
        """평가 결과 요약"""
        if not self.results:
            return {"error": "평가 결과가 없습니다."}
        
        summary = {}
        for model_name in self.models.keys():
            model_results = [r for r in self.results if r.model_name == model_name]
            if not model_results:
                continue
                
            criteria_scores = {}
            for criteria in self.evaluation_criteria:
                criteria_results = [r for r in model_results if r.criteria_name == criteria.name]
                if criteria_results:
                    avg_score = np.mean([r.score for r in criteria_results])
                    criteria_scores[criteria.name] = {
                        "average_score": float(avg_score),
                        "weight": criteria.weight
                    }
            
            # 가중 평균 계산
            weights = [criteria_scores[c.name]["weight"] for c in self.evaluation_criteria 
                      if c.name in criteria_scores]
            scores = [criteria_scores[c.name]["average_score"] for c in self.evaluation_criteria 
                     if c.name in criteria_scores]
            
            if weights and scores:
                weighted_average = np.average(scores, weights=weights)
            else:
                weighted_average = 0.0
            
            summary[model_name] = {
                "criteria_scores": criteria_scores,
                "weighted_average": float(weighted_average)
            }
        
        # 평가 소요 시간 추가
        if self.eval_start_time and self.eval_end_time:
            duration = self.eval_end_time - self.eval_start_time
            duration_seconds = int(duration.total_seconds())
            duration_str = str(timedelta(seconds=duration_seconds))
        else:
            duration_str = None
        
        # prompt별 총점도 함께 반환
        prompt_score_summary = self.prompt_scores
        return {
            "summary": summary,
            "prompt_scores": prompt_score_summary,
            "evaluation_start_time": self.eval_start_time.isoformat() if self.eval_start_time else None,
            "evaluation_end_time": self.eval_end_time.isoformat() if self.eval_end_time else None,
            "duration": duration_str
        }
    
    def save_results(self, filepath: str):
        """평가 결과를 JSON 파일로 저장"""
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "results": [vars(r) for r in self.results],
            "summary": self.get_results_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

    def load_dataset(self, filepath: str) -> List[Dict[str, str]]:
        """CSV 파일에서 평가 데이터셋을 로드합니다."""
        self.dataset_filepath = filepath
        dataset = []
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset.append(row)
            print(f"'{filepath}'에서 {len(dataset)}개의 평가 항목을 로드했습니다.")
            return dataset
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다 - {filepath}")
            return []
        except Exception as e:
            print(f"오류: 데이터셋 로드 중 오류 발생 - {e}")
            return []

    def generate_responses_on_dataset(self, dataset: List[Dict[str, str]]):
        """데이터셋 전체에 대해 응답을 생성합니다."""
        self.eval_start_time = datetime.now()

        if not self.dataset_filepath:
            raise ValueError("데이터셋 파일 경로가 설정되지 않았습니다. load_dataset을 먼저 호출해주세요.")

        try:
            df = pd.read_csv(self.dataset_filepath, keep_default_na=False)
        except FileNotFoundError:
            df = pd.DataFrame(dataset)

        dataset_changed = False

        for index, row in tqdm(df.iterrows(), total=len(df), desc="응답 생성 및 저장"):
            prompt = row.get("prompt")
            if not prompt:
                continue

            item = row.to_dict()
            reference_answer = item.get("reference_answer")
            
            for model_name, model in self.models.items():
                response_col = f"{model_name}_response"
                if response_col not in df.columns:
                    df[response_col] = ""                    
                
                response = item.get(response_col)
                # print("\n ***********************************")
                # print("\n model : ", model_name, "response_col1 : ", response)
                # print("\n ***********************************")
                if not response:
                    print(f"응답 생성 중: {model_name} for '{prompt[:20]}...'")
                    response = model.generate_response(prompt)
                    df.loc[index, response_col] = response
                    item[response_col] = response
                    dataset_changed = True
                
        if dataset_changed:
            print(f"\n새로운 응답을 '{self.dataset_filepath}'에 저장합니다.")
            df.to_csv(self.dataset_filepath, index=False, encoding='utf-8-sig')

                    
    def evaluate_responses_on_dataset(self, dataset: List[Dict[str, str]], judge_model: Optional[LLMInterface] = None):
        """데이터셋 전체에 대해 평가를 실행하고 응답을 캐싱합니다."""
        self.eval_start_time = datetime.now()

        if not self.dataset_filepath:
            raise ValueError("데이터셋 파일 경로가 설정되지 않았습니다. load_dataset을 먼저 호출해주세요.")

        try:
            df = pd.read_csv(self.dataset_filepath, keep_default_na=False)
        except FileNotFoundError:
            df = pd.DataFrame(dataset)

        dataset_changed = False

        for index, row in tqdm(df.iterrows(), total=len(df), desc="평가 실행"):
            prompt = row.get("prompt")
            if not prompt:
                continue

            item = row.to_dict()




    def run_evaluation_on_dataset(self, 
                                  dataset: List[Dict[str, str]], 
                                  judge_model: Optional[LLMInterface] = None):
        """데이터셋 전체에 대해 평가를 실행하고 응답을 캐싱합니다."""
        self.eval_start_time = datetime.now()

        if not self.dataset_filepath:
            raise ValueError("데이터셋 파일 경로가 설정되지 않았습니다. load_dataset을 먼저 호출해주세요.")
        
        try:
            df = pd.read_csv(self.dataset_filepath, keep_default_na=False)
        except FileNotFoundError:
            df = pd.DataFrame(dataset)

        dataset_changed = False
        for index, row in tqdm(df.iterrows(), total=len(df), desc="평가 실행"):
            prompt = row.get("prompt")
            if not prompt:
                continue
            
            item = row.to_dict()
            reference_answer = item.get("reference_answer")
            
            for model_name, model in self.models.items():
                response_col = f"{model_name}_response"
                if response_col not in df.columns:
                    df[response_col] = ""                    
                
                response = item.get(response_col)
                print("\n ***********************************")
                print("\n model : ", model_name, "response_col1 : ", response)
                print("\n ***********************************")
                if not response:
                    print(f"응답 생성 중: {model_name} for '{prompt[:20]}...'")
                    response = model.generate_response(prompt)
                    df.loc[index, response_col] = response
                    item[response_col] = response
                    dataset_changed = True

        self.evaluate_response(
            model_name=model_name,
            prompt=prompt,
            response=response,
            reference_answer=reference_answer,
            judge_model=judge_model
        )
        


        if dataset_changed:
            print(f"\n새로운 응답을 '{self.dataset_filepath}'에 저장합니다.")
            df.to_csv(self.dataset_filepath, index=False, encoding='utf-8-sig')
        
        self.eval_end_time = datetime.now() 