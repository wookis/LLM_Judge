import json
import os
import psutil
import gc
from llm_judge.core_optimized import LLMJudgeOptimized, EvaluationCriteria
from llm_judge.llm_interfaces import OpenAILLM, KT_MAGMA_DEV_LLM
from dotenv import load_dotenv
from utils.logger import logger
from dataset.parser_result import parse_eval_feedback_to_results
from llm_judge.core import PROMPT_TEMPLATES 

load_dotenv()

def monitor_memory_usage():
    """메모리 사용량 모니터링"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }

def log_memory_usage(stage: str):
    """메모리 사용량 로깅"""
    memory = monitor_memory_usage()
    logger.info(f"[{stage}] 메모리 사용량: {memory['rss_mb']:.1f}MB (RSS), {memory['percent']:.1f}%")

def main():
    """메모리 최적화된 LLM Judge 실행"""
    
    # 초기 메모리 상태 로깅
    log_memory_usage("시작")
    
    # 1. LLM Judge 인스턴스 생성 (청크 크기 30으로 설정)
    judge = LLMJudgeOptimized(chunk_size=30)
    log_memory_usage("Judge 인스턴스 생성")

    # 2. 평가 기준 추가
    judge.add_criteria(EvaluationCriteria(
        name="총점", 
        description="각 평가 점수의 평균을 총점으로 0-1사이 소수점 2자리로 계산해주세요.", 
        max_score=1.0
    ))

    # 3. 평가할 LLM 모델 및 데이터셋 추가
    try:
        gpt4_o = OpenAILLM(model_name="gpt-4o")
        judge.add_model(gpt4_o)
        log_memory_usage("모델 추가")

        # 4. Judge용 모델 설정
        judge_model = gpt4_o

        # 5. 메모리 최적화된 평가 실행
        dataset_file = "dataset/3.eval_data/702-samples-eval_gpt-4o.csv"
        
        if os.path.exists(dataset_file):
            logger.info(f"--- 메모리 최적화된 데이터셋 평가 시작 ---")
            logger.info(f"청크 크기: {judge.chunk_size}")
            
            # 메모리 사용량 모니터링과 함께 평가 실행
            judge.run_evaluation_on_dataset_optimized(dataset_file, judge_model)
            
            log_memory_usage("평가 완료")
            
            # 6. 결과 요약 및 저장
            logger.info("--- 평가 결과 파싱 시작 ---")
            csv_files = [
                "dataset/4.eval_result_data/702-samples-eval_feedback_midm-mini-inst-2.3.1.csv",
                "dataset/4.eval_result_data/702-samples-eval_feedback_midm-base-inst-2.3.2.csv",
                "dataset/4.eval_result_data/702-samples-eval_feedback_midm-pro-inst-2.3.csv",
                "dataset/4.eval_result_data/702-samples-eval_feedback_llama-3-1-74b.csv"
            ]
            output_file = "dataset/5.matrix_data/702-samples-eval_result_all.json"

            # 메모리 정리
            del judge
            del gpt4_o
            gc.collect()
            log_memory_usage("메모리 정리 후")

            # 모든 CSV 파일을 합쳐서 하나의 JSON 파일로 저장
            results = parse_eval_feedback_to_results(csv_files, output_file)
            logger.info(f"파싱 완료: {len(results)}개의 결과 생성")
            
            log_memory_usage("최종 완료")
        else:
            logger.error(f"데이터셋 파일을 찾을 수 없습니다: {dataset_file}")

    except (ValueError, ImportError) as e:
        logger.error(f"오류: {e}")
    except Exception as e:
        logger.error(f"예상치 못한 오류가 발생했습니다: {e}")
    finally:
        # 최종 메모리 정리
        gc.collect()
        log_memory_usage("프로그램 종료")

if __name__ == '__main__':
    main()




