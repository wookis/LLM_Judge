import json
from llm_judge.core import LLMJudge, EvaluationCriteria
from llm_judge.llm_interfaces import OpenAILLM, AnthropicLLM
from dotenv import load_dotenv

load_dotenv()

def main():
    """LLM Judge 실행을 위한 메인 함수"""
    
    # 참고: 이 예제를 실행하려면 OPENAI_API_KEY와 ANTHROPIC_API_KEY 환경 변수를 설정해야 합니다.
    # 필요한 라이브러리: pip install openai anthropic tqdm numpy
    
    # 1. LLM Judge 인스턴스 생성
    judge = LLMJudge()

    # 2. 평가 기준 추가
    judge.add_criteria(EvaluationCriteria(name="정확성", description="응답이 사실에 근거하고 정확한가?", weight=1.5))
    judge.add_criteria(EvaluationCriteria(name="완결성", description="응답이 질문의 모든 부분을 다루는가?"))
    judge.add_criteria(EvaluationCriteria(name="스타일", description="지정된 스타일(예: 전문가, 친근함)을 잘 따르는가?", max_score=5.0))
    judge.add_criteria(EvaluationCriteria(name="총점", description="정확성, 완결성, 스타일 점수에 가중치를 포함한 최종 점수를 0-1사이 소수점 3자리로 계산해주세요.", max_score=1.0))

    # 3. 평가할 LLM 모델 추가
    try:
        gpt4_o = OpenAILLM(model_name="gpt-4o")
        #midm2 = OpenAILLM(model_name="midm2")
        #GPT_K = OpenAILLM(model_name="GPT_K")
        #claude3_opus = AnthropicLLM(model_name="claude-3-opus-20240229")
        judge.add_model(gpt4_o)
        #judge.add_model(midm2)
        #judge.add_model(GPT_K)
        #judge.add_model(claude3_opus)

        # 4. Judge용 모델 설정 (예: GPT-4o를 심판으로 사용)
        judge_model = gpt4_o

        # 5. 평가 데이터셋 로드
        dataset = judge.load_dataset("dataset/eval_dataset.csv")
        
        if dataset:
            #6. 응답셋 생성 
            print("\n--- 데이터셋 기반 응답 생성 ---")
            judge.generate_responses_on_dataset(dataset)
            
            #7. 데이터셋 기반 평가 실행
            print("\n--- 데이터셋 기반 평가 시작 ---")
            #judge.evaluate_responses_on_dataset(dataset, judge_model)
            judge.run_evaluation_on_dataset(dataset, judge_model)
            
            #8. 결과 요약 및 저장
            summary = judge.get_results_summary()
            print("\n[종합 요약]")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            
            judge.save_results("evaluation_results_from_dataset.json")
            print("\n평가 결과가 'evaluation_results_from_dataset.json' 파일에 저장되었습니다.")

    except (ValueError, ImportError) as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    main() 