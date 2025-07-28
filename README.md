# LLM Judge - 대규모 언어 모델 평가 시스템

LLM Judge는 대규모 언어 모델(LLM)의 성능을 체계적으로 평가하고 분석하는 Python 기반 시스템입니다. 다양한 LLM 모델의 응답 품질을 다중 기준으로 평가하고, 결과를 시각화하여 비교 분석할 수 있습니다.

## 🚀 주요 기능

- **다중 LLM 모델 지원**: KT 내부 LLM 모델 및 외부 LLM 모델 평가 가능
- **체계적인 평가 기준**: 정확성, 일관성, 이해도, 완결성, 요구사항 등 다중 기준 평가
- **대규모 데이터셋 처리**: 702-샘플 데이터셋 기반 평가
- **자동화된 평가 프로세스**: 프롬프트 생성부터 결과 분석까지 자동화
- **시각화 대시보드**: Streamlit 기반 결과 시각화 및 비교 분석
- **결과 파싱 및 변환**: CSV → JSON 변환 및 데이터 통합

## 📁 프로젝트 구조

```
LLM_Judge/
├── main.py                          # 메인 실행 파일
├── llm_judge/                       # 핵심 평가 엔진
│   ├── core.py                      # LLMJudge 클래스 및 평가 로직
│   └── llm_interfaces.py            # LLM API 인터페이스
├── dataset/                         # 데이터 처리 및 파싱
│   ├── 1.tldc_data/                 # Labeling(TLDC)데이터
│   ├── 2.answer_data/               # 모델 응답 데이터
│   ├── 3.eval_data/                 # 평가용 데이터셋
│   ├── 4.eval_result_data/          # 평가 결과 CSV
│   ├── 5.matrix_data/               # 변환된 JSON 결과
│   ├── parser_eval_make.py          # 평가 데이터 생성기
│   └── parser_result.py             # 결과 파싱 및 변환
├── matrix/                          # 시각화 및 분석
│   ├── matrix.py                    # Streamlit 대시보드
│   └── matrix.ipynb                 # Jupyter 노트북 분석
├── utils/                           # 유틸리티
│   └── logger.py                    # 로깅 시스템
├── config/                          # 설정 파일
│   └── eval_prompt.yaml             # 평가 프롬프트 템플릿
└── logs/                            # 로그 파일
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
uv sync
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 API 키를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 데이터 준비

평가할 데이터셋을 `dataset/3.eval_data/` 디렉토리에 준비하세요. CSV 형식으로 다음 컬럼이 필요합니다:

- `no`: 샘플 번호
- `domain`: 도메인 (D1~D10)
- `task`: 태스크 (T1~T11)
- `level`: 난이도 (L1~L2)
- `model`: 모델명
- `token_usage`: 토큰 사용량
- `prompt`: 원본 프롬프트
- `response`: 모델 응답

## 🚀 사용법

### 1. 기본 평가 실행

template 파일 형태의 평가 프롬프트 : /config/eval_proompt.yaml
```python
single_criteria_prompt: |
  [role]
  당신은 대규모 언어 모델(LLM)이 생성한 답변의 품질을 평가하는 전문가입니다. 
  제공된 원래 프롬프트에 대한 모델응답을 바탕으로 LLM의 답변 품질을 정확하게 평가해 주세요.
 
  [원래 프롬프트]
  {{prompt}}
  
  [모델응답]
  {{response}}
```

평가 기준 추가 삽입 가능

```python
from llm_judge.core import LLMJudge, EvaluationCriteria
from llm_judge.llm_interfaces import OpenAILLM

# LLM Judge 인스턴스 생성
judge = LLMJudge()

# 평가 기준 추가
judge.add_criteria(EvaluationCriteria(
    name="정확성", 
    description="응답이 사실에 근거하고 정확한가?", 
    weight=1.5
))
judge.add_criteria(EvaluationCriteria(
    name="완결성", 
    description="응답이 질문의 모든 부분을 다루는가?"
))

# 모델 추가
gpt4_o = OpenAILLM(model_name="gpt-4o")
judge.add_model(gpt4_o)

# 데이터셋 로드 및 평가 실행
dataset = judge.load_dataset("dataset/3.eval_data/702-samples-eval_gpt-4o.csv")
judge.run_evaluation_on_dataset(dataset, gpt4_o)
```

### 2. 메인 스크립트 실행

```bash
python main.py
```

### 3. 결과 시각화

```bash
cd matrix
streamlit run matrix.py
```

### 4. 다중 CSV 파일 처리

```python
from dataset.parser_result import parse_eval_feedback_to_results

# 여러 CSV 파일을 하나의 JSON으로 통합
csv_files = [
    "dataset/4.eval_result_data/702-samples-eval_feedback_midm-mini-inst-2.3.1.csv",
    "dataset/4.eval_result_data/702-samples-eval_feedback_midm-base-inst-2.3.2.csv",
    "dataset/4.eval_result_data/702-samples-eval_feedback_midm-pro-inst-2.3.csv",
    "dataset/4.eval_result_data/702-samples-eval_feedback_llama-3-1-74b.csv"
]

results = parse_eval_feedback_to_results(
    csv_files, 
    "dataset/5.matrix_data/702-samples-eval_result_all.json"
)
```

## 📊 평가 기준

시스템은 다음 5가지 기준으로 LLM 응답을 평가합니다:

1. **정확성 (Accuracy)**: 제공된 정보의 정확성 (가중치: 1.5)
2. **일관성 (Consistency)**: 응답 내 논리적 일관성 (가중치: 1.0)
3. **이해도 (Understanding)**: 질문 의도 파악 능력 (가중치: 1.3)
4. **완결성 (Completeness)**: 질문에 대한 포괄적 답변 (가중치: 1.0)
5. **요구사항 (Requirements)**: 구체적이고 명확한 답변 (가중치: 1.0)

## 🔧 주요 클래스 및 함수

### LLMJudge
- **핵심 평가 엔진**: 다중 기준 평가 및 결과 관리
- **메서드**:
  - `add_criteria()`: 평가 기준 추가
  - `add_model()`: 평가할 모델 추가
  - `generate_responses_on_dataset()`: 모델 응답 생성
  - `run_evaluation_on_dataset()`: 데이터셋 기반 평가 실행
  - `save_results()`: 결과 저장




## 📈 결과 분석

### 시각화 대시보드
- **히트맵**: 도메인별, 태스크별, 난이도별 점수 시각화
- **모델 비교**: 여러 모델의 성능 비교
- **인터랙티브 필터링**: 모델 및 데이터 필터링

### 데이터 형식
평가 결과는 다음 JSON 형식으로 저장됩니다:

```json
{
  "no": 1,
  "domain": "D1",
  "task": "T1",
  "level": "L1",
  "model": "gpt-4o",
  "token": 150,
  "score": 0.832
}
```

## 🔍 평가 프로세스

1. **데이터 준비**: TLDC 데이터 → 모델 응답 → 평가용 데이터셋
2. **평가 실행**: 다중 기준 평가 및 결과 생성
3. **결과 파싱**: CSV → JSON 변환 및 통합
4. **시각화**: Streamlit 대시보드를 통한 결과 분석

## 📝 평가 셋 생성

평가용 데이터 셋을 생성합니다:
- **parser_eval_make.py**: 평가용 정답셋 규격화(label, response)
- **parser_result.py**: 평가결과 시각화용 데이터 변환
### 로그 확인

```bash
tail -f logs/app.log
```
