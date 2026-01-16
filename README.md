# Elasticsearch Rank Eval Grid Search Kit

검색 시스템(쿼리/랭킹/튜닝 파라미터)이 **사용자 의도(정답지)** 를 얼마나 잘 만족하는지 `_rank_eval`로 점수화하고,  
템플릿 파라미터 조합을 **그리드 서치**로 돌려서 “현재 데이터셋에서 가장 좋은 조합”을 뽑는 키트입니다.

> 한 줄 요약:  
> **queries.json(사용자 질의) + gold.json(정답지)** 를 기준으로,  
> **template.json(검색 쿼리 틀)** 을 **grid.yaml(파라미터 조합)** 으로 바꿔가며 점수를 비교

---

## 0. 취지

검색 시스템 만들고 나면 보통 이런 질문이 생깁니다.

- “내 검색이 사용자 의도대로 나오나?”
- “쿼리/랭킹 설정 바꿨는데 좋아진 건가? 나빠진 건가?”
- “boost / operator / minimum_should_match / fuzziness 같은 knob를 뭘로 잡는 게 맞지?”

이 키트는 그걸 **정답지 기반으로 숫자(precision/recall/MRR/DCG 등)** 로 판단하게 해줍니다.  
즉, **“이 검색 시스템이면 사용자 의도대로 이 정도 나온다”** 를 정량적으로 말할 수 있게 하는 도구

---

## `_rank_eval`이란?

`_rank_eval`은 간단히 정리하면:

1) 여러 개의 검색 쿼리(request)를 실행해보고  
2) 각 쿼리별로 정답지(gold, rated docs)를 참고해서  
3) precision/recall/MRR/DCG 같은 점수를 계산해주는 API.

metric 감으로 이해:
- **Precision@k**: top-k 중 정답 비율
- **Recall@k**: gold 정답 중 top-k에 들어온 비율
- **MRR**: “첫 정답이 몇 등인지” (1등이면 1.0, 2등이면 0.5)
- **DCG**: 정답이 위에 있을수록 + rating이 높을수록 점수 증가  
  (그래서 rating 3/2/1을 주면 “좋은 정답을 더 위로” 튜닝하는 의미가 생김)

---

## 1. 구성 파일/폴더

```
.
├─ run_grid.py             # 실행기(엔진). 여기서 전부 컨트롤함
├─ config.yaml             # 실행 설정(ES 주소/인증/metrics/k/출력폴더/메인메트릭 등)
├─ template.json           # Mustache 템플릿(검색 DSL 뼈대)
├─ queries.json            # 테스트할 쿼리 목록(qid + params)
├─ gold.json               # 정답지(qid별 정답 문서 + rating)
├─ grid.yaml               # 튜닝 파라미터 후보 목록(조합 생성)
├─ setup_demo_index.ps1    # (옵션) 데모 인덱스/샘플 문서 생성
└─ outputs/
   ├─ results.json
   ├─ results.csv
   ├─ best.json
   ├─ best_debug.json      # (있다면) best 조합 눈검증용
   └─ run.log
```

---

## 2. 전체 동작 흐름 (핵심)

`run_grid.py`가 메인 프로그램이고, 나머지는 입력/설정 파일.


1) `grid.yaml`: 파라미터 조합 생성

2) 각 조합마다:
- `queries.json`에 있는 모든 쿼리(q id)를 하나씩 꺼내서
- `template.json`에 (쿼리 파라미터 + 조합)를 넣고
- Elasticsdearch의 `POST /_render/template` 로 “최종 검색 DSL”을 생성

3) NEXT:
- 렌더된 검색 DSL + `gold.json` 정답지를 묶어서
- ES `POST /{index}/_rank_eval` 로 metric 점수를 계산한다

4) 저장:
- 조합별 점수/에러 → `outputs/results.json`, `outputs/results.csv`
- “best 조합 1개” → `outputs/best.json`
- best 조합으로 실제 `_search` top-k 결과를 뽑아서 정답지(gold.json)와 같이 눈으로 비교할 수 있게 → `outputs/best_debug.json`

---

## 3. 사전준비

- Python 3.10+ 권장
- Elasticsearch (Basic Auth 환경 가능)
- 패키지:
  - requests
  - pyyaml
  - pandas
  - tqdm

설치: (윈도우 powershell 기준)

```
pip install requests pyyaml pandas tqdm
```

---

## 4. 빠른 시작 (PowerShell)

### 4.1 (옵션) 데모 인덱스 생성

보통 로컬 환경이면 elasticseaerch가 https연결일 경우 TLS 검증이 필요합니다.  
해당 PowerShell 스크립트는 `-Insecure` 옵션으로 해당 건을 임시로 우회합니다.

```powershell
.\setup_demo_index.ps1 -Insecure
```

### 4.2 config.yaml 설정

예시:

```yaml
# Elasticsearch 연결
es_url: "https://localhost:9200" (localhost를 elasticsearch IP로 변경)
index: "rankeval_demo" (예시 인덱스. 인덱스가 없더라도 자동 생성)

# Basic Auth
auth:
  username: "elastic"
  password: "YOUR_PASSWORD"

# 네트워크
timeout_sec: 30
retries: 3
backoff_base: 1.5

# grid 탐색 옵션
max_combinations: 0         # 0이면 전체 조합 실행
shuffle_combinations: false # true면 조합 순서를 랜덤으로 섞음

# best 선정 기준(1차. 해당 모듈에서는 MRR로 설정)
main_metric: "mean_reciprocal_rank"

# metrics: _rank_eval은 한 번에 metric 1개만 계산하므로 여러 개면 순차 호출
metrics:
  precision:
    k: 10
    relevant_rating_threshold: 1
    ignore_unlabeled: false
  recall:
    k: 10
    relevant_rating_threshold: 1
  mean_reciprocal_rank:
    k: 10
    relevant_rating_threshold: 1
  dcg:
    k: 10
    normalize: false

# best_debug.json용 top-k
debug_k: 10

output_dir: "outputs"
```

### 4.3 실행

```powershell
python .\run_grid.py --config .\config.yaml
```

---

## 5. 각 파일 역할

### 5.1 run_grid.py (엔진 / 실행기)
**역할:** 모든 파일을 읽고, 반복 실행하고, ES 요청하고, 결과 저장까지 전부 수행

**읽는 것**
- `config.yaml` : ES 주소/인증/메트릭/k/출력폴더/메인메트릭 등 실행 옵션
- `template.json` : Mustache 템플릿(쿼리 DSL 뼈대)
- `queries.json` : 테스트할 쿼리 목록(qid + 파라미터)
- `gold.json` : qid별 정답 문서(_id + rating)
- `grid.yaml` : 튜닝할 파라미터 조합 후보(그리드)

**ES에 호출하는 API**
- `POST {es_url}/_render/template`  
  template.json + (queries params + grid combo) → 최종 DSL 생성

- `POST {es_url}/{index}/_rank_eval`  
  렌더된 DSL + gold 정답지로 metric 계산

(기능이 포함되어 있다면)
- `POST {es_url}/{index}/_search`  
  best 조합으로 qid별 top-k hits 뽑아서 best_debug.json 생성

**쓰는 것(outputs)**
- `outputs/results.json` : 전체 조합 결과(raw)
- `outputs/results.csv` : 조합별 점수표(분석/정렬용)
- `outputs/best.json` : best 조합 1개 (main_metric + tie-break 적용)
- `outputs/best_debug.json` : best 조합 눈검증용(있다면)
- `outputs/run.log` : 로그

---

### 5.2 config.yaml (실행 설정 / 런타임 옵션)
**역할:** run_grid.py가 “어떻게 실행할지” 결정

- ES 연결: `es_url`, `index`, `auth`
- 안정성: `timeout_sec`, `retries`, `backoff_base`
- 조합 실행: `max_combinations`, `shuffle_combinations`
- 평가 지표: `metrics` (precision/recall/mrr/dcg 등 + k, threshold)
- best 선정 1차 기준: `main_metric`

> “점수 계산 방식(k/threshold)”과 “best 선정 기준”은 config.yaml에서 정함

---

### 5.3 template.json (검색 쿼리 틀 / Mustache 템플릿)
**역할:** 검색 DSL의 “뼈대”. `{{변수}}` 자리에 값이 들어가서 최종 쿼리가 됨.

예:
- `{{searchKeyword}}` ← 보통 `queries.json`에서 옴
- `{{boost_title}}`, `{{operator}}`, `{{msm_terms}}`, `{{fuzziness}}` ← 보통 `grid.yaml`에서 옴

> “무엇을 튜닝할지(조절 가능한 knob)”는 template.json이 결정하고,  
> grid.yaml은 “그 knob에 넣어볼 값 목록”

---

### 5.4 grid.yaml (파라미터 후보 목록 / 조합 생성기)
**역할:** template.json의 변수들에 넣어볼 값들을 정의

```yaml
boost_title: [0.5, 1.0, 2.0]
operator: ["or","and"]
```

run_grid.py가 파라미터 조합 생성:
- 예시
- (0.5, or)
- (0.5, and)
- (1.0, or)
- ...

> “총 몇 번 실행할지(조합 수)”는 grid.yaml에서 정함

---

### 5.5 queries.json (테스트 쿼리 목록 / 시나리오)
**역할:** 평가할 “질의들” 목록. 각 쿼리는 반드시 `id(q id)`가 있어야 합니다.

```json
[
  {"id":"q1","params":{"searchKeyword":"dream ambition goals"}},
  {"id":"q2","params":{"searchKeyword":"san giorgio manual"}}
]
```

조합마다 각 쿼리마다:
- params = (queries.json params) + (grid 조합 파라미터)
- 이 파라미터로 template 렌더링

> “무슨 검색을 테스트할지”는 queries.json이 정의.

---

### 5.6 gold.json (정답지)
**역할:** 각 q id에 대해 “정답 문서와 등급(rating)”을 정의합니다.

```json
{
  "q1": [
    {"_id":"doc1","rating":3},
    {"_id":"doc2","rating":1}
  ]
}
```

- gold.json의 key `"q1"`은 반드시 queries.json의 `"id":"q1"`과 매칭돼야 함.
- rating은 graded relevance(보통 3/2/1):
  - 3: 완전 정답(가장 관련 있음.)
  - 2: 꽤 관련
  - 1: 약간 관련

> `_index`가 필요한 ES 버전/설정에서는 스크립트가 gold에 `_index`를 자동으로 채우는 방식(코드 반영 버전 기준)으로 처리됨.

> “정답이 뭔지/얼마나 좋은 정답인지”는 gold.json에서 정의

---

## 6. 결과 파일 보는 법

### 6.1 outputs/results.csv
조합별 점수표.
- 엑셀/스프레드시트로 열고
- main_metric 기준 내림차순 정렬
- 동점이면 dcg/recall/precision 같은 보조 지표로 비교

### 6.2 outputs/best.json
best 조합 하나를 요약한 파일.

예:
```json
{
  "main_metric": "mean_reciprocal_rank",
  "tie_breaks": ["dcg","recall","precision"],
  "best_params": { ... },
  "best_scores": { ... }
}
```

- `main_metric`: best 선정 1순위 기준
- `tie_breaks`: main_metric 동점일 때 추가 비교 기준
- `best_params`: template.json의 `{{ }}` 자리에 들어간 최종 값들
- `best_scores`: 그 조합의 전체 평균 점수 요약

> 점수 해석은 절대값보다 “조합 간 비교”가 목적.

### 7.3 outputs/best_debug.json
best 조합으로 각 qid를 실제 `_search` 해서 top-k 결과(_id, _score, title/content)를 저장.

예시
- gold의 rating 3 문서가 1~2등에 올라오나?
- rating 1이 rating 3보다 위에 뜨는 구간이 있나?
- 어떤 q id가 조합에 따라 많이 흔들리나?

---

## 8. 테스트가 의미 있게 되려면 (팁)

- queries는 현실 사용자 질의처럼 “다단어/오타/롱테일”이 섞일수록 좋습니다.
- gold는 “정답이 하나만 있는 쉬운 문제”만 있으면 점수가 다 동점 나옵니다.
- rating(3/2/1)을 주면 DCG 같은 지표가 의미가 생깁니다.
  - “정답이긴 한데(1) 진짜 베스트 정답(3)이 더 위로 올라오게 만들고 싶다” 같은 튜닝이 가능해집니다.

---