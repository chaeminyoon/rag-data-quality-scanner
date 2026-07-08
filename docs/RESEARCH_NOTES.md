# Research Notes — RAG 성능 저하 원인과 설계 근거

이 리팩터링의 설계 결정들이 어떤 연구에 근거하는지 정리한 문서.
아래 두 편은 **원문을 직접 확인**했으며, 각 설계 항목에 근거 조항을 명시한다.

---

## 1. The Power of Noise (Cuconasu et al., SIGIR 2024)

> Cuconasu, F. et al. "The Power of Noise: Redefining Retrieval for RAG Systems."
> SIGIR 2024. [arXiv:2401.14887](https://arxiv.org/abs/2401.14887)

### 문서 4분류 체계 (원문 §3)

| 분류 | 정의 |
|---|---|
| **Gold** | 정답을 포함한 원본 문맥 (NQ 기준 위키피디아 해당 구절) |
| **Relevant** | 정답을 포함하며 쿼리 응답에 유용한 문서 |
| **Related** | 쿼리와 의미적으로 유사하지만 **정답이 없는** 문서 |
| **Irrelevant** | 쿼리와 무관하고 정답도 없는 문서 |

### 핵심 발견 (원문 실험, NQ-open / Llama2·Falcon·Phi-2·MPT / Contriever·BM25·ADORE)

- **Related 문서 1개 추가만으로 정확도 최대 -25%**, 18개면 **-67%** — 근사중복·유사문서가 위험한 진짜 이유
- Irrelevant(랜덤) 문서는 오히려 **최대 +35% 개선** (Near 배치 기준) — 직관과 반대
- Gold 문서 위치: 쿼리 인접(Near) > 끝(Far) > **중간(Mid)이 최악** — "lost in the middle" 재확인
- 권고: 초기 검색은 3~5개로 제한, related-but-not-relevant 문서 검색 회피

### 본 프로젝트 반영

- **평가셋 설계**: 합성 평가 데이터셋의 문서 클래스를 이 4분류(+품질결함 클래스)로 구성
- **Hard-distractor 분석기** (예정): 코퍼스에서 "related" 문서(쿼리 유사도 높으나 정답 미포함)를
  식별하는 기능 — 단순 중복 탐지를 넘어서는 이 도구의 차별화 지점
- 주의: 위 수치는 **생성(reader) 단계** 실험 결과. 본 도구는 검색 단계를 다루므로
  "related 문서가 gold를 top-k에서 밀어내는" 검색 측 피해로 해석해 적용

---

## 2. Seven Failure Points When Engineering a RAG System (Barnett et al., CAIN 2024)

> Barnett, S. et al. "Seven Failure Points When Engineering a Retrieval Augmented
> Generation System." CAIN 2024. [arXiv:2401.05856](https://arxiv.org/abs/2401.05856)
> (사례연구 3건: Cognitive Reviewer / AI Tutor / BioASQ 4,017문서·1,000쿼리)

### 7가지 실패 지점 (원문 §5, 파이프라인 단계는 원문 Fig.1)

| FP | 원문 명칭 | 정의 | 단계 |
|---|---|---|---|
| FP1 | Missing Content | 답이 코퍼스에 없는데 시스템이 그럴듯하게 답변 | Index/DB |
| FP2 | Missed the Top Ranked Documents | 답이 문서에 있으나 top-K에 못 듦 | Retriever |
| FP3 | Not in Context — Consolidation strategy Limitations | 검색은 됐으나 컨텍스트 조립에서 탈락 | Consolidator |
| FP4 | Not Extracted | 컨텍스트에 있으나 LLM이 추출 실패 (노이즈·모순) | Reader |
| FP5 | Wrong Format | 표/리스트 등 형식 지시 무시 | Reader |
| FP6 | Incorrect Specificity | 너무 일반적이거나 너무 상세한 답 | Response |
| FP7 | Incomplete | 컨텍스트에 있는 정보를 일부 누락한 답 | Reader |

### 원문 교훈 (Table 2에서 본 프로젝트 관련 항목)

- **메타데이터(파일명·청크번호) 추가가 검색·추출(FP2, FP4) 개선** (AI Tutor)
- **작은 텍스트에서는 오픈소스 임베딩 모델이 상용 API와 동등** (BioASQ, AI Tutor)
  → 본 프로젝트의 로컬 우선(sentence-transformers) 전환을 지지
- **RAG 시스템은 지속적 캘리브레이션 필요** — 청크 크기·임베딩·검색 전략은
  설계 시점이 아닌 운영 중 검증됨 (FP2-7)
- 청킹: 휴리스틱 vs 시맨틱 비교는 **미해결 연구 과제**로 명시 (§6.1)

### 본 프로젝트 반영

- **실패 유형 자동 분류** (예정): 검색 벤치마크에서 쿼리별 실패를
  FP1(정답 문서가 코퍼스에 없음 — 예: 과도한 클리닝으로 gold 삭제)과
  FP2(코퍼스에 있으나 top-K 진입 실패)로 자동 분류해 리포트
- 청크에 소스 문서 ID 메타데이터 유지 (Table 2 교훈)

---

## 3. 보조 근거 (2차 자료 — 요약 수준 확인)

- **하이브리드 검색 + 리랭킹**: dense-only 대비 hybrid+rerank가 Recall@5 상대 +39%,
  리랭킹이 최대 단일 효과 컴포넌트라는 보고 다수
  ([벤치마크 예시](https://arxiv.org/pdf/2604.01733))
- **청크 크기**: 너무 작으면(≤128자) 맥락 부족, 너무 크면(≥1000) 관련성 희석 —
  중간 크기가 최적이라는 경험적 연구들
  ([사례](https://arxiv.org/html/2603.06976))
- **중복 제거**: MinHash/SimHash + LSH 기반 fuzzy dedup이 대규모 표준,
  임베딩 기반 탐지와 상보적
- 위 항목들은 해당 실험(하이브리드 검색·청킹 스윕) 구현 시 본 프로젝트의
  자체 벤치마크로 재검증한다 — 외부 수치를 그대로 신뢰하지 않음

---

## 4. 방법론 원칙

1. **측정 없이 개선 없다** — 모든 변경은 동일 평가셋의 before/after로 판정
2. **평가셋 크기가 판정을 좌우한다** — 쿼리 수가 작으면(≤5) ±1개 차이가
   큰 %변화로 보임. 부트스트랩 신뢰구간 병기
3. **공정한 ablation** — 클리닝 효과와 리랭킹 효과를 분리 (2×2 비교)
4. **부정적 결과도 기록** — 개선이 재현되지 않으면 그대로 문서화
