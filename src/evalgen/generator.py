"""
Controlled evaluation dataset generator.

Generates a synthetic Korean IT-helpdesk corpus where every document carries
a class label from the taxonomy of Cuconasu et al. (SIGIR 2024, "The Power
of Noise") plus data-quality defect classes. Because each document's class
and answer-bearing status are known, retrieval failures can be attributed
to specific data-quality causes — which a real scraped corpus cannot do.

Document classes:
    gold          — contains the exact answer fact for its topic's queries
    relevant      — paraphrase of the gold fact (also answer-bearing)
    related       — same entity/topic but does NOT contain the answer fact
                    (the "hard distractor" class shown to hurt RAG the most)
    irrelevant    — unrelated-domain text
    exact_dup     — verbatim copy of a gold document
    near_dup      — lightly perturbed copy of a gold document (answer-bearing)
    low_quality   — too short / special chars / repetitive boilerplate

Everything is deterministic given a seed. No external APIs.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class DocClass(str, Enum):
    GOLD = "gold"
    RELEVANT = "relevant"
    RELATED = "related"
    IRRELEVANT = "irrelevant"
    EXACT_DUP = "exact_dup"
    NEAR_DUP = "near_dup"
    LOW_QUALITY = "low_quality"


# 답변 가능한(정답 포함) 클래스 — ground truth의 relevant_doc_ids 구성
ANSWER_BEARING = {DocClass.GOLD, DocClass.RELEVANT, DocClass.EXACT_DUP, DocClass.NEAR_DUP}


# ---------------------------------------------------------------------------
# 콘텐츠 뱅크 (가상의 IT 서비스 헬프데스크 도메인)
# ---------------------------------------------------------------------------

SERVICES = [
    "클라우드포트", "데이터브릿지", "메일허브", "독스페이스", "파이프라인X",
    "시큐어게이트", "로그스트림", "캐시타워", "알림봇", "폼빌더",
    "스토리지원", "서치엔진K", "비전API", "챗커넥트", "일정마스터",
    "결제링크", "지도서비스", "번역엔진", "음성노트", "화면공유",
    "백업센터", "모니터링허브", "테스트러너", "배포매니저", "코드리뷰어",
    "이미지팩토리", "동영상클라우드", "설문도구", "리포트빌더", "대시보드프로",
    "워크플로우", "티켓데스크", "지식베이스", "인증센터", "프록시게이트",
    "큐매니저", "이벤트버스", "피처플래그", "실험플랫폼", "메트릭수집기",
]

# 속성: (속성명, 질문 템플릿, 값 생성기, gold 템플릿 변형들, 패러프레이즈 템플릿들,
#         속성 정렬 hard distractor 템플릿들 — 같은 속성 주제를 다루지만 정답(값) 없음)
ATTRIBUTES = [
    (
        "api_limit",
        "{svc}의 API 요청 한도는 어떻게 되나요?",
        lambda rng: f"분당 {rng.choice([60, 100, 120, 300, 600, 1000])}회",
        [
            "{svc}의 API 요청 한도는 {val}입니다. 한도를 초과하면 429 응답 코드가 반환됩니다.",
            "요금제와 무관하게 {svc} API 호출은 {val}로 제한됩니다. 초과 요청은 큐에 쌓이지 않고 즉시 거부됩니다.",
            "개발자 문서 기준으로 {svc}의 호출 상한은 {val}이며, 엔터프라이즈 계약 시 상향 협의가 가능합니다.",
        ],
        [
            "{svc} API는 {val}까지 호출할 수 있으며, 초과 시 요청이 제한됩니다.",
            "정리하면 {svc}에서 허용되는 요청 빈도의 상한선은 {val}입니다.",
        ],
        [
            "{svc} API 사용량이 급증하면 대시보드에서 경고 알림을 설정할 수 있습니다. 사용량 통계는 시간 단위로 집계됩니다.",
            "{svc} API 키가 유출된 경우 즉시 콘솔에서 키를 회전하세요. 이전 키는 24시간 동안 유예 기간을 갖습니다.",
            "{svc}의 API 응답이 느릴 때는 페이지네이션 파라미터를 줄이고 필드 필터를 적용하는 것이 좋습니다.",
        ],
    ),
    (
        "storage",
        "{svc}의 기본 저장 용량은 얼마인가요?",
        lambda rng: f"{rng.choice([5, 10, 20, 50, 100, 200])}GB",
        [
            "{svc}의 기본 요금제는 {val}의 저장 공간을 제공합니다. 추가 용량은 관리 콘솔에서 구매할 수 있습니다.",
            "신규 가입 시 {svc} 계정에는 {val} 용량이 할당됩니다. 팀 플랜은 구성원 수에 비례해 늘어납니다.",
            "저장소 정책 문서에 따르면 {svc}의 무료 제공 용량은 {val}입니다.",
        ],
        [
            "{svc}를 처음 구독하면 {val} 저장소가 기본으로 포함되어 있습니다.",
            "{svc} 기본 플랜에 포함된 디스크 공간은 {val} 규모입니다.",
        ],
        [
            "{svc}의 저장 공간이 부족해지면 오래된 파일을 아카이브 계층으로 옮겨 비용을 줄일 수 있습니다.",
            "{svc} 저장소 사용량 그래프는 관리 콘솔의 통계 탭에서 확인할 수 있으며 매일 갱신됩니다.",
            "{svc}에서 삭제한 파일은 휴지통에 30일간 보관된 후 영구 삭제되며, 휴지통 용량도 사용량에 포함됩니다.",
        ],
    ),
    (
        "sla",
        "{svc}의 SLA 가동률 보장은 몇 퍼센트인가요?",
        lambda rng: f"{rng.choice(['99.5%', '99.9%', '99.95%', '99.99%'])}",
        [
            "{svc}는 월간 {val}의 가동률을 SLA로 보장합니다. 미달 시 크레딧이 지급됩니다.",
            "서비스 약관에 명시된 {svc}의 가용성 보장 수준은 {val}입니다.",
            "{svc} 엔터프라이즈 계약의 핵심 조항은 {val} 가동률 보장이며, 분기별로 준수 보고서가 발행됩니다.",
        ],
        [
            "{svc}의 서비스 수준 협약은 가동률 {val}를 기준으로 하며, 위반 시 보상 정책이 적용됩니다.",
            "요약하자면 {svc}가 계약상 약속하는 업타임은 {val}입니다.",
        ],
        [
            "{svc}의 실시간 가동 상태와 과거 장애 이력은 상태 페이지에서 누구나 확인할 수 있습니다.",
            "{svc}에 장애가 발생하면 SLA 크레딧 신청서를 30일 이내에 제출해야 보상을 받을 수 있습니다.",
            "{svc}의 정기 점검은 매월 둘째 주 일요일 새벽에 진행되며, 점검 시간은 SLA 계산에서 제외됩니다.",
        ],
    ),
    (
        "session",
        "{svc}의 세션 만료 시간은 얼마나 되나요?",
        lambda rng: f"{rng.choice([15, 30, 60, 120, 720])}분",
        [
            "{svc}의 로그인 세션은 {val} 동안 유지되며, 만료 후에는 재인증이 필요합니다.",
            "보안 정책상 {svc}의 세션 유효 시간은 {val}로 설정되어 있습니다.",
            "{svc}에 로그인하면 발급되는 토큰의 수명은 {val}이며 연장은 불가능합니다.",
        ],
        [
            "{svc}에서 비활성 상태가 {val}을 넘으면 세션이 자동으로 종료됩니다.",
            "다시 말해 {svc} 접속 후 {val}이 지나면 로그인이 풀립니다.",
        ],
        [
            "{svc}에서 로그아웃이 자주 발생한다면 브라우저의 쿠키 차단 설정을 먼저 확인해 보세요.",
            "{svc}는 동시 접속 기기 수를 제한하며, 새 기기에서 로그인하면 가장 오래된 세션이 종료됩니다.",
            "{svc}의 2단계 인증을 켜면 새 위치에서의 로그인 시 추가 확인 코드가 요구됩니다.",
        ],
    ),
    (
        "filesize",
        "{svc}에서 업로드 가능한 최대 파일 크기는?",
        lambda rng: f"{rng.choice([25, 50, 100, 500, 1024, 2048])}MB",
        [
            "{svc}의 단일 파일 업로드 제한은 {val}입니다. 대용량 파일은 분할 업로드 API를 사용하세요.",
            "업로드 정책상 {svc}에서 한 번에 올릴 수 있는 파일은 {val}까지입니다.",
            "{svc} 헬프 문서에 따르면 파일 하나의 크기 상한은 {val}로 정해져 있습니다.",
        ],
        [
            "{svc}에 올릴 수 있는 파일 하나의 최대 크기는 {val}로 설정되어 있습니다.",
            "간단히 말해 {svc}의 파일 업로드 한도는 {val}입니다.",
        ],
        [
            "{svc}에서 업로드가 중간에 실패하면 이어올리기 기능으로 중단 지점부터 재시도할 수 있습니다.",
            "{svc}는 업로드 시 바이러스 검사를 수행하며, 검사 중인 파일은 다운로드가 잠시 제한됩니다.",
            "{svc}에서 지원하는 파일 형식 목록은 도움말 센터에서 확인할 수 있으며 실행 파일은 차단됩니다.",
        ],
    ),
    (
        "backup",
        "{svc}의 백업 주기는 어떻게 되나요?",
        lambda rng: rng.choice(["매시간", "6시간마다", "매일 자정", "12시간마다", "주 1회"]),
        [
            "{svc}의 데이터는 {val} 자동으로 백업되며, 백업본은 30일간 보관됩니다.",
            "운영 정책상 {svc}는 {val} 전체 백업을 수행합니다.",
            "{svc}의 백업 스케줄은 {val}로 고정되어 있으며 변경하려면 지원팀 승인이 필요합니다.",
        ],
        [
            "{svc}는 {val} 스냅샷을 생성해 데이터를 보호합니다.",
            "요컨대 {svc}의 자동 백업은 {val} 실행됩니다.",
        ],
        [
            "{svc}의 백업본에서 특정 파일만 복원하려면 복원 마법사에서 경로를 지정하면 됩니다.",
            "{svc} 백업 데이터는 지리적으로 분리된 두 리전에 암호화되어 저장됩니다.",
            "{svc}에서 수동 백업을 실행하면 자동 백업 스케줄에는 영향을 주지 않습니다.",
        ],
    ),
]

# 서비스 도메인 필러 문장 (문서마다 어휘 다양성 부여, 정답과 무관)
FILLER_SENTENCES = [
    "자세한 내용은 공식 문서를 참고하세요.",
    "이 정책은 사전 공지 후 변경될 수 있습니다.",
    "문의 사항은 지원 포털을 통해 접수됩니다.",
    "최신 변경 사항은 릴리스 노트에 정리되어 있습니다.",
    "설정 변경은 관리자 권한이 필요합니다.",
    "적용까지 최대 몇 분이 걸릴 수 있습니다.",
    "",
    "",
]

IRRELEVANT_SENTENCES = [
    "김치찌개를 맛있게 끓이려면 잘 익은 김치와 돼지고기를 먼저 볶는 것이 중요합니다.",
    "제주 올레길 7코스는 외돌개에서 시작해 월평포구까지 이어지는 해안 길입니다.",
    "마라톤 완주를 위해서는 대회 8주 전부터 주간 주행 거리를 점진적으로 늘려야 합니다.",
    "커피 원두는 로스팅 후 2주 이내에 소비하는 것이 향미가 가장 좋습니다.",
    "실내 화분은 계절이 바뀔 때마다 물 주기 간격을 조정해 주어야 합니다.",
    "전기차 배터리는 급속 충전을 반복하면 수명이 단축될 수 있습니다.",
    "겨울철 등산은 일몰 시간이 빠르므로 하산 계획을 여유 있게 잡아야 합니다.",
    "홈베이킹에서 버터 온도는 반죽의 식감을 좌우하는 핵심 요소입니다.",
    "수영 자유형에서 호흡은 팔 동작 리듬과 맞추는 것이 중요합니다.",
    "반려견 산책은 하루 두 번, 각 30분 이상이 권장됩니다.",
]

LOW_QUALITY_DOCS = [
    "확인했습니다",
    "ㅇㅋ",
    "재부팅 해보세요",
    "!!!@#$%^&*()_+@#$%^&*()!!!",
    "ㅋㅋㅋㅋㅋㅋㅋ 넵넵",
    "해결됨",
]


@dataclass
class GeneratedDataset:
    """Documents + queries + per-document labels."""

    documents: List[Dict]         # {id, text, category}
    queries: List[Dict]           # {query_id, query, relevant_doc_ids: List[str]}
    labels: Dict[str, Dict]       # doc_id -> {doc_class, topic_id, answer_bearing}

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for meta in self.labels.values():
            counts[meta["doc_class"]] = counts.get(meta["doc_class"], 0) + 1
        return counts


class EvalDatasetGenerator:
    """
    Deterministic generator for the controlled evaluation corpus.

    Args:
        seed: RNG seed (same seed -> identical dataset)
        n_topics: number of (service, attribute) fact topics
        related_per_topic: hard distractors per topic (Power-of-Noise "related")
        n_irrelevant: unrelated-domain filler docs
        n_exact_dup / n_near_dup: duplicate injections (copies of gold docs)
        n_low_quality: junk documents
    """

    def __init__(
        self,
        seed: int = 42,
        n_topics: int = 40,
        related_per_topic: int = 2,
        n_irrelevant: int = 20,
        n_exact_dup: int = 15,
        n_near_dup: int = 15,
        n_low_quality: int = 12,
    ):
        self.rng = random.Random(seed)
        self.n_topics = n_topics
        self.related_per_topic = related_per_topic
        self.n_irrelevant = n_irrelevant
        self.n_exact_dup = n_exact_dup
        self.n_near_dup = n_near_dup
        self.n_low_quality = n_low_quality

    # -- 근사중복 변형: 조사/공백/문장부호 수준의 가벼운 교란 ------------------
    def _perturb(self, text: str) -> str:
        t = text
        swaps = [
            ("입니다.", "입니다 ."),
            ("있습니다.", "있습니다 !"),
            ("됩니다.", "됩니다 ."),
            ("는 ", "는  "),
            ("의 ", "의  "),
        ]
        self.rng.shuffle(swaps)
        applied = 0
        for old, new in swaps:
            if old in t and applied < 2:
                t = t.replace(old, new, 1)
                applied += 1
        # 문두 접두 추가(내용 불변)
        if self.rng.random() < 0.5:
            t = "참고: " + t
        return t

    def generate(self) -> GeneratedDataset:
        documents: List[Dict] = []
        queries: List[Dict] = []
        labels: Dict[str, Dict] = {}
        doc_seq = 0

        def add_doc(text: str, doc_class: DocClass, topic_id: str) -> str:
            nonlocal doc_seq
            doc_seq += 1
            doc_id = f"doc_{doc_seq:04d}"
            documents.append({"id": doc_id, "text": text, "category": doc_class.value})
            labels[doc_id] = {
                "doc_class": doc_class.value,
                "topic_id": topic_id,
                "answer_bearing": doc_class in ANSWER_BEARING,
            }
            return doc_id

        # 토픽 = (서비스, 속성) 조합. 서비스별로 서로 다른 속성을 뽑아 구성.
        combos: List[Tuple[str, tuple]] = []
        services = SERVICES.copy()
        self.rng.shuffle(services)
        for svc in services:
            attrs = self.rng.sample(ATTRIBUTES, k=min(2, len(ATTRIBUTES)))
            for attr in attrs:
                combos.append((svc, attr))
        self.rng.shuffle(combos)
        combos = combos[: self.n_topics]

        gold_ids: List[str] = []
        for t_idx, (svc, attr) in enumerate(combos):
            topic_id = f"topic_{t_idx:03d}"
            attr_name, q_tpl, val_fn, gold_tpls, para_tpls, related_tpls = attr
            val = val_fn(self.rng)

            def with_filler(text: str) -> str:
                filler = self.rng.choice(FILLER_SENTENCES)
                return f"{text} {filler}".strip()

            gold_text = with_filler(self.rng.choice(gold_tpls).format(svc=svc, val=val))
            para_text = with_filler(self.rng.choice(para_tpls).format(svc=svc, val=val))
            gold_id = add_doc(gold_text, DocClass.GOLD, topic_id)
            para_id = add_doc(para_text, DocClass.RELEVANT, topic_id)
            gold_ids.append(gold_id)

            answer_ids = [gold_id, para_id]

            # related(hard distractor): 같은 서비스·같은 속성 주제를 다루지만
            # 정답 값은 없는 문서 (Power of Noise의 "related" 정의)
            related_pool = related_tpls.copy()
            self.rng.shuffle(related_pool)
            for r in range(min(self.related_per_topic, len(related_pool))):
                add_doc(
                    with_filler(related_pool[r].format(svc=svc)),
                    DocClass.RELATED,
                    topic_id,
                )

            queries.append({
                "query_id": f"q_{t_idx:03d}",
                "query": q_tpl.format(svc=svc),
                "relevant_doc_ids": answer_ids,  # dup 주입 후 갱신됨
            })

        # 중복 주입: gold 문서의 사본 (answer-bearing → GT에 반영)
        dup_targets = self.rng.sample(gold_ids, k=min(self.n_exact_dup, len(gold_ids)))
        for gid in dup_targets:
            src = next(d for d in documents if d["id"] == gid)
            topic = labels[gid]["topic_id"]
            new_id = add_doc(src["text"], DocClass.EXACT_DUP, topic)
            q = queries[int(topic.split("_")[1])]
            q["relevant_doc_ids"].append(new_id)

        near_targets = self.rng.sample(gold_ids, k=min(self.n_near_dup, len(gold_ids)))
        for gid in near_targets:
            src = next(d for d in documents if d["id"] == gid)
            topic = labels[gid]["topic_id"]
            new_id = add_doc(self._perturb(src["text"]), DocClass.NEAR_DUP, topic)
            q = queries[int(topic.split("_")[1])]
            q["relevant_doc_ids"].append(new_id)

        # 무관 문서
        for i in range(self.n_irrelevant):
            base = IRRELEVANT_SENTENCES[i % len(IRRELEVANT_SENTENCES)]
            extra = self.rng.choice(IRRELEVANT_SENTENCES)
            add_doc(f"{base} {extra}", DocClass.IRRELEVANT, "none")

        # 저품질 문서 (짧음 / 특수문자 / 반복)
        for i in range(self.n_low_quality):
            if i % 3 == 2:
                text = ("문의하신 내용 확인 부탁드립니다. " * 8).strip()  # 반복
            else:
                text = LOW_QUALITY_DOCS[i % len(LOW_QUALITY_DOCS)]
            add_doc(text, DocClass.LOW_QUALITY, "none")

        return GeneratedDataset(documents=documents, queries=queries, labels=labels)
