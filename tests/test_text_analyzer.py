"""Tests for TextAnalyzer quality-issue detection."""

import pytest

from src.scanner.text_analyzer import QualityIssueType, TextAnalyzer


@pytest.fixture
def analyzer():
    return TextAnalyzer(min_length=10, max_length=2000)


def issue_types(analyzer, text):
    result = analyzer.analyze([{"id": "d1", "text": text}])
    return {issue.issue_type for issue in result.quality_issues}


class TestBasicIssues:
    def test_too_short(self, analyzer):
        assert QualityIssueType.TOO_SHORT in issue_types(analyzer, "ㅎㅎ")

    def test_too_long(self, analyzer):
        assert QualityIssueType.TOO_LONG in issue_types(analyzer, "가나다라마 " * 500)

    def test_high_special_char_ratio(self, analyzer):
        assert QualityIssueType.HIGH_SPECIAL_CHAR_RATIO in issue_types(
            analyzer, "!!!@#$%^&*()@#$%^&*()"
        )

    def test_clean_document_has_no_issues(self, analyzer):
        text = "RAG는 검색 증강 생성 기술입니다. 외부 지식을 활용해 정확한 답변을 만듭니다."
        assert issue_types(analyzer, text) == set()


class TestMissingPunctuation:
    def test_long_text_without_punctuation_flagged(self, analyzer):
        text = "단어 " * 25  # 25 words, no sentence punctuation
        assert QualityIssueType.MISSING_PUNCTUATION in issue_types(analyzer, text.strip())

    def test_short_text_not_flagged(self, analyzer):
        assert QualityIssueType.MISSING_PUNCTUATION not in issue_types(
            analyzer, "짧은 제목 텍스트"
        )

    def test_punctuated_text_not_flagged(self, analyzer):
        text = ("문장입니다. " * 25).strip()
        assert QualityIssueType.MISSING_PUNCTUATION not in issue_types(analyzer, text)


class TestRepetitiveContent:
    def test_repeated_phrase_flagged(self, analyzer):
        text = ("오류가 발생했습니다 다시 시도하세요. " * 10).strip()
        assert QualityIssueType.REPETITIVE_CONTENT in issue_types(analyzer, text)

    def test_varied_text_not_flagged(self, analyzer):
        text = (
            "임베딩은 텍스트를 벡터로 변환합니다. 벡터 데이터베이스는 이를 저장합니다. "
            "검색 시스템은 코사인 유사도로 후보를 찾습니다. 리랭커는 순위를 다시 조정합니다. "
            "평가 지표로는 NDCG와 재현율이 자주 쓰입니다."
        )
        assert QualityIssueType.REPETITIVE_CONTENT not in issue_types(analyzer, text)

    def test_short_text_returns_none_ratio(self):
        assert TextAnalyzer._trigram_repetition_ratio("짧은 텍스트") is None


class TestStats:
    def test_document_stats_computed(self, analyzer):
        result = analyzer.analyze(
            [{"id": "d1", "text": "첫 번째 문장입니다. 두 번째 문장입니다."}]
        )
        stats = result.document_stats[0]
        assert stats.sentence_count == 2
        assert stats.word_count == 6
        assert 0.0 <= stats.quality_score <= 1.0
