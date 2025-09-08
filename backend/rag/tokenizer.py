# backend/rag/tokenizer.py
from typing import List
from kiwipiepy import Kiwi
import logging

logger = logging.getLogger(__name__)


class KiwiTokenizer:
    def __init__(self):
        """Kiwi 토크나이저 초기화"""
        try:
            self.kiwi = Kiwi(typos="basic", model_type="knlm")
            logger.info("✅ Kiwi 토크나이저 초기화 완료")

            # 실업급여 도메인 복합명사
            self.compound_nouns = {
                "육아휴직",
                "산전후휴가",
                "권고사직",
                "계약만료",
                "임금체불",
                "고용보험",
                "실업급여",
                "구직활동",
                "평균임금",
                "자진퇴사",
                "비자발적퇴사",
                "정당한사유",
                "체불확인서",
                "이직확인서",
                "수급자격",
                "소정급여일수",
            }

            # 실업급여 도메인 사용자 사전 추가
            self.add_user_dictionary()
        except Exception as e:
            logger.error(f"Kiwi 초기화 실패: {e}")
            self.kiwi = None

    def add_user_dictionary(self):
        """도메인 특화 용어 사전 추가"""
        if not self.kiwi:
            return

        user_words = [
            ("실업급여", "NNP", 10.0),
            ("구직활동", "NNP", 10.0),
            ("고용보험", "NNP", 10.0),
            ("수급자격", "NNP", 10.0),
            ("이직확인서", "NNP", 10.0),
            ("고용센터", "NNP", 10.0),
            ("자발적퇴사", "NNP", 10.0),
            ("비자발적퇴사", "NNP", 10.0),
            ("권고사직", "NNP", 10.0),
            ("계약만료", "NNP", 10.0),
            ("구직급여", "NNP", 10.0),
            ("취업촉진수당", "NNP", 10.0),
            ("조기재취업수당", "NNP", 10.0),
            ("광역구직활동비", "NNP", 10.0),
            ("이주비", "NNP", 10.0),
            ("직업능력개발수당", "NNP", 10.0),
            ("소정급여일수", "NNP", 10.0),
            ("수급기간", "NNP", 10.0),
            ("대기기간", "NNP", 10.0),
            ("실업인정", "NNP", 10.0),
            ("재취업활동", "NNP", 10.0),
            ("워크넷", "NNP", 10.0),
            ("고용24", "NNP", 10.0),
            ("중도해지", "NNP", 10.0),
            ("만기해지", "NNP", 10.0),
            ("피보험자격", "NNP", 10.0),
            ("육아휴직", "NNP", 10.0),  # 추가
            ("산전후휴가", "NNP", 10.0),  # 추가
            ("임금체불", "NNP", 10.0),  # 추가
            ("평균임금", "NNP", 10.0),  # 추가
            ("체불확인서", "NNP", 10.0),  # 추가
            ("정당한사유", "NNP", 10.0),  # 추가
        ]

        try:
            for word, pos, score in user_words:
                self.kiwi.add_user_word(word, pos, score)
            logger.info(f"✅ 사용자 사전 {len(user_words)}개 단어 추가")
        except Exception as e:
            logger.warning(f"사용자 사전 추가 실패: {e}")

    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰화 - 복합명사 처리 추가"""
        if not self.kiwi:
            return self._simple_fallback(text)

        try:
            # 1차 시도: 정상 토큰화
            result = self.kiwi.tokenize(text, normalize_coda=True)

            if not result:
                return self._simple_fallback(text)

            tokens = self._extract_tokens(result)

            # 복합명사 추가 (중복 제거)
            tokens_set = set(tokens)
            for compound in self.compound_nouns:
                if compound in text:
                    tokens_set.add(compound)

            tokens = list(tokens_set)

            # 품질 검증
            if self._check_quality(tokens):
                return tokens

            # 2차 시도: 필터링 강화
            tokens = self._extract_tokens_strict(result)

            # 복합명사 추가 (2차에도)
            tokens_set = set(tokens)
            for compound in self.compound_nouns:
                if compound in text:
                    tokens_set.add(compound)
            tokens = list(tokens_set)

            if self._check_quality(tokens):
                return tokens

            # 3차 시도: normalize_coda 끄고 재시도
            result = self.kiwi.tokenize(text, normalize_coda=False)
            tokens = self._extract_tokens_strict(result)

            # 복합명사 추가 (3차에도)
            tokens_set = set(tokens)
            for compound in self.compound_nouns:
                if compound in text:
                    tokens_set.add(compound)
            tokens = list(tokens_set)

            if tokens and len(tokens) > 0:
                return tokens

            # 최종 폴백
            return self._simple_fallback(text)

        except Exception as e:
            logger.error(f"토큰화 실패: {e}")
            return self._simple_fallback(text)

    def _extract_tokens(self, result: List) -> List[str]:
        """기본 토큰 추출"""
        tokens = []
        skip_next = False

        # 진짜 불필요한 조사만 정의
        UNNECESSARY_PARTICLES = {"은", "는", "이", "가", "을", "를", "의"}

        for i, token in enumerate(result):
            if skip_next:
                skip_next = False
                continue

            # 불필요한 조사만 제거, 나머지 조사는 보존
            if token.tag.startswith("J"):
                if token.form not in UNNECESSARY_PARTICLES:
                    tokens.append(token.form)
                continue

            # 어미(E), 접사(X) 제외, SF(구두점)만 제외
            if token.tag[0] in ["E", "X"] or token.tag == "SF":
                continue

            # 숫자+단위 복합 처리
            if token.tag == "SN":
                if i + 1 < len(result) and result[i + 1].tag == "NNB":
                    combined = token.form + result[i + 1].form
                    tokens.append(combined)
                    skip_next = True
                    continue
                else:
                    tokens.append(token.form)
                    continue

            # 일반 토큰 처리
            if (
                token.tag.startswith("N")
                or token.tag.startswith("V")
                or token.tag == "VA"
                or token.tag.startswith("M")
                or token.tag.startswith("SL")
            ):
                tokens.append(token.form)

        return tokens

    def _extract_tokens_strict(self, result: List) -> List[str]:
        """엄격한 토큰 추출 - 2글자 이상만"""
        tokens = []
        skip_next = False

        # 진짜 불필요한 조사만 정의
        UNNECESSARY_PARTICLES = {"은", "는", "이", "가", "을", "를", "의"}

        for i, token in enumerate(result):
            # 이전 토큰과 합쳐진 경우 스킵
            if skip_next:
                skip_next = False
                continue

            # 불필요한 조사만 제거, 나머지 조사는 보존
            if token.tag.startswith("J"):
                if token.form not in UNNECESSARY_PARTICLES:
                    if len(token.form) >= 2:  # 2글자 이상 조사만
                        tokens.append(token.form)
                continue

            # 어미(E), 접사(X) 제외, SF(구두점)만 제외
            if token.tag[0] in ["E", "X"] or token.tag == "SF":
                continue

            # 숫자+단위 복합 처리
            if token.tag == "SN":
                # 다음 토큰이 NNB(년,월,일,개월 등)이면 합치기
                if i + 1 < len(result) and result[i + 1].tag == "NNB":
                    combined = token.form + result[i + 1].form
                    tokens.append(combined)
                    skip_next = True
                    continue
                else:
                    # 숫자만 있는 경우도 포함
                    tokens.append(token.form)
                    continue

            # 길이 필터링 (숫자 제외)
            if len(token.form) < 2:
                continue

            # 명사(N), 동사(V), 형용사(VA), 수사(M), 외국어(SL) 추출
            if (
                token.tag.startswith("N")
                or token.tag.startswith("V")
                or token.tag == "VA"
                or token.tag.startswith("M")
                or token.tag.startswith("SL")
            ):
                tokens.append(token.form)

        return tokens

    def _check_quality(self, tokens: List[str]) -> bool:
        """토큰 품질 검증"""
        if not tokens:
            return False

        short_tokens = [t for t in tokens if len(t) < 2 and not t.isdigit()]
        ratio = len(short_tokens) / len(tokens)

        if ratio > 0.35:
            logger.debug(f"토큰 품질 미달 - 1글자 비율: {ratio:.1%}")
            return False

        return True

    def _simple_fallback(self, text: str) -> List[str]:
        """간단한 폴백 토크나이저"""
        tokens = []
        words = text.split()

        # 조사/어미 제거
        particles = [
            "에서",
            "으로",
            "에게",
            "부터",
            "까지",
            "을",
            "를",
            "이",
            "가",
            "은",
            "는",
        ]
        endings = ["습니다", "입니다", "나요", "어요", "는데", "었고", "았고"]

        for word in words:
            cleaned = word.strip("?.,!")

            # 조사 제거
            for p in particles:
                if cleaned.endswith(p) and len(cleaned) > len(p) + 1:
                    cleaned = cleaned[: -len(p)]
                    break

            # 어미 제거
            for e in endings:
                if cleaned.endswith(e) and len(cleaned) > len(e) + 1:
                    cleaned = cleaned[: -len(e)]
                    break

            if cleaned and len(cleaned) > 0:
                tokens.append(cleaned)

        return tokens

    def expand_query(self, tokens: List[str]) -> List[str]:
        """쿼리 토큰에만 동의어 확장 적용 - 순서 유지"""
        synonyms = {
            "퇴사": ["사직", "퇴직", "그만두다"],
            "해고": ["해촉", "권고사직", "정리해고"],
            "가능": ["되다", "수급", "자격"],
            "급여": ["수당", "금액", "돈"],
            "조건": ["자격", "요건", "기준"],
        }

        expanded = []
        seen = set()

        # 원본 순서 유지하면서 확장
        for token in tokens:
            if token not in seen:
                expanded.append(token)
                seen.add(token)

            # 동의어 추가
            if token in synonyms:
                for syn in synonyms[token]:
                    if syn not in seen:
                        expanded.append(syn)
                        seen.add(syn)

        return expanded
