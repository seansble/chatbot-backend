# backend/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Railway는 환경변수 자동 주입, 로컬 개발시만 .env 필요
if os.path.exists(".env"):
    load_dotenv()


def clean_key(s):
    """모든 비가시 문자 제거"""
    if not s:
        return ""
    return (
        s.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\n", "")
        .replace("\r", "")
        .strip()
    )


# Together AI 키
raw_key = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_API_KEY = clean_key(raw_key)

# 키 검증
if not TOGETHER_API_KEY or len(TOGETHER_API_KEY) < 20:
    print(f"WARNING: Invalid TOGETHER API key, using dummy")
    TOGETHER_API_KEY = "dummy-key-for-testing"

print(f"Together AI Key loaded successfully")

# API 설정
API_BASE_URL = "https://api.together.xyz/v1"
OPENROUTER_API_KEY = TOGETHER_API_KEY  # 기존 변수명 유지
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
MODEL = MODEL_NAME  # workflow.py가 config.MODEL을 참조하므로
EVAL_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

# Temperature 설정
EVAL_TEMPERATURE = 0.1  # 평가 모델 - 일관성 중요
MODEL_TEMPERATURE = 0.1  # 생성 모델 - 정확성 최우선 (0.2 → 0.1)

# 변수 추출용 프롬프트 템플릿 (새로 추가)
VARIABLE_EXTRACTION_PROMPT = """질문을 분석하여 실업급여 계산에 필요한 변수를 추출하세요.

[추출 대상]
- 나이 또는 생년월일
- 월급 (세전)
- 근무 이력 (각 직장별)
- 퇴사 사유
- 반복수급 여부
- 장애 여부
- 특수 상황 (육아휴직, 병역 등)

[분류 기준]
퇴사사유:
- 비자발적: 권고사직, 해고, 계약만료, 회사폐업
- 정당한자발적: 임금체불(2개월↑), 괴롭힘, 질병(4주↑)
- 단순자발적: 이직, 개인사정

고용형태:
- 정규직/계약직: 고용보험 100%
- 프리랜서/특고: 월 80만원 이상
- 일용직: 월 15일 이상
- 예술인: 특별기준

JSON 형식으로만 응답하세요."""

# 구조화된 실업급여 정보 (확장 버전)
UNEMPLOYMENT_FACTS = {
    "amounts": {
        "daily_max": "66,000원",
        "daily_min": "64,192원",
        "rate": "평균임금의 60%",
        "min_wage_hourly": "10,030원",
        "youth_bonus": "10% 추가 (18-34세)",
    },
    "eligibility": {
        "insurance_period": "18개월 중 180일 이상",
        "resignation_type": "비자발적 이직 필수 (정당한 자발적 포함)",
        "age_limit": "65세 미만",
        "claim_deadline": "이직 후 1년 이내",
        "youth_exception": "청년(18-34세)은 3개월 이상 가능",
        "disability_exception": "장애인은 조건 완화",
    },
    "periods": {
        "under_30": "90~180일",
        "30_to_50": "120~210일",
        "50_plus": "120~240일",
        "disabled_50_plus": "120~270일",
        "long_term": "20년 이상 근무 시 30일 추가",
    },
    "repetition": {
        "info": "5년 이내 반복수급 시 감액",
        "3rd": "감액 가능 (심사 결정)",
        "4th": "감액 가능 (심사 결정)",
        "5th": "감액 가능 (심사 결정)",
        "detailed": "정확한 감액률은 고용센터 심사 필요",
    },
    "special_cases": {
        "임금체불": "2개월 이상 체불 시 정당한 자발적 퇴사",
        "직장내괴롭힘": "증거 있을 시 정당한 자발적 퇴사",
        "질병": "4주 이상 치료 필요 시 정당한 자발적 퇴사",
        "육아": "육아휴직 중 이직 시 특별 심사",
        "통근곤란": "편도 3시간 이상 시 정당한 사유",
    },
}

# 자주 틀리는 정보 (후처리용)
COMMON_MISTAKES = {
    "69,000원": "66,000원",
    "6만 9천원": "6만 6천원",
    "68,000원": "66,000원",
    "24개월 중 18개월": "18개월 중 180일",
    "18개월 중 6개월": "18개월 중 180일",
    "14일 이내": "1년 이내",
    "평균임금의 50%": "평균임금의 60%",
    "18개월 모두 가입": "18개월 중 180일만 가입하면 됨",
}

# 중요 케이스 폴백 (개선된 버전)
FALLBACK_ANSWERS = {
    "부정수급": """부정수급은 엄격히 처벌됩니다:
- 부정수급액의 5배 추징
- 형사처벌 가능
- 향후 수급 제한
정당한 수급권을 보호하기 위한 제도이니 정직하게 신고하세요.""",
    "반복수급_감액": """반복수급 감액은 5년 이내 수급 횟수에 따라 결정됩니다.
구체적인 감액률은 개인별 상황에 따라 고용센터에서 심사합니다.
자세한 내용은 고용센터 1350으로 문의하세요.""",
    "금액_계산_금지": """실업급여 금액은 개인별 상황에 따라 다릅니다.
- 일 상한액: 66,000원
- 일 하한액: 64,192원
- 계산식: 퇴직 전 3개월 평균임금의 60%
- 청년(18-34세): 10% 추가 지급
실제 수급액은 고용센터에서 정확히 확인하세요.""",
}

# app.py가 찾는 CALCULATION_GUIDE 추가 (서버 오류 해결용)
CALCULATION_GUIDE = FALLBACK_ANSWERS["금액_계산_금지"]

# 복잡한 질문 패턴 (새로 추가)
COMPLEX_QUERY_PATTERNS = [
    r"\d+\s*개월.*\d+\s*개월",  # 여러 기간
    r"[A-Z]회사.*[A-Z]회사",  # 여러 회사
    r"프리랜서.*정규직",  # 복합 고용
    r"육아휴직|병역",  # 특수 상황
    r"이전에.*그전에",  # 복수 이력
]

# CORS 설정
ALLOWED_ORIGINS = [
    "https://sudanghelp.co.kr",
    "https://www.sudanghelp.co.kr",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5000",
    "http://localhost:8000",
]

# Railway 앱 도메인도 자동 추가
RAILWAY_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN")
if RAILWAY_DOMAIN:
    ALLOWED_ORIGINS.append(f"https://{RAILWAY_DOMAIN}")

print(f"CORS configured for: {ALLOWED_ORIGINS}")

# Railway 환경 체크
ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
if ENVIRONMENT:
    print(f"Running in {ENVIRONMENT} mode")

# 토큰 관련 설정 (일일 3회 제한 필수)
REDIS_ENABLED = False
DAILY_LIMIT = 3
TOKEN_COOKIE_NAME = "user_token"
TOKEN_MAX_AGE = 86400 * 30  # 30일

# 입력 제한
MAX_INPUT_LENGTH = 400
MAX_OUTPUT_TOKENS = 500
MASTER_FINGERPRINTS = ["DEV_FINGERPRINT", "test999"]

# 실업급여 키워드 (확장된 목록)
UNEMPLOYMENT_KEYWORDS = [
    # 기본 키워드
    "실업급여",
    "실업",
    "급여",
    "구직급여",
    "고용보험",
    "퇴사",
    "퇴직",
    "해고",
    "권고사직",
    "계약만료",
    "수급",
    "실업인정",
    "구직활동",
    "재취업",
    "워크넷",
    "이직확인서",
    "상한액",
    "하한액",
    "프리랜서",
    "정규직",
    "계약직",
    "아르바이트",
    "65세",
    "임금체불",
    "반복수급",
    "자영업",
    "폐업",
    "부정수급",
    "조기재취업",
    "구직촉진수당",
    # 추가 키워드
    "180일",
    "6개월",
    "7개월",
    "8개월",
    "9개월",
    "10개월",
    "11개월",
    "12개월",
    "주5일",
    "주6일",
    "근무",
    "일했",
    "다녔",
    "다니다",
    "월급",
    "연봉",
    "시급",
    "일당",
    "급여액",
    "얼마",
    "금액",
    "받을",
    "받나",
    "받아",
    "가능",
    "되나",
    "되는지",
    "계산",
    "예상",
    "대략",
    "정도",
    "250만원",
    "300만원",
    "200만원",
    "100만원",
    "권고사직당했",
    "짤렸",
    "잘렸",
    "그만뒀",
    "때려치",
    "나이",
    "조건",
    "자격",
    "요건",
    "기준",
    "청년",
    "장애",
    "체불",
    "못받",
    "미지급",
]

# 로깅 설정
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "maxBytes": 10485760,
            "backupCount": 5,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },  # file 핸들러 제거 (Railway에서 문제 방지)
}

# LLM 검증 설정 (확장 버전)
LLM_VERIFICATION_ENABLED = True  # True로 변경
LLM_VERIFICATION_THRESHOLD = 0.75
LLM_CACHE_SIZE = 1000
LLM_VERIFICATION_TIMEOUT = 3

# RAG 검색 파라미터 개선
RAG_SEARCH_TOP_K = 10  # 3 → 10으로 증가
RAG_HIGH_CONFIDENCE_THRESHOLD = 0.3  # 0.12 → 0.3으로 상향
RAG_MIN_HIT_RATIO = 0.4  # 최소 히트 비율 추가
