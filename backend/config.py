# backend/config.py

import os
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

# 구조화된 실업급여 정보 (후처리용)
UNEMPLOYMENT_FACTS = {
    "amounts": {
        "daily_max": "66,000원",
        "daily_min": "64,192원",
        "rate": "평균임금의 60%",
        "min_wage_hourly": "10,030원",
    },
    "eligibility": {
        "insurance_period": "18개월 중 180일 이상",
        "resignation_type": "비자발적 이직 필수",
        "age_limit": "65세 미만",
        "claim_deadline": "이직 후 1년 이내",
    },
    "periods": {
        "under_30": "90~120일",
        "30_to_50": "120~180일",
        "50_plus": "120~210일",
        "disabled_50_plus": "120~270일",
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
}

# 중요 케이스 하드코딩 (RAG 실패시 폴백)
FALLBACK_ANSWERS = {
    "부정수급": """부정수급 처벌 (2025년 강화):
- 적발시 받은 금액의 5배 추징
- 형사처벌 + 명단 공개
- 향후 3년간 실업급여 제한""",
    "반복수급": """2025년 반복수급 감액 기준:
- 5년 이내 3회: 10% 감액
- 5년 이내 4회: 25% 감액
- 5년 이내 5회: 40% 감액
- 5년 이내 6회 이상: 50% 감액""",
    "금액_계산_금지": """실업급여 금액은 개인별 상황에 따라 다릅니다.
- 일 상한액: 66,000원
- 일 하한액: 64,192원
- 계산식: 퇴직 전 3개월 평균임금의 60%
실제 수급액은 고용센터에서 정확히 확인하세요.""",
}

# app.py가 찾는 CALCULATION_GUIDE 추가 (서버 오류 해결용)
CALCULATION_GUIDE = FALLBACK_ANSWERS["금액_계산_금지"]

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
    # 추가 키워드 (서버 오류 해결용)
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
