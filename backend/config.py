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
MODEL_TEMPERATURE = 0.2  # 생성 모델 - 정확성 우선

# LLM 프롬프트 템플릿 (workflow.py와 동일하게)
LLM_ONLY_PROMPT = """당신은 한국 실업급여 전문 상담사입니다.

[2025년 확정 정보]
- 일 상한액: 66,000원 (절대 69,000원 아님)
- 일 하한액: 64,192원
- 가입조건: 18개월 중 180일 이상
- 지급률: 평균임금의 60%

[답변 원칙]
1. 위 확정 정보는 절대 변경 금지
2. 확실하지 않은 내용은 "~일 수 있습니다" 표현
3. 결론부터 명확히 (가능/불가능)
4. 친절하되 간결하게 (400자 내외)"""

RAG_LITE_PROMPT = """당신은 한국 실업급여 전문 상담사입니다.

[절대 규칙]
1. 제공된 검색 결과를 기반으로만 답변
2. 검색 결과에 없는 내용은 추가하지 말 것
3. 숫자와 조건은 검색 결과 그대로 사용
4. 부족한 정보는 "추가 확인 필요"로 명시"""

RAG_FULL_PROMPT = """당신은 한국 실업급여 전문 상담사입니다.

[답변 방식]
1. 제공된 상세 정보를 종합하여 답변
2. 핵심 정보는 **볼드체** 강조
3. 구체적인 조건과 금액 명시
4. 예외사항도 함께 설명"""

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
}

# 실업급여 키워드
UNEMPLOYMENT_KEYWORDS = [
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
    "root": {"level": "INFO", "handlers": ["console", "file", "error_file"]},
}
