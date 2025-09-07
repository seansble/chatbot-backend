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

# CORS 설정 - 개발중에는 모두 허용
ALLOWED_ORIGINS = [
    "https://sudanghelp.co.kr",  # 실제 서비스
    "https://www.sudanghelp.co.kr",  # www 버전 추가
    "http://localhost:3000",  # 로컬 개발
    "http://127.0.0.1:5500",  # Live Server 추가
    "http://localhost:5000",  # 로컬 백엔드 테스트
    "http://localhost:8000",  # Python 서버 추가
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
REDIS_ENABLED = False  # Redis 사용 안 함 (메모리 사용)
NEW_USER_LIMIT = 1  # 신규 사용자 일일 제한
REGULAR_USER_LIMIT = 3  # 기존 사용자 일일 제한
TOKEN_COOKIE_NAME = "user_token"
TOKEN_MAX_AGE = 86400 * 30  # 30일

# 입력 제한
MAX_INPUT_LENGTH = 400
MAX_OUTPUT_TOKENS = 900
MASTER_FINGERPRINTS = ["DEV_FINGERPRINT", "test999"]

# 2025년 현재 정보
CURRENT_INFO = """
- 일 상한액: 66,000원 / 하한액: 64,192원
- 최저임금: 시간당 10,030원
- 반복수급 감액: 3회 10%, 4회 25%, 5회 40%, 6회 50%
"""

# 계산기 안내
CALCULATION_GUIDE = """실업급여 계산은 개인별 상황에 따라 달라집니다.

정확한 계산은 여기서 해보세요:
👉 실업급여 계산기 바로가기

필요 정보:
- 퇴직 전 3개월 급여
- 연령  
- 고용보험 가입기간"""

# 중요 케이스 하드코딩 (RAG 실패시 폴백)
FALLBACK_ANSWERS = {
    "부정수급": """부정수급 처벌 (2025년 강화):
- 적발시 받은 금액의 5배 추징
- 형사처벌 + 명단 공개
- 향후 3년간 실업급여 제한""",
    "금액_계산_금지": """정확한 금액 계산은 복잡합니다:
- 퇴직 전 3개월 평균임금
- 연령과 근무기간에 따라 120~270일
- 평균임금의 60% (상한 66,000원, 하한 64,192원)

정확한 계산은 계산기를 이용하세요.""",
    "반복수급_감액": """2025년 반복수급 감액 기준:
- 5년 이내 3회: 10% 감액
- 5년 이내 4회: 25% 감액
- 5년 이내 5회: 40% 감액
- 5년 이내 6회 이상: 50% 감액

※ 2025년부터 강화된 기준 적용""",
}

# AI 시스템 프롬프트
SYSTEM_PROMPT = """당신은 한국 실업급여 전문 상담사입니다.

[핵심 역할]
RAG 시스템이 제공하는 정보를 바탕으로 정확하고 간결한 답변을 제공합니다.

[답변 원칙]
1. RAG 검색 결과를 우선적으로 활용
2. 실업 상태 = 퇴직 후 구직활동 중 (이미 재취업했다면 신청 불가)
3. 구체적 금액 계산 금지 → 계산기 안내로 대체
4. 결론 먼저, 근거는 2-3개만

[2025년 핵심 정보]
{current_info}

[답변 형식]
- 결론 제시 (가능/불가능/조건부 가능)
- 핵심 근거 2-3개
- 필요시 계산기 안내
- 500자 이내, 한국어"""

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
