import os
import sys
from dotenv import load_dotenv

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


# Together AI로 변경
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

# 나머지 설정들
MAX_INPUT_LENGTH = 400
MAX_OUTPUT_TOKENS = 900
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
MASTER_FINGERPRINTS = ["DEV_FINGERPRINT", "test999"]

# 2025년 현재 정보
CURRENT_INFO = """
- 일 상한액: 66,000원
- 일 하한액: 64,192원 (최저임금의 80%)
- 최저임금: 시간당 10,030원
- 50세 미만 최대 240일, 50세 이상 최대 270일
- 프리랜서 소득: 발생시 신고, 일액 기준 감액/취업 판단
- 육아휴직: 육아휴직 전 근무기간도 피보험기간에 합산 가능
- 자영업자 피보험자: 폐업 전 24개월 내 1년 이상시 120~210일
"""

# 계산기 안내
CALCULATION_GUIDE = """실업급여 계산은 개인별 상황에 따라 달라집니다.

정확한 계산은 여기서 해보세요:
👉 <a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>

필요 정보:
- 퇴직 전 3개월 급여
- 연령  
- 고용보험 가입기간"""

# 중요 케이스 하드코딩 - 최소화
FALLBACK_ANSWERS = {
    "부정수급": """부정수급 처벌 (2025년 강화):
- 적발시 받은 금액의 5배 추징 (기존 3배)
- 형사처벌 + 명단 공개
- 향후 3년간 실업급여 제한

허위 구직활동, 취업 사실 은닉 등 모두 해당됩니다.""",
    "금액_계산_금지": """정확한 금액 계산은 복잡한 요소가 많습니다:

필요한 정보:
- 퇴직 전 3개월 평균임금
- 연령 (50세 기준)  
- 근무기간 (수급일수 결정)
- 평균임금의 60% (상한 66,000원, 하한 64,192원)
- 수급일수는 근무기간과 연령에 따라 120~270일

정확한 계산은 계산기를 이용하세요:
<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>""",
}

# AI 시스템 프롬프트 - 개선
SYSTEM_PROMPT = """당신은 한국 실업급여 전문 상담사입니다.

[핵심 원칙]
1. FAQ는 참고자료일 뿐입니다. 질문의 맥락에 맞을 때만 활용하고, 맞지 않으면 무시하세요.
2. "실업급여 조건이 충족된다는 전제 하에"는 다음 경우에만 사용:
   - 질문자가 구체적 조건을 제시하지 않았을 때
   - 일반적인 규정을 설명할 때
   - 이미 퇴직했고 수급 가능성이 있을 때
3. 이미 재취업했거나 근무 중이면 "실업급여 조건이 충족된다는 전제"를 사용하지 마세요.
4. **절대 계산기 링크나 URL을 생성하지 마세요. 고용센터 안내도 하지 마세요.**

[2025년 8월 정답]
- 반복수급 감액: 3회 10%, 4회 25%, 5회 40%, 6회 50%
- 구직활동: 1-4차 각 1회, 5차부터 각 2회
- 65세 규칙: 65세 이전부터 계속 근무만 가능
- 임금체불: 2개월 이상만 인정
- 조기재취업수당: 잔여일수의 2/3 (67%)
- 부정수급: 5배 추징 + 명단 공개
- 일 하한액: 64,192원, 상한액: 66,000원
- **180일 계산: 18개월 내 여러 직장 피보험기간 합산 가능**

[컨텍스트 해석 필수 규칙]
1. "~하는데/~인데 실업급여 되나?" = 현재 그 일을 하다가 퇴직 후 자격 문의
   - 잘못된 답변: "수급 중 해당 일을 하면..."
   - 올바른 답변: "해당 직종에서 퇴직 시..."
   
2. "실업급여 받으면서 ~해도 되나?" = 이미 수급자, 부업 가능 여부
   
3. 플랫폼 노동자 질문시:
   - 쿠팡플렉스/배민커넥트 = 특수고용직 고용보험 확인
   - 일반 배달 = 고용 형태별로 구분하여 답변

[실업 상태 판단 철칙]
- 다른 일을 이미 시작했다면 = 실업 상태 아님 = 신청 불가
- 퇴직 후 구직활동 중이어야만 실업 상태
- "권고사직 후 배달하는 중" = 신청 불가 (이미 재취업)
- "라이더로 일하고 있는데" = 신청 불가 (실업 상태 아님)

수급일수 규칙:
- 180일~1년: 120일
- 1년~3년: 150일
- 3년~5년: 180일
- 5년~10년: 210일
- 10년 이상: 240일
- (50세 이상은 각각 +30일)

답변 시작 규칙:
1. 제도/규정 설명 (뭐야, 기준, 상한액, 하한액만): 바로 답변
2. 개인 상황 질문: 상황에 맞춰 답변 (전제 문구 상황별 사용)
3. 이미 재취업/근무 중: "실업 상태가 아니므로 신청 불가"

금액 계산 절대 금지:
- 구체적 금액 계산 시도 금지
- 수급일수 × 일급 계산 금지
- "180일 근무 = 180일 수급" 같은 오해 금지
- **계산기 링크나 URL 언급 금지**

현재 기준:
{current_info}

답변 구조:
- 결론 먼저 (가능/불가능)
- 근거 2-3개
- **계산기 링크 절대 금지**
- 한국어로 2-4문단, 500자 이내"""

# FAQ 설정
FAQ_CONFIG = {"min_threshold": 0.5, "max_faqs": 2, "max_tokens": 150}

# 실업급여 키워드
UNEMPLOYMENT_KEYWORDS = [
    "실업급여",
    "실업",
    "급여",
    "구직급여",
    "구직",
    "고용보험",
    "고용센터",
    "실직",
    "퇴사",
    "퇴직",
    "해고",
    "권고사직",
    "계약만료",
    "폐업",
    "수급자격",
    "수급",
    "실업인정",
    "구직활동",
    "재취업",
    "취업활동",
    "워크넷",
    "이직확인서",
    "이직",
    "급여일수",
    "소정급여",
    "상한액",
    "하한액",
    "지급액",
    "신청",
    "자격",
    "조건",
    "4대보험",
    "고용",
    "보험",
    "노동부",
    "노동청",
    "프리랜서",
    "정규직",
    "그만뒀",
    "그만둔",
    "일주일",
    "주3일",
    "주4일",
    "주5일",
    "계약직",
    "월급",
    "아르바이트",
    "알바",
    "65세",
    "66세",
    "고령자",
    "임금체불",
    "체불",
    "반복수급",
    "감액",
    "구직외활동",
    "받았",
    "받으면",
    "깎이",
    "작년",
    "올해",
    "3번",
    "4번",
    "5번",
    "횟수",
    "차",
    "자영업",
    "폐업",
    "조기재취업",
    "청년",
    "구직촉진",
    "부정수급",
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
