import os
import sys
from dotenv import load_dotenv

# .env 파일이 있으면 로드 (로컬 개발용)
load_dotenv()

# 환경변수 디버깅 부분을 찾아서 수정
print("=== ENVIRONMENT VARIABLES DEBUG ===")
for key in sorted(os.environ.keys()):
    if 'API' in key.upper() or 'KEY' in key.upper():
        print(f"{key}: [HIDDEN]")  # 값을 완전히 숨김

# API Key 가져오기 (환경변수에서)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# 가능한 모든 변수명 시도
possible_keys = [
    'OPENROUTER_API_KEY',
    'openrouter_api_key',
    'OPENROUTER_API',
    'openrouter_api'
]

for key in possible_keys:
    value = os.getenv(key, "").strip()
    if value:
        OPENROUTER_API_KEY = value
        print(f"Found API key in: {key}")
        break

# 그래도 못 찾으면 수동으로 검색
if not OPENROUTER_API_KEY:
    for key, value in os.environ.items():
        if 'OPENROUTER' in key.upper():
            OPENROUTER_API_KEY = value.strip()
            print(f"Found API key in environ: {key}")
            break
        
# API Key 검증 부분을 이렇게 수정
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment")
    print(f"Available env vars (first 20): {list(os.environ.keys())[:20]}")
    
    # Railway에서 실행 중이면 더미 키 사용
    if os.getenv("RAILWAY_ENVIRONMENT"):
        print("ERROR: API key must be set in Railway Variables")
        sys.exit(1)  # 더미 키로 실행 방지
    else:
        # 로컬에서는 종료
        print("ERROR: Set OPENROUTER_API_KEY in environment")
        sys.exit(1)
        
# API 설정
API_PROVIDER = "openrouter"
MODEL_NAME = "qwen/qwen3-235b-a22b-instruct-2507"
API_BASE_URL = "https://openrouter.ai/api/v1"

# 토큰 제한 설정 (증가)
MAX_INPUT_LENGTH = 400  # 300 → 400
MAX_OUTPUT_TOKENS = 900  # 800 → 900

# 개발 환경 설정
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
MASTER_FINGERPRINTS = ["DEV_FINGERPRINT", "test999"]

# ===== 2025년 현재 정보 (정확한 수치) =====
CURRENT_INFO = """
- 일 상한액: 66,000원
- 일 하한액: 64,192원 (최저임금의 80%)
- 최저임금: 시간당 10,030원
- 50세 미만 최대 240일, 50세 이상 최대 270일
- 프리랜서 소득: 발생시 신고, 일액 기준 감액/취업 판단
- 육아휴직: 육아휴직 전 근무기간도 피보험기간에 합산 가능
- 자영업자 피보험자: 폐업 전 24개월 내 1년 이상시 120~210일
"""

# ===== 계산기 안내 =====
CALCULATION_GUIDE = """실업급여 계산은 개인별 상황에 따라 달라집니다.

정확한 계산은 여기서 해보세요:
👉 <a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>

필요 정보:
- 퇴직 전 3개월 급여
- 연령  
- 고용보험 가입기간"""

# ===== 중요 케이스 하드코딩 (2025년 버전) =====
FALLBACK_ANSWERS = {
    "권고사직_사직서": """네, 사직서를 작성했어도 권고사직으로 인정받을 수 있습니다.

준비하실 서류:
- 권고받은 증거 (녹음, 문자, 이메일 등)
- 동료 증언서
- 퇴사 경위서 (자필)

고용센터에서 이직사유 심사를 신청하시면 됩니다. 
실제 승인률은 약 70% 정도이니 증거자료를 잘 준비하세요.

<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>

문의: 고용노동부 상담센터 1350""",

    "자진퇴사": """자진퇴사는 원칙적으로 실업급여를 받을 수 없습니다.

예외적으로 인정되는 경우:
- 임금체불 2개월 이상 (체불확인서 필요)
- 최저임금 미달 (급여명세서 필요)
- 직장내 괴롭힘/성희롱 (증빙 필수)
- 통근 왕복 3시간 이상 (주소지 증명)
- 질병/부상 (4주 이상 진단서)

각 사유별로 증빙서류가 꼭 필요합니다.

<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>

자세한 상담: 고용노동부 상담센터 1350""",

    "반복수급_감액": """2025년 반복수급자 감액 기준:
- 3회째: 10% 감액
- 4회째: 25% 감액  
- 5회째: 40% 감액
- 6회째 이상: 50% 감액
※ 대기기간 연장 가능
※ 2025년 이전 수급 이력은 제외
※ 일용직, 정당한 이직 사유는 횟수에서 제외

<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>""",

    "구직활동_횟수": """구직활동 요건:
- 1차~4차: 각 1회
- 5차부터: 각 2회 (최소 1회는 실제 구직활동)

인정되는 활동:
- 입사지원, 면접, 직업훈련, 자격증 취득
- 같은 날 여러 활동해도 1개만 인정
- 2025년부터 온라인 실업인정 원칙

워크넷 이용시 자동으로 인정되니 가장 편리합니다.""",

    "자영업자": """자영업자도 고용보험 가입시 실업급여 가능합니다.

요건: 폐업 전 24개월 내 피보험기간 1년 이상
급여일수: 120~210일 (일반 근로자와 다름)
필요서류: 폐업 증명서류, 매출 감소 증빙

<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>""",

    "조기재취업수당": """조기재취업수당 (2025년 개선):
- 지급액: 잔여 급여일수의 2/3 (67%)
- 조건: 대기기간 7일 + 수급기간 1/2 이전 재취업
- 12개월 이상 고용 유지 필수

<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>""",

    "부정수급": """부정수급 처벌 (2025년 강화):
- 적발시 받은 금액의 5배 추징 (기존 3배)
- 형사처벌 + 명단 공개
- 향후 3년간 실업급여 제한

허위 구직활동, 취업 사실 은닉 등 모두 해당됩니다.""",

    "금액_계산_금지": """실업급여 조건이 충족된다는 전제 하에,
정확한 금액 계산은 복잡한 요소가 많습니다:

필요한 정보:
- 퇴직 전 3개월 평균임금
- 연령 (50세 기준)  
- 근무기간 (수급일수 결정)
- 평균임금의 60% (상한 66,000원, 하한 64,192원)
- 수급일수는 근무기간과 연령에 따라 120~270일

정확한 계산은 계산기를 이용하세요:
<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>"""
}

# ===== AI 시스템 프롬프트 (2025년 정확한 정보) =====
SYSTEM_PROMPT = """당신은 한국 실업급여 전문 상담사입니다.

[절대 규칙 - 2025년 8월 정답]
- 반복수급 감액: 3회 10%, 4회 25%, 5회 40%, 6회 50%
- 구직활동: 1-4차 각 1회, 5차부터 각 2회
- 65세 규칙: 65세 이전부터 계속 근무만 가능
- 임금체불: 2개월 이상만 인정
- 조기재취업수당: 잔여일수의 2/3 (67%)
- 부정수급: 5배 추징 + 명단 공개
- 이직사유 판단: 마지막 직장의 이직사유만 중요
- 근무기간 ≠ 수급일수 (절대 다름!)

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
2. 그 외 모든 경우: "실업급여 조건이 충족된다는 전제 하에," 시작

금액 계산 절대 금지:
- 구체적 금액 계산 시도 금지
- 수급일수 × 일급 계산 금지
- "180일 근무 = 180일 수급" 같은 오해 금지

안내 규칙:
- 자격/조건 확인 → "자세한 조건은 가이드를 참고하세요"
- 금액 계산 → 계산기 버튼 표시
- 복잡한 상황 → 가이드 + 계산기 둘 다

금액 질문 표준 답변:
"실업급여 조건이 충족된다는 전제 하에,
자세한 조건: 가이드 참고
정확한 금액: <a href="https://sudanghelp.co.kr/unemployment/" target="_blank">📊 실업급여 계산기 바로가기</a>"

현재 기준:
{current_info}

핵심 규칙:
1. FAQ는 참고만, 사용자의 구체적 수치(근무기간, 임금)를 180일, 상한/하한액 규칙에 직접 대입하여 답변
2. 정확한 정보만 제공 (추측 금지)
3. 불확실하면 "고용노동부 상담센터 1350" 안내

계산기 안내:
- 반드시 이 형태로: <a href="https://sudanghelp.co.kr/unemployment/" target="_blank">📊 실업급여 계산기 바로가기</a>
- 단순 URL 텍스트 금지

답변 구조:
- 결론 먼저 (가능/불가능)
- 근거 2-3개
- 계산기 링크 (금액 관련시)
- 한국어로 2-4문단, 500자 이내"""

# ===== FAQ 설정 =====
FAQ_CONFIG = {
    "min_threshold": 2.5,  # 최소 관련도 점수 유지
    "max_faqs": 2,         # 최대 주입 FAQ 수
    "max_tokens": 150      # 100 → 150 증가
}

# ===== 실업급여 키워드 (2025년 확장) =====
UNEMPLOYMENT_KEYWORDS = [
    '실업급여', '실업', '급여', '구직급여', '구직', '고용보험', '고용센터',
    '실직', '퇴사', '퇴직', '해고', '권고사직', '계약만료', '폐업',
    '수급자격', '수급', '실업인정', '구직활동', '재취업', '취업활동',
    '워크넷', '이직확인서', '이직', '급여일수', '소정급여',
    '상한액', '하한액', '지급액', '신청', '자격', '조건',
    '4대보험', '고용', '보험', '노동부', '노동청',
    '프리랜서', '정규직', '그만뒀', '그만둔',
    # 추가 키워드
    '일주일', '주3일', '주4일', '주5일', '계약직', '월급',
    '아르바이트', '알바', '65세', '66세', '고령자',
    '임금체불', '체불', '반복수급', '감액', '구직외활동',
    # 2025년 추가
    '받았', '받으면', '깎이', '작년', '올해',
    '3번', '4번', '5번', '횟수', '차', '자영업', '폐업',
    '조기재취업', '청년', '구직촉진', '부정수급'
]

# ===== 로깅 설정 =====
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/error.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file', 'error_file']
    }
}