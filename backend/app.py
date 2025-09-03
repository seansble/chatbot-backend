print("APP.PY IS LOADING")
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI
import requests
import re
import logging
import logging.config
import bleach
import csv
import hashlib
import uuid
from datetime import datetime, date
from collections import defaultdict
import os
import json
import unicodedata
import config

# 필요한 폴더들 생성
for folder in ["logs", "qa_logs", "data", "stats"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Production 보안 설정
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
)

CORS(app, origins=["*"], supports_credentials=True)


# Rate Limiting 설정
def get_real_ip():
    return (
        request.headers.get("X-Forwarded-For", request.remote_addr)
        .split(",")[0]
        .strip()
    )


limiter = Limiter(
    app=app,
    key_func=get_real_ip,
    default_limits=["100 per hour"],
    storage_uri="memory://",
)

# 로깅 설정
logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# 메모리 기반 추적
calculator_users = {}
daily_usage = defaultdict(lambda: {"date": None, "count": 0})
feedback_counts = defaultdict(lambda: {"like": 0, "dislike": 0})

# 통계 관리
STATS_FILE = "stats/site_stats.json"
VISITORS_FILE = "stats/visitors.txt"

def load_stats():
    """통계 로드"""
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except:
        # 초기값: 방문자 1500명부터 시작
        return {
            "total_visitors": 1500,
            "total_likes": 0,
            "last_updated": datetime.now().isoformat()
        }

def save_stats(stats):
    """통계 저장"""
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Stats save error: {e}")

def track_visitor(fingerprint):
    """방문자 추적"""
    try:
        # 고유 방문자 체크
        visitors = set()
        if os.path.exists(VISITORS_FILE):
            with open(VISITORS_FILE, 'r') as f:
                visitors = set(line.strip() for line in f)
        
        if fingerprint not in visitors:
            visitors.add(fingerprint)
            with open(VISITORS_FILE, 'a') as f:
                f.write(f"{fingerprint}\n")
            
            # 통계 업데이트
            stats = load_stats()
            stats["total_visitors"] += 1
            save_stats(stats)
            return True
    except Exception as e:
        logger.error(f"Visitor tracking error: {e}")
    return False

# 초기 통계 로드
site_stats = load_stats()

# 금액 계산 의도 감지 - 개선
RX_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
ASK_AMT = re.compile(
    r"(얼마|금액|일액|일당|월급|상한|하한|수당|이액|받(?:나요|아|을까요)|나오(?:나요|니|게))"
)
HAS_NUMW = re.compile(rf"{RX_NUM}\s*(원|만원)")
VERB_CALC = re.compile(r"(계산|산정|얼추|대략)\s*(해|해줘|가능|방법)")
INFO_ONLY = re.compile(r"(상한|하한|기준|정의|의미|뭐[야|에요])")


def detect_amount_intent(q: str) -> str:
    """금액 계산 의도 감지 - 개선"""
    t = unicodedata.normalize("NFKC", q).lower()

    # 근무기간 질문은 제외
    if "얼마나 일" in t or "얼마나 근무" in t or "몇 개월" in t or "얼마나 다녀" in t:
        return None

    hits = 0
    hits += 1 if ASK_AMT.search(t) else 0
    hits += 1 if HAS_NUMW.search(t) else 0
    hits += 1 if VERB_CALC.search(t) else 0

    if INFO_ONLY.search(t) and hits == 1:
        return None

    return "AMOUNT_CALC" if hits >= 2 or VERB_CALC.search(t) else None


# 기존 함수들 (변경 없음)
def get_user_keys(request, fingerprint):
    client_ip = (
        request.headers.get("X-Forwarded-For", request.remote_addr)
        .split(",")[0]
        .strip()
    )
    usage_cookie = request.cookies.get("usage_token")

    keys = {
        "ip": f"ip_{client_ip}",
        "fingerprint": f"fp_{client_ip}_{fingerprint}",
        "cookie": f"ck_{client_ip}_{usage_cookie}" if usage_cookie else None,
        "primary": None,
    }

    if usage_cookie:
        keys["primary"] = keys["cookie"]
    else:
        keys["primary"] = keys["fingerprint"]

    return keys


def check_all_limits(keys, limit=3):
    """모든 키로 제한 체크"""
    today = date.today()

    if (
        daily_usage[keys["ip"]]["date"] == today
        and daily_usage[keys["ip"]]["count"] >= limit
    ):
        return False

    if (
        daily_usage[keys["fingerprint"]]["date"] == today
        and daily_usage[keys["fingerprint"]]["count"] >= limit
    ):
        return False

    if (
        keys["cookie"]
        and daily_usage[keys["cookie"]]["date"] == today
        and daily_usage[keys["cookie"]]["count"] >= limit
    ):
        return False

    return True


def increment_all_usage(keys):
    """모든 키의 사용 횟수 증가"""
    today = date.today()

    if daily_usage[keys["ip"]]["date"] != today:
        daily_usage[keys["ip"]] = {"date": today, "count": 0}
    daily_usage[keys["ip"]]["count"] += 1

    if daily_usage[keys["fingerprint"]]["date"] != today:
        daily_usage[keys["fingerprint"]] = {"date": today, "count": 0}
    daily_usage[keys["fingerprint"]]["count"] += 1

    if keys["cookie"]:
        if daily_usage[keys["cookie"]]["date"] != today:
            daily_usage[keys["cookie"]] = {"date": today, "count": 0}
        daily_usage[keys["cookie"]]["count"] += 1


def get_remaining_count(keys):
    """남은 횟수 계산"""
    today = date.today()
    remaining = 3

    if daily_usage[keys["ip"]]["date"] == today:
        remaining = min(remaining, 3 - daily_usage[keys["ip"]]["count"])

    if daily_usage[keys["fingerprint"]]["date"] == today:
        remaining = min(remaining, 3 - daily_usage[keys["fingerprint"]]["count"])

    if keys["cookie"] and daily_usage[keys["cookie"]]["date"] == today:
        remaining = min(remaining, 3 - daily_usage[keys["cookie"]]["count"])

    return max(0, remaining)


def check_calculator_usage(keys):
    """계산기 사용 체크"""
    if keys["ip"] in calculator_users:
        return True
    if keys["fingerprint"] in calculator_users:
        return True
    if keys["cookie"] and keys["cookie"] in calculator_users:
        return True
    return False


def mark_calculator_usage(keys):
    """모든 키에 계산기 사용 표시"""
    calculator_users[keys["ip"]] = True
    calculator_users[keys["fingerprint"]] = True
    if keys["cookie"]:
        calculator_users[keys["cookie"]] = True


def is_unemployment_related(question):
    """실업급여 관련 질문인지 체크 - 완화"""
    # 실업급여가 직접 포함되면 무조건 통과
    if "실업급여" in question or "실업 급여" in question:
        return True

    # 키워드 체크
    return any(keyword in question.lower() for keyword in config.UNEMPLOYMENT_KEYWORDS)


def check_malicious_input(text):
    """악성 패턴 체크"""
    blocked = ["ignore previous", "무시하고", "system:", "assistant:", "<script"]
    for pattern in blocked:
        if pattern in text.lower():
            return False
    return True


def validate_input_length(text):
    """입력 길이 체크"""
    return len(text) <= config.MAX_INPUT_LENGTH


def mask_personal_info(text):
    """개인정보 마스킹"""
    text = re.sub(r"\d{6}-\d{7}", "XXX-XXXX", text)
    text = re.sub(r"010-\d{4}-\d{4}", "010-XXXX-XXXX", text)
    text = re.sub(r"\d{3,4}-\d{3,4}-\d{4}", "XXXX-XXXX-XXXX", text)
    return text


def save_qa_with_user(question, answer, user_key):
    """사용자별로 구분해서 Q&A 저장"""
    user_id = hashlib.md5(user_key.encode()).hexdigest()[:8]

    filename = f"qa_{datetime.now().strftime('%Y_%m')}.csv"
    filepath = os.path.join("qa_logs", filename)

    file_exists = os.path.exists(filepath)

    with open(filepath, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["날짜시간", "사용자ID", "질문", "답변"])

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_id,
                mask_personal_info(question),
                mask_personal_info(answer[:500]),
            ]
        )


def should_use_premise(question):
    """'실업급여 조건이 충족된다는 전제 하에' 사용 여부 판단"""
    question_lower = question.lower()

    # 사용하지 않는 경우들
    dont_use = [
        # 이미 재취업/근무 중
        "일하고",
        "근무하고",
        "활동하고",
        "라이더로",
        "배달하는",
        "프리랜서로",
        "다니고",
        "취직",
        "재취업",
        "시작했",
        "시작한",
        # 제도 설명 질문
        "뭐야",
        "뭐에요",
        "무엇",
        "얼마나",
        "기준",
        "상한",
        "하한",
        # 자격 없음이 명확한 경우
        "3개월",
        "4개월",
        "5개월",  # 6개월 미만
    ]

    # 사용하는 경우들
    do_use = [
        # 일반적 설명이 필요한 경우
        "권고사직",
        "계약만료",
        "해고",
        # 가정적 질문
        "받을 수 있",
        "가능한가",
        "되나요",
        # 과거형 (이미 퇴직)
        "퇴사했",
        "그만뒀",
        "퇴직했",
    ]

    # 사용하지 않는 패턴이 있으면 False
    if any(pattern in question_lower for pattern in dont_use):
        return False

    # 사용하는 패턴이 있으면 True
    if any(pattern in question_lower for pattern in do_use):
        return True

    # 기본값: 사용하지 않음
    return False


def validate_answer(answer, question):
    """답변 검증 및 교정"""
    # 반복수급 관련 오류 체크
    if "반복수급" in question or "네 번째" in question or "4회" in question:
        if "30%" in answer or "3회 이상" in answer:
            return config.FALLBACK_ANSWERS.get("반복수급_감액", answer)

    # 하한액 오류 체크
    if "63,816원" in answer:
        answer = answer.replace("63,816원", "64,192원")
    if "68,640원" in answer:
        answer = answer.replace("68,640원", "66,000원")

    # 비현실적 금액 차단
    MAX_DAILY = 66000
    MAX_TOTAL = MAX_DAILY * 270

    if re.search(rf"{RX_NUM}\s*만\s*원", answer):
        nums = [int(x.replace(",", "")) for x in re.findall(RX_NUM, answer)]
        if any(n > MAX_TOTAL * 1.1 for n in nums):
            return config.FALLBACK_ANSWERS["금액_계산_금지"]

    return answer


def generate_ai_answer(question, calc_data=None):
    """AI 답변 생성 - FAQ 제거 버전"""
    try:
        # 금액 계산 의도 차단
        if detect_amount_intent(question) == "AMOUNT_CALC":
            return config.FALLBACK_ANSWERS["금액_계산_금지"]

        # 6개월 미만 체크 - 개선
        if "년" not in question:  # "8년 3개월" 같은 경우 제외
            month_match = re.search(r"(\d+)\s*개월", question)
            if month_match:
                months = int(month_match.group(1))
                if months < 6:
                    return """고용보험 가입기간이 180일(6개월) 이상이어야 실업급여 수급이 가능합니다. 
6개월 미만 근무시에는 수급 자격이 없습니다.

자세한 상담: 고용노동부 1350"""

        # 부정수급은 항상 경고
        if "부정수급" in question:
            return config.FALLBACK_ANSWERS["부정수급"]

        # 시스템/유저 메시지 구성
        system_prompt = config.SYSTEM_PROMPT.format(current_info=config.CURRENT_INFO)

        # "실업급여 조건이 충족된다는 전제" 사용 여부 결정
        use_premise = should_use_premise(question)

        user_msg = f"질문: {question}"

        # 계산기 데이터 활용
        if calc_data and calc_data.get("calculated"):
            user_msg += f"\n\n[계산기 사용 데이터]"
            user_msg += f"\n- 월 평균임금: {calc_data.get('salary', '미입력')}원"
            user_msg += f"\n- 연령: {calc_data.get('age', '미입력')}세"
            user_msg += f"\n- 예상 일 급여: {calc_data.get('daily_amount', '미계산')}원"
            user_msg += f"\n- 수급 일수: {calc_data.get('days', '미계산')}일"

        # 전제 사용 지침 추가
        if use_premise:
            user_msg += '\n\n지침: 이 질문은 일반적인 설명이 필요하므로 "실업급여 조건이 충족된다는 전제 하에"로 시작하세요.'
        else:
            user_msg += '\n\n지침: 이 질문은 구체적 상황이므로 "실업급여 조건이 충족된다는 전제 하에"를 사용하지 마세요.'

        # 중요 지침 추가
        user_msg += "\n\n⚠️ 중요: 계산기 링크나 고용센터 안내를 직접 하지 마세요. URL을 생성하지 마세요. 순수한 답변만 제공하세요."

        # 컨텍스트 명확화
        if ("하는데" in question or "인데" in question) and "실업급여" in question:
            if "받으면서" not in question and "수급" not in question:
                user_msg += "\n\n⚠️ 중요: 질문자는 현재 해당 일을 하고 있으며, 퇴직 후 실업급여 자격을 묻는 것입니다. 수급 중 부업이 아닙니다!"
                user_msg += "\n답변 구조: 1) 해당 직종의 고용보험 가입 여부 2) 퇴직 후 수급 조건"

        # 특정 케이스 강조
        if "임금체불" in question:
            user_msg += "\n\n중요: 임금체불 2개월 이상시 자진퇴사도 실업급여 가능. 이 점을 반드시 언급하세요."

        if "180일" in question or "합산" in question:
            user_msg += "\n\n중요: 18개월 내 여러 직장 피보험기간은 합산 가능. 연속일 필요 없음."

        if "65세" in question or "66세" in question:
            user_msg += "\n\n중요: 65세 이전부터 계속 근무한 경우만 가능. 65세 이후 신규 고용은 제외."

        if ("회사" in question and "후" in question) or (
            "퇴사" in question and "다시" in question
        ):
            user_msg += "\n\n중요: 실업급여는 마지막 직장의 이직사유만 판단합니다. 이전 직장은 180일 계산에만 사용."

        if "알바" in question or "일하면서" in question:
            user_msg += (
                "\n\n중요: 실업급여 수급 중 근로는 반드시 신고. 미신고시 5배 추징."
            )

        if "다시" in question or "현재" in question or "지금" in question:
            if any(
                word in question
                for word in ["일하고", "근무하고", "활동하고", "라이더로"]
            ):
                user_msg += "\n\n⚠️ 매우 중요: 이미 새로운 일을 시작했다면 실업 상태가 아니므로 실업급여 신청 자체가 불가능합니다!"

        # OpenAI 클라이언트 사용
        logger.info(f"Using Together AI with model: {config.MODEL_NAME}")

        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=config.OPENROUTER_API_KEY,
        )

        completion = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

        # 응답 처리
        answer = completion.choices[0].message.content
        logger.info("AI call successful")

        # 답변 검증
        answer = validate_answer(answer, question)

        # 후처리 (태그 버튼 추가)
        answer = postprocess_answer(answer)

        return answer

    except Exception as e:
        logger.error(f"AI error: {str(e)}")

        # API 실패시 최소한의 fallback
        if any(word in question for word in ["얼마", "금액", "계산"]):
            return config.CALCULATION_GUIDE

        return "일시적 오류가 발생했습니다. 고용노동부 상담센터 1350으로 문의하세요."


def postprocess_answer(answer):
    """답변 후처리 - AI가 생성한 모든 링크 제거 후 통일된 태그 추가"""
    # 마크다운 제거
    answer = (
        answer.replace("**", "").replace("###", "").replace("##", "").replace("#", "")
    )

    # AI가 생성한 모든 링크/계산기 안내 제거
    # 고용노동부 관련 링크 제거
    answer = re.sub(r"\[.*?\]\(https?://[^\)]+\)", "", answer)  # [텍스트](URL) 형식
    answer = re.sub(r"https?://www\.moel\.go\.kr[^\s]*", "", answer)  # 직접 URL
    answer = re.sub(
        r"https?://sudanghelp\.co\.kr[^\s<]*", "", answer  # 우리 사이트 URL도 제거
    )
    answer = re.sub(r"<a[^>]*>.*?</a>", "", answer, flags=re.DOTALL)  # 기존 a 태그 제거

    # 계산기 관련 문구 제거
    answer = re.sub(
        r"(정확한 산정은|정확한 계산은|자세한 계산은|정확한 금액은).*?(계산기|고용센터|1350|확인).*?[\n\.]",
        "",
        answer,
        flags=re.DOTALL,
    )
    answer = re.sub(r"👉.*?(?:확인하세요|바로가기)[\.]?", "", answer)
    answer = re.sub(r"📊.*?바로가기.*?(?=\n|$)", "", answer)

    # 중복 줄바꿈 제거
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

    # 통일된 태그 버튼 추가 (답변 끝에)
    tag_buttons = """

<div class="tag-wrapper" style="overflow-x:auto;white-space:nowrap;padding:15px 0;margin-top:20px;border-top:1px solid #e0e0e0;-webkit-overflow-scrolling:touch;scrollbar-width:none;">
    <a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="display:inline-block;padding:8px 20px;margin:0 6px 0 0;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">실업급여 계산기</a>
    <a href="https://sudanghelp.co.kr/unemployment-guide-2025/" target="_blank" style="display:inline-block;padding:8px 20px;margin:0 6px;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">2025 최신 매뉴얼</a>
    <a href="tel:1350" style="display:inline-block;padding:8px 20px;margin:0 6px;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">고용센터 1350</a>
    <a href="https://www.work24.go.kr" target="_blank" style="display:inline-block;padding:8px 20px;margin:0 6px;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">고용24 바로가기</a>
</div>"""

    return answer + tag_buttons


# 통계 API 추가
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """사이트 통계 조회"""
    try:
        stats = load_stats()
        
        # 전체 좋아요 수 계산
        total_likes = sum(counts["like"] for counts in feedback_counts.values())
        stats["total_likes"] = total_likes
        
        return jsonify({
            "visitors": stats.get("total_visitors", 1500),
            "total_likes": total_likes,
            "last_updated": stats.get("last_updated")
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            "visitors": 1500,
            "total_likes": 0
        })


# 루트 경로
@app.route("/", methods=["GET"])
def index():
    """루트 경로 - Railway 헬스체크용"""
    return jsonify(
        {
            "service": "Unemployment Benefits Chat API",
            "status": "running",
            "version": "2025.09.02",
            "endpoints": {
                "health": "/health",
                "chat": "/api/chat",
                "feedback": "/api/feedback",
                "calculator": "/api/mark-calculator-used",
                "stats": "/api/stats",
                "test": "/api/test-openrouter",
                "debug": "/api/debug",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """헬스체크 엔드포인트"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model": config.MODEL_NAME,
            "version": "2025.09.02",
            "openrouter_key_len": (
                len(config.OPENROUTER_API_KEY) if config.OPENROUTER_API_KEY else 0
            ),
        }
    )


# 디버그 엔드포인트 추가
@app.route("/api/debug", methods=["GET"])
def debug_info():
    """환경 디버그 정보"""
    return jsonify(
        {
            "key_exists": bool(config.OPENROUTER_API_KEY),
            "key_length": (
                len(config.OPENROUTER_API_KEY) if config.OPENROUTER_API_KEY else 0
            ),
            "key_prefix": (
                config.OPENROUTER_API_KEY[:15] if config.OPENROUTER_API_KEY else "N/A"
            ),
            "key_starts_with_valid": (
                config.OPENROUTER_API_KEY.startswith("sk-or-v1-")
                if config.OPENROUTER_API_KEY
                else False
            ),
            "model": config.MODEL_NAME,
            "base_url": config.API_BASE_URL,
            "has_newline": "\\n" in (config.OPENROUTER_API_KEY or ""),
            "has_space": " " in (config.OPENROUTER_API_KEY or ""),
            "railway_env": bool(os.getenv("RAILWAY_ENVIRONMENT")),
        }
    )


# OpenRouter 연결 테스트
@app.route("/api/test-openrouter", methods=["GET"])
def test_openrouter():
    """OpenRouter 연결 테스트"""
    try:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        }

        # 모델 목록 가져오기
        response = requests.get(
            f"{config.API_BASE_URL}/models", headers=headers, timeout=10
        )

        if response.status_code == 200:
            return jsonify(
                {
                    "status": "connected",
                    "code": response.status_code,
                    "message": "API connected successfully",
                }
            )
        else:
            return jsonify(
                {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text[:200],
                }
            )
    except requests.exceptions.ConnectionError:
        return jsonify(
            {"status": "blocked", "message": "Connection blocked or network issue"}
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)[:200]})


@app.route("/api/mark-calculator-used", methods=["POST"])
def mark_calculator_used():
    """계산기 사용 표시"""
    fingerprint = request.json.get("fingerprint")
    if fingerprint:
        is_dev = (
            fingerprint in config.MASTER_FINGERPRINTS
            or config.ENVIRONMENT == "development"
        )

        if not is_dev:
            keys = get_user_keys(request, fingerprint)

            if not check_all_limits(keys, 3):
                return jsonify({"error": "일일 제한으로 계산기 사용 불가"}), 403

            mark_calculator_usage(keys)

        resp = make_response(jsonify({"status": "ok"}))
        if not request.cookies.get("usage_token"):
            new_token = str(uuid.uuid4())
            resp.set_cookie(
                "usage_token", new_token, max_age=86400, httponly=True, samesite="Lax"
            )
        return resp

    return jsonify({"error": "fingerprint required"}), 400


@app.route("/api/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def feedback():
    """좋아요/싫어요 피드백 처리"""
    try:
        data = request.json
        feedback_type = data.get("type")
        answer_hash = hashlib.md5(data.get("answer", "").encode()).hexdigest()[:16]

        if feedback_type == "dislike":
            logger.warning(f"Dislike feedback: {data.get('question')[:100]}")

        # 피드백 카운트 증가
        feedback_counts[answer_hash][feedback_type] += 1
        
        # 좋아요일 때 전체 통계 업데이트
        if feedback_type == "like":
            stats = load_stats()
            stats["total_likes"] = stats.get("total_likes", 0) + 1
            save_stats(stats)

        feedback_file = "qa_logs/feedback.csv"
        file_exists = os.path.exists(feedback_file)

        with open(feedback_file, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["시간", "ID", "타입", "질문", "답변"])

            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    hashlib.md5(data.get("fingerprint", "").encode()).hexdigest()[:8],
                    feedback_type,
                    data.get("question", "")[:200],
                    data.get("answer", "")[:200],
                ]
            )

        return jsonify(
            {
                "status": "ok",
                "counts": {
                    "like": feedback_counts[answer_hash]["like"],
                    "dislike": feedback_counts[answer_hash]["dislike"],
                },
            }
        )
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"error": "failed"}), 500


@app.route("/api/feedback/count/<answer_hash>", methods=["GET"])
def get_feedback_count(answer_hash):
    """특정 답변의 좋아요/싫어요 수 조회"""
    return jsonify(
        {
            "like": feedback_counts[answer_hash]["like"],
            "dislike": feedback_counts[answer_hash]["dislike"],
        }
    )


@app.route("/api/chat", methods=["POST"])
@limiter.limit("5 per minute")
@limiter.limit("50 per hour", key_func=get_remote_address)
def chat():
    try:
        question = request.json.get("question", "")
        fingerprint = request.json.get("fingerprint", "")
        calc_data = request.json.get("calcData")

        # 방문자 추적
        track_visitor(fingerprint)

        # 개발자 체크
        is_dev = (
            fingerprint in config.MASTER_FINGERPRINTS
            or config.ENVIRONMENT == "development"
        )

        # User-Agent 체크
        user_agent = request.headers.get("User-Agent", "")
        if not user_agent or "bot" in user_agent.lower():
            return jsonify({"error": "접근이 차단되었습니다"}), 403

        # 빈 질문 체크
        if not question:
            return jsonify({"error": "질문이 없습니다"}), 400

        # HTML 태그 제거
        question = bleach.clean(question, tags=[], strip=True)

        # 입력 길이 체크
        if not validate_input_length(question):
            return (
                jsonify(
                    {"error": f"질문은 {config.MAX_INPUT_LENGTH}자 이내로 작성해주세요"}
                ),
                400,
            )

        # 악성 패턴 체크
        if not check_malicious_input(question):
            return jsonify({"error": "허용되지 않는 입력입니다"}), 400

        # 실업급여 관련 체크
        if not is_unemployment_related(question):
            return jsonify(
                {
                    "answer": "실업급여 관련 질문만 답변 가능합니다. 문의: 고용노동부 상담센터 1350",
                    "remaining": (
                        999
                        if is_dev
                        else get_remaining_count(get_user_keys(request, fingerprint))
                    ),
                }
            )

        # 개발자가 아닐 때만 제한 체크
        if not is_dev:
            keys = get_user_keys(request, fingerprint)

            # 일일 3회 제한
            if not check_all_limits(keys, 3):
                return jsonify(
                    {"error": "일일 3회 초과. 내일 다시 이용하세요", "remaining": 0}
                )

            increment_all_usage(keys)
            remaining = get_remaining_count(keys)
        else:
            remaining = 999
            keys = {"primary": f"dev_{fingerprint}"}

        # AI로 답변 생성
        answer = generate_ai_answer(question, calc_data)

        # 답변 해시 생성
        answer_hash = hashlib.md5(answer.encode()).hexdigest()[:16]

        # Q&A 저장
        save_qa_with_user(question, answer, keys["primary"])

        # 로깅
        logger.info(
            {
                "action": "chat_request",
                "user_id": keys["primary"][:8],
                "is_dev": is_dev,
                "remaining": remaining,
            }
        )

        # 응답 생성
        resp = make_response(
            jsonify(
                {
                    "answer": answer,
                    "answer_hash": answer_hash,
                    "sources": [],
                    "remaining": remaining,
                    "updated": "2025-09-02",
                }
            )
        )

        # 쿠키 설정
        if not request.cookies.get("usage_token"):
            new_token = str(uuid.uuid4())
            resp.set_cookie(
                "usage_token", new_token, max_age=86400, httponly=True, samesite="Lax"
            )

        return resp

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"error": "서버 오류가 발생했습니다"}), 500


# 보안 헤더 추가
@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


# 등록된 라우트 확인
print("REGISTERED ROUTES:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.endpoint}: {rule.rule}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)