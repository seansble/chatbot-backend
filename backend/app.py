# backend/app.py 맨 위에 추가
import sys
import traceback

print("="*60)
print("DEBUG: Starting imports...")
print("="*60)

try:
    print("1. Importing Flask...")
    from flask import Flask, request, jsonify, make_response
    print("✓ Flask imported")
    
    print("2. Importing config...")
    import config
    print("✓ config imported")
    
    print("3. Importing RAG modules...")
    from rag.retriever import RAGRetriever
    print("✓ RAGRetriever imported")
    from rag.workflow import SemanticRAGWorkflow
    print("✓ SemanticRAGWorkflow imported")
    
    print("4. All imports successful!")
    
except Exception as e:
    print("="*60)
    print(f"❌ IMPORT ERROR: {e}")
    print("="*60)
    traceback.print_exc()
    print("="*60)
    sys.exit(1)
    
print("APP.PY IS LOADING")
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI
from datetime import datetime, date, timedelta
import requests
import re
import logging
import logging.config
import bleach
import csv
import hashlib
import uuid
from collections import defaultdict
import os
import json
import unicodedata
import config
import sqlite3
import time

# RAG 시스템 임포트
import sys

sys.path.append("backend")
from rag.retriever import RAGRetriever
from rag.workflow import SemanticRAGWorkflow as RAGWorkflow

# 필요한 폴더들 생성
for folder in ["logs", "qa_logs", "data", "stats", "backend/rag", "cache"]:
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

# CORS 설정 - config에서 가져옴
CORS(app, origins=config.ALLOWED_ORIGINS, supports_credentials=True)

# RAG 시스템 초기화
try:
    print("Initializing RAG system...")
    retriever = RAGRetriever(use_reranker=False, use_hybrid=True)
    workflow = RAGWorkflow(retriever)
    USE_RAG = True
    print("RAG system initialized successfully")
except Exception as e:
    print(f"RAG initialization failed: {e}")
    retriever = None
    workflow = None
    USE_RAG = False


# SQLite DB 초기화 (채팅 로그용)
def init_database():
    conn = sqlite3.connect("chat_feedback.db")
    cursor = conn.cursor()

    # 채팅 로그 테이블
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            question TEXT,
            answer TEXT,
            confidence REAL,
            method TEXT,
            response_time REAL,
            coverage_score REAL
        )
    """
    )

    # 피드백 테이블
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            answer_hash TEXT,
            thumbs_up INTEGER DEFAULT 0,
            thumbs_down INTEGER DEFAULT 0
        )
    """
    )

    conn.commit()
    conn.close()


init_database()


# 캐시 시스템
class SimpleCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "response_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Cache save error: {e}")

    def get(self, key):
        cached = self.cache.get(key)
        if cached:
            # 1시간 만료
            if time.time() - cached.get("timestamp", 0) > 3600:
                del self.cache[key]
                return None
        return cached

    def set(self, key, value):
        # 캐시 크기 제한 (1000개)
        if len(self.cache) > 1000:
            # 오래된 항목 제거
            oldest = sorted(self.cache.items(), key=lambda x: x[1].get("timestamp", 0))[
                :100
            ]
            for k, _ in oldest:
                del self.cache[k]

        self.cache[key] = {"value": value, "timestamp": time.time()}
        self._save_cache()


cache_system = SimpleCache()


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

# 토큰 기반 추적 (메모리 Redis 대체) - 일일 3회 제한 핵심
token_usage = {}  # {token: {date: count}}

# 통계 관리
STATS_FILE = "stats/site_stats.json"
VISITORS_FILE = "stats/visitors.txt"


# 비용 추적
class CostTracker:
    def __init__(self):
        self.daily_costs = {}
        self.cost_file = "stats/daily_costs.json"
        self._load_costs()

    def _load_costs(self):
        if os.path.exists(self.cost_file):
            try:
                with open(self.cost_file, "r") as f:
                    self.daily_costs = json.load(f)
            except:
                self.daily_costs = {}

    def track_api_call(self, input_tokens, output_tokens, model="qwen3-235b"):
        today = date.today().isoformat()

        if today not in self.daily_costs:
            self.daily_costs[today] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0,
            }

        # 비용 계산 (Qwen3-235B 기준)
        input_cost = (input_tokens / 1000000) * 0.2
        output_cost = (output_tokens / 1000000) * 0.6

        self.daily_costs[today]["calls"] += 1
        self.daily_costs[today]["input_tokens"] += input_tokens
        self.daily_costs[today]["output_tokens"] += output_tokens
        self.daily_costs[today]["cost"] += input_cost + output_cost

        # 저장
        with open(self.cost_file, "w") as f:
            json.dump(self.daily_costs, f)

        return self.daily_costs[today]["cost"]


cost_tracker = CostTracker()


def load_stats():
    """통계 로드"""
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "total_visitors": 1500,
            "total_likes": 0,
            "last_updated": datetime.now().isoformat(),
        }


def save_stats(stats):
    """통계 저장"""
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Stats save error: {e}")


def track_visitor(fingerprint):
    """방문자 추적"""
    try:
        visitors = set()
        if os.path.exists(VISITORS_FILE):
            with open(VISITORS_FILE, "r") as f:
                visitors = set(line.strip() for line in f)

        if fingerprint not in visitors:
            visitors.add(fingerprint)
            with open(VISITORS_FILE, "a") as f:
                f.write(f"{fingerprint}\n")

            stats = load_stats()
            stats["total_visitors"] += 1
            save_stats(stats)
            return True
    except Exception as e:
        logger.error(f"Visitor tracking error: {e}")
    return False


# 토큰 관리 함수들 (일일 3회 제한 핵심)
def generate_user_token():
    """고유 토큰 생성"""
    return str(uuid.uuid4())


def get_or_create_token(request):
    """토큰 확인 또는 생성"""
    token = request.cookies.get(config.TOKEN_COOKIE_NAME)
    is_new = False

    if not token:
        token = generate_user_token()
        is_new = True

    return token, is_new


def check_token_usage(token, is_new_token):
    today_str = date.today().isoformat()
    
    if token not in token_usage:
        token_usage[token] = {}
    
    current_count = token_usage[token].get(today_str, 0)
    
    # 신규 사용자라도 이미 사용했으면 일반 사용자로 처리
    if current_count > 0:
        is_new_token = False
    
    # 한도 체크
    limit = config.DAILY_LIMIT
    if current_count >= limit:
        return False, 0
    
    return True, limit - current_count

def increment_token_usage(token):
    """토큰 사용량 증가"""
    today_str = date.today().isoformat()

    if token not in token_usage:
        token_usage[token] = {}

    token_usage[token][today_str] = token_usage[token].get(today_str, 0) + 1

    # 오래된 날짜 정리
    for date_key in list(token_usage[token].keys()):
        if date_key != today_str:
            del token_usage[token][date_key]


# 초기 통계 로드
site_stats = load_stats()

# 금액 계산 의도 감지
RX_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
ASK_AMT = re.compile(
    r"(얼마|금액|일액|일당|월급|상한|하한|수당|총액|받(?:나요|아|을까요)|나오(?:나요|니|게))"
)
HAS_NUMW = re.compile(rf"{RX_NUM}\s*(원|만원)")
VERB_CALC = re.compile(r"(계산|산정|예측|대략)\s*(해|해줘|가능|방법)")
INFO_ONLY = re.compile(r"(상한|하한|기준|정의|의미|뭔[야|에요])")


def detect_amount_intent(q: str) -> str:
    """금액 계산 의도 감지"""
    t = unicodedata.normalize("NFKC", q).lower()

    if "얼마나 일" in t or "얼마나 근무" in t or "몇 개월" in t or "얼마나 다녀" in t:
        return None

    hits = 0
    hits += 1 if ASK_AMT.search(t) else 0
    hits += 1 if HAS_NUMW.search(t) else 0
    hits += 1 if VERB_CALC.search(t) else 0

    if INFO_ONLY.search(t) and hits == 1:
        return None

    return "AMOUNT_CALC" if hits >= 2 or VERB_CALC.search(t) else None


# 기존 함수들 유지
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


# backend/app.py의 is_unemployment_related 함수 수정 (약 1020번째 줄)

def is_unemployment_related(question):
    """실업급여 관련 질문인지 엄격하게 체크"""
    
    question_lower = question.lower()
    
    # 차단 키워드 (무조건 차단)
    BLOCK_KEYWORDS = [
        '주택관리사', '인강', '자격증', '시험', '강의',
        '비트코인', '주식', '부동산', '대출', '펀드',
        '다이어트', '운동', '요리', '레시피', '여행',
        '게임', '영화', '드라마', '날씨', '뉴스',
        'ai', '인공지능', '챗봇', '프로그래밍', '코딩',
        '맛집', '카페', '쇼핑', '패션', '뷰티'
    ]
    
    # 인사말도 차단
    GREETINGS = ['안녕', '하이', 'hello', 'hi', '뭐해', '뭐하니', '반가워']
    
    # 차단 키워드 체크
    for keyword in BLOCK_KEYWORDS:
        if keyword in question_lower:
            return False
    
    # 인사말 체크
    if len(question_lower) < 10:  # 짧은 문장
        for greeting in GREETINGS:
            if greeting in question_lower:
                return False
    
    # 필수 키워드 (최소 하나는 포함해야 함)
    REQUIRED_KEYWORDS = [
        '실업', '급여', '퇴사', '퇴직', '해고', '권고사직',
        '고용보험', '수급', '구직', '실직', '일했', '근무',
        '월급', '연봉', '계약만료', '이직', '회사', '직장',
        '프리랜서', '계약직', '정규직', '근로', '퇴직금',
        '상한액', '하한액', '수당', '일당', '일급',
        '180일', '6개월', '18개월', '반복수급', '구직활동'
    ]
    
    # 필수 키워드 체크
    has_required = any(keyword in question_lower for keyword in REQUIRED_KEYWORDS)
    
    # 숫자+근무 패턴 (예: "8개월 일했어요")
    import re
    has_work_pattern = bool(re.search(r'\d+\s*(개월|년|만\s*원|만원|일|살)', question_lower))
    
    # 필수 키워드가 있거나 근무 패턴이 있어야만 통과
    return has_required or has_work_pattern  # 둘 중 하나라도 있어야 True


def check_malicious_input(text):
    """악성 패턴 체크"""
    blocked = [
        "ignore previous",
        "무시하고",
        "system:",
        "assistant:",
        "<script",
        "javascript:",
    ]
    for pattern in blocked:
        if pattern in text.lower():
            return False
    return True


def validate_input_length(text):
    """입력 길이 체크"""
    return 2 <= len(text) <= config.MAX_INPUT_LENGTH


def mask_personal_info(text):
    """개인정보 마스킹"""
    text = re.sub(r"\d{6}-\d{7}", "XXX-XXXX", text)
    text = re.sub(r"010-\d{4}-\d{4}", "010-XXXX-XXXX", text)
    text = re.sub(r"\d{3,4}-\d{3,4}-\d{4}", "XXXX-XXXX-XXXX", text)
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "****@****.***", text
    )
    return text


def save_to_database(
    question,
    answer,
    user_id,
    confidence=0,
    method="",
    response_time=0,
    coverage_score=0,
):
    """데이터베이스에 저장"""
    conn = sqlite3.connect("chat_feedback.db")
    cursor = conn.cursor()

    # 파라미터화된 쿼리 (SQL Injection 방지)
    cursor.execute(
        """
        INSERT INTO chat_logs (user_id, question, answer, confidence, method, response_time, coverage_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            user_id,
            mask_personal_info(question[:500]),
            mask_personal_info(answer[:1000]),
            confidence,
            method,
            response_time,
            coverage_score,
        ),
    )

    conn.commit()
    conn.close()


def save_qa_with_user(question, answer, user_key, answer_hash=""):
    """사용자별로 구분해서 Q&A 저장"""
    user_id = hashlib.md5(user_key.encode()).hexdigest()[:8]

    like_count = feedback_counts.get(answer_hash, {}).get("like", 0)
    dislike_count = feedback_counts.get(answer_hash, {}).get("dislike", 0)

    filename = f"qa_{datetime.now().strftime('%Y_%m')}.csv"
    filepath = os.path.join("qa_logs", filename)

    file_exists = os.path.exists(filepath)

    question_short = mask_personal_info(question[:50].replace("\n", " "))
    answer_short = mask_personal_info(answer[:100].replace("\n", " "))

    with open(filepath, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writerow(
                ["날짜시간", "사용자ID", "질문(50자)", "답변(100자)", "👍", "👎"]
            )

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_id,
                question_short,
                answer_short,
                like_count,
                dislike_count,
            ]
        )


def should_use_premise(question):
    """'실업급여 조건이 충족된다는 전제 하에' 사용 여부 판단"""
    question_lower = question.lower()

    dont_use = [
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
        "뭐야",
        "뭐에요",
        "무엇",
        "얼마나",
        "기준",
        "상한",
        "하한",
        "3개월",
        "4개월",
        "5개월",
    ]

    do_use = [
        "권고사직",
        "계약만료",
        "해고",
        "받을 수 있",
        "가능한가",
        "되나요",
        "퇴사했",
        "그만뒀",
        "퇴직했",
    ]

    if any(pattern in question_lower for pattern in dont_use):
        return False

    if any(pattern in question_lower for pattern in do_use):
        return True

    return False


def validate_answer(answer, question):
    """답변 검증 및 교정"""
    if "반복수급" in question or "네 번째" in question or "4회" in question:
        if "30%" in answer or "3회 이상" in answer:
            return config.FALLBACK_ANSWERS.get("반복수급_감액", answer)

    if "63,816원" in answer:
        answer = answer.replace("63,816원", "64,192원")
    if "68,640원" in answer:
        answer = answer.replace("68,640원", "66,000원")

    MAX_DAILY = 66000
    MAX_TOTAL = MAX_DAILY * 270

    if re.search(rf"{RX_NUM}\s*만\s*원", answer):
        nums = [int(x.replace(",", "")) for x in re.findall(RX_NUM, answer)]
        if any(n > MAX_TOTAL * 1.1 for n in nums):
            return config.FALLBACK_ANSWERS["금액_계산_금지"]

    return answer


def generate_ai_answer_with_rag(question, calc_data=None):
    """RAG를 사용한 AI 답변 생성"""
    try:
        start_time = time.time()

        # 1. 캐시 체크
        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached = cache_system.get(cache_key)
        if cached:
            logger.info("Cache hit for question")
            return cached["value"]

        # 2. 금액 계산 의도 차단
        if len(question) < 100 and detect_amount_intent(question) == "AMOUNT_CALC":
            return config.FALLBACK_ANSWERS["금액_계산_금지"]

        # 3. 6개월 미만 체크
        if "년" not in question:
            month_match = re.search(r"(\d+)\s*개월", question)
            if month_match:
                months = int(month_match.group(1))
                if months < 6:
                    return """고용보험 가입기간이 180일(6개월) 이상이어야 실업급여 수급이 가능합니다. 
6개월 미만 근무시에는 수급 자격이 없습니다.

자세한 상담: 고용노동부 1350"""

        # 4. 부정수급 경고
        if "부정수급" in question:
            return config.FALLBACK_ANSWERS["부정수급"]

        # 5. 계산기 데이터가 있으면 질문 확장
        enriched_query = question
        if calc_data and calc_data.get("calculated"):
            enriched_query = f"""
            {question}
            
            [계산 정보]
            평균임금: {calc_data.get('salary', '')}
            근무기간: {calc_data.get('work_period', '')}
            나이: {calc_data.get('age', '')}
            퇴사사유: {calc_data.get('reason', '')}
            """

        # 6. RAG 워크플로우 실행
        logger.info(f"Running RAG workflow for: {mask_personal_info(question[:50])}")
        result = workflow.run(enriched_query)

        # 7. 결과 처리
        coverage_score = result.get("coverage_score", 0)
        confidence = result.get("confidence", 0)
        method = result.get("method", "unknown")

        logger.info(
            f"Coverage: {coverage_score:.2f}, Confidence: {confidence:.2f}, Method: {method}"
        )

        # 8. 답변 가져오기
        answer = result.get("answer", "")

        if not answer and result.get("documents"):
            answer = result["documents"][0]["text"]

        # 9. 답변 검증 및 후처리
        answer = validate_answer(answer, question)
        answer = postprocess_answer(answer)

        # 10. 캐시 저장
        cache_system.set(cache_key, answer)

        # 11. 응답 시간 및 비용 추적
        response_time = time.time() - start_time

        # 대략적인 토큰 추정
        input_tokens = len(enriched_query) * 2 + 500
        output_tokens = len(answer) * 2

        # 비용 추적 (LLM 사용한 경우만)
        if method in ["enhanced", "regenerated"]:
            cost_tracker.track_api_call(input_tokens, output_tokens)

        # 12. DB 저장
        user_id = hashlib.md5(question.encode()).hexdigest()[:8]
        save_to_database(
            question, answer, user_id, confidence, method, response_time, coverage_score
        )

        return answer

    except Exception as e:
        logger.error(f"RAG AI error: {str(e)}")
        return generate_ai_answer(question, calc_data)


def generate_ai_answer(question, calc_data=None, stream=False):
    """기존 AI 답변 생성 (폴백용)"""
    try:
        if detect_amount_intent(question) == "AMOUNT_CALC":
            return config.FALLBACK_ANSWERS["금액_계산_금지"]

        if "년" not in question:
            month_match = re.search(r"(\d+)\s*개월", question)
            if month_match:
                months = int(month_match.group(1))
                if months < 6:
                    return """고용보험 가입기간이 180일(6개월) 이상이어야 실업급여 수급이 가능합니다. 
6개월 미만 근무시에는 수급 자격이 없습니다.

자세한 상담: 고용노동부 1350"""

        if "부정수급" in question:
            return config.FALLBACK_ANSWERS["부정수급"]

        system_prompt = RAGWorkflow.SYSTEM_PROMPT_BASE  # 추가
        use_premise = should_use_premise(question)

        user_msg = f"질문: {question}"

        if calc_data and calc_data.get("calculated"):
            user_msg += f"\n\n[계산기 사용 데이터]"
            user_msg += f"\n- 월 평균임금: {calc_data.get('salary', '미입력')}원"
            user_msg += f"\n- 연령: {calc_data.get('age', '미입력')}세"
            user_msg += f"\n- 예상 일 급여: {calc_data.get('daily_amount', '미계산')}원"
            user_msg += f"\n- 수급 일수: {calc_data.get('days', '미계산')}일"

        if use_premise:
            user_msg += '\n\n지침: 이 질문은 일반적인 설명이 필요하므로 "실업급여 조건이 충족된다는 전제 하에"로 시작하세요.'
        else:
            user_msg += '\n\n지침: 이 질문은 구체적 상황이므로 "실업급여 조건이 충족된다는 전제 하에"를 사용하지 마세요.'

        user_msg += "\n\n⚠️ 중요: 계산기 링크나 고용센터 안내를 직접 하지 마세요. URL을 생성하지 마세요. 순수한 답변만 제공하세요."

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
            stream=stream,
        )

        if stream:
            return completion

        answer = completion.choices[0].message.content
        logger.info("AI call successful")

        answer = validate_answer(answer, question)
        answer = postprocess_answer(answer)

        return answer

    except Exception as e:
        logger.error(f"AI error: {str(e)}")

        if any(word in question for word in ["얼마", "금액", "계산"]):
            return config.CALCULATION_GUIDE

        return "일시적 오류가 발생했습니다. 고용노동부 상담센터 1350으로 문의하세요."


def postprocess_answer(answer):
    """답변 후처리"""
    answer = (
        answer.replace("**", "").replace("###", "").replace("##", "").replace("#", "")
    )

    answer = re.sub(r"\[.*?\]\(https?://[^\)]+\)", "", answer)
    answer = re.sub(r"https?://www\.moel\.go\.kr[^\s]*", "", answer)
    answer = re.sub(r"https?://sudanghelp\.co\.kr[^\s<]*", "", answer)
    answer = re.sub(r"<a[^>]*>.*?</a>", "", answer, flags=re.DOTALL)

    answer = re.sub(
        r"(정확한 산정은|정확한 계산은|자세한 계산은|정확한 금액은).*?(계산기|고용센터|1350|확인).*?[\n\.]",
        "",
        answer,
        flags=re.DOTALL,
    )
    answer = re.sub(r"👉.*?(?:확인하세요|바로가기)[\.]?", "", answer)
    answer = re.sub(r"📊.*?바로가기.*?(?=\n|$)", "", answer)

    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

    tag_buttons = """

<div class="tag-wrapper" style="position:relative;overflow-x:auto;white-space:nowrap;padding:15px 0;margin-top:20px;border-top:1px solid #e0e0e0;-webkit-overflow-scrolling:touch;scrollbar-width:none;">
    <div style="position:absolute;right:0;top:0;bottom:0;width:50px;background:linear-gradient(to right,transparent,rgba(26,26,26,0.95));pointer-events:none;z-index:1;"></div>
    <a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="display:inline-block;padding:8px 20px;margin:0 6px 0 0;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">실업급여 계산기</a>
    <a href="https://sudanghelp.co.kr/unemployment-guide-2025/" target="_blank" style="display:inline-block;padding:8px 20px;margin:0 6px;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">2025 최신 매뉴얼</a>
    <a href="tel:1350" style="display:inline-block;padding:8px 20px;margin:0 6px;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">고용센터 1350</a>
    <a href="https://www.work24.go.kr" target="_blank" style="display:inline-block;padding:8px 20px;margin:0 6px;background:#f5f5f5;border:1px solid #ddd;border-radius:20px;text-decoration:none;color:#333;font-size:14px;white-space:nowrap;transition:all 0.2s;">고용24 바로가기</a>
</div>"""

    return answer + tag_buttons


# API 엔드포인트들
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """사이트 통계 조회"""
    try:
        stats = load_stats()
        total_likes = sum(counts["like"] for counts in feedback_counts.values())
        stats["total_likes"] = total_likes

        today = date.today().isoformat()
        today_cost = cost_tracker.daily_costs.get(today, {}).get("cost", 0)

        return jsonify(
            {
                "visitors": stats.get("total_visitors", 1500),
                "total_likes": total_likes,
                "last_updated": stats.get("last_updated"),
                "today_cost": f"${today_cost:.2f}",
            }
        )
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({"visitors": 1500, "total_likes": 0})


@app.route("/", methods=["GET"])
def index():
    """루트 경로"""
    return jsonify(
        {
            "service": "Unemployment Benefits Chat API",
            "status": "running",
            "version": "2025.09.03",
            "rag_enabled": USE_RAG,
            "token_system": True,
            "cache_enabled": True,
            "endpoints": {
                "health": "/health",
                "chat": "/api/chat",
                "feedback": "/api/feedback",
                "calculator": "/api/mark-calculator-used",
                "stats": "/api/stats",
                "test": "/api/test-openrouter",
                "test_rag": "/api/test-rag",
                "debug": "/api/debug",
                "costs": "/api/costs",
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
            "version": "2025.09.03",
            "rag_enabled": USE_RAG,
            "token_system": True,
            "cache_hits": len(cache_system.cache),
            "openrouter_key_len": (
                len(config.OPENROUTER_API_KEY) if config.OPENROUTER_API_KEY else 0
            ),
        }
    )


@app.route("/api/costs", methods=["GET"])
def get_costs():
    """비용 조회 API"""
    try:
        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()

        weekly_cost = 0
        for day, data in cost_tracker.daily_costs.items():
            if day >= week_ago:
                weekly_cost += data["cost"]

        return jsonify(
            {
                "today": cost_tracker.daily_costs.get(today, {}),
                "weekly_total": f"${weekly_cost:.2f}",
                "daily_average": f"${weekly_cost/7:.2f}",
            }
        )
    except Exception as e:
        logger.error(f"Costs API error: {e}")
        return jsonify({"error": "Failed to get costs"}), 500


@app.route("/api/test-rag", methods=["GET"])
def test_rag():
    """RAG 시스템 테스트"""
    if not USE_RAG:
        return jsonify({"error": "RAG system not initialized"}), 500

    test_queries = [
        "배민 라이더 실업급여 받을 수 있나요?",
        "세번째 실업급여 얼마나 깎이나요?",
        "권고사직 증거가 없으면 어떻게 되나요?",
    ]

    results = []
    for query in test_queries:
        try:
            result = workflow.run(query)
            results.append(
                {
                    "query": query,
                    "documents_found": len(result.get("documents", [])),
                    "coverage_score": result.get("coverage_score", 0),
                    "confidence": result.get("confidence", 0),
                    "method": result.get("method", "unknown"),
                }
            )
        except Exception as e:
            results.append({"query": query, "error": str(e)})

    return jsonify({"test_results": results})


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
            "model": config.MODEL_NAME,
            "base_url": config.API_BASE_URL,
            "railway_env": bool(os.getenv("RAILWAY_ENVIRONMENT")),
            "rag_enabled": USE_RAG,
            "token_system": True,
            "active_tokens": len(token_usage),
            "cache_size": len(cache_system.cache),
            "environment": config.ENVIRONMENT,
        }
    )


@app.route("/api/test-openrouter", methods=["GET"])
def test_openrouter():
    """API 연결 테스트"""
    try:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        }

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
            mark_calculator_usage(keys)

        resp = make_response(jsonify({"status": "ok"}))
        if not request.cookies.get("usage_token"):
            new_token = str(uuid.uuid4())
            resp.set_cookie(
                "usage_token",
                new_token,
                max_age=86400,
                httponly=True,
                secure=request.is_secure,
                samesite="Lax",
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
        answer_hash = data.get("answer_hash", "")

        if not answer_hash:
            answer_hash = hashlib.md5(data.get("answer", "").encode()).hexdigest()[:16]

        if feedback_type == "dislike":
            logger.warning(
                f"Dislike feedback: {mask_personal_info(data.get('question', '')[:100])}"
            )

        feedback_counts[answer_hash][feedback_type] += 1

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
                    mask_personal_info(data.get("question", "")[:200]),
                    mask_personal_info(data.get("answer", "")[:200]),
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
@limiter.limit("10 per minute")
@limiter.limit("50 per hour", key_func=get_remote_address)
def chat():
    try:
        question = request.json.get("question", "")
        fingerprint = request.json.get("fingerprint", "")
        calc_data = request.json.get("calcData")

        # 방문자 추적
        track_visitor(fingerprint)

        # 토큰 확인/생성
        token, is_new = get_or_create_token(request)

        # 개발자 체크
        is_dev = (
            fingerprint in config.MASTER_FINGERPRINTS
            or config.ENVIRONMENT == "development"
        )

        # User-Agent 체크
        user_agent = request.headers.get("User-Agent", "")
        if not user_agent or "bot" in user_agent.lower():
            return jsonify({"error": "접근이 차단되었습니다"}), 403

        # 입력 검증
        if not question:
            return jsonify({"error": "질문이 없습니다"}), 400

        # XSS 방지
        question = bleach.clean(question, tags=[], strip=True)

        if not validate_input_length(question):
            return (
                jsonify(
                    {
                        "error": f"질문은 2자 이상 {config.MAX_INPUT_LENGTH}자 이내로 작성해주세요"
                    }
                ),
                400,
            )

        if not check_malicious_input(question):
            return jsonify({"error": "허용되지 않는 입력입니다"}), 400

        # 실업급여 관련 체크
        if not is_unemployment_related(question):
            # 카운트 차감하지 않고 현재 상태만 확인
            if not is_dev:
                today_str = date.today().isoformat()
                current_count = token_usage.get(token, {}).get(today_str, 0)
                limit = config.DAILY_LIMIT  # 모든 사용자 3회로 통일
                remaining = limit - current_count
            else:
                remaining = 999
            
            return jsonify({
                "answer": "실업급여 관련 질문만 답변 가능합니다...",
                "remaining": remaining,  # 현재 남은 횟수만 표시
                "is_new_user": is_new
            })

        # 개발자가 아닐 때 토큰 제한 체크 (일일 3회)
        if not is_dev:
            can_use, remaining = check_token_usage(token, is_new)

            if not can_use:
                error_msg = "일일 3회 초과. 내일 다시 이용하세요"  # 통일된 메시지

                return (
                    jsonify(
                        {"error": error_msg, "remaining": 0, "is_new_user": is_new}
                    ),
                    429,
                )

            # 사용량 증가
            increment_token_usage(token)
            remaining = remaining - 1
        else:
            remaining = 999

        # RAG 시스템 사용 여부에 따라 분기
        if USE_RAG:
            answer = generate_ai_answer_with_rag(question, calc_data)
        else:
            answer = generate_ai_answer(question, calc_data)

        answer_hash = hashlib.md5(answer.encode()).hexdigest()[:16]

        # 로깅용 user_key
        user_key = f"token_{token[:8]}"
        save_qa_with_user(question, answer, user_key, answer_hash)

        logger.info(
            f"Chat: token={token[:8]}, is_new={is_new}, "
            f"remaining={remaining}, method=RAG, "
            f"question_related={is_unemployment_related(question)}"
        )

        resp = make_response(
            jsonify(
                {
                    "answer": answer,
                    "answer_hash": answer_hash,
                    "sources": [],
                    "remaining": remaining,
                    "updated": "2025-09-03",
                    "rag_enabled": USE_RAG,
                    "is_new_user": is_new,
                }
            )
        )

        # 신규 사용자면 쿠키 설정
        if is_new:
            resp.set_cookie(
                config.TOKEN_COOKIE_NAME,
                token,
                max_age=config.TOKEN_MAX_AGE,
                secure=request.is_secure,
                httponly=True,
                samesite="Lax",
            )

        return resp

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"error": "서버 오류가 발생했습니다"}), 500


# 보안 헤더
@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


# 404 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


# 500 핸들러
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


print("REGISTERED ROUTES:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.endpoint}: {rule.rule}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
