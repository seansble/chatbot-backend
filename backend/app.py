print("APP.PY IS LOADING")
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI
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
for folder in ['logs', 'qa_logs', 'data']:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Production 보안 설정
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Strict'
)

CORS(app, 
     origins=['*'],
     supports_credentials=True)

# Rate Limiting 설정
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"],
    storage_uri="memory://"
)

# 로깅 설정
logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# OpenRouter 클라이언트 초기화 (헤더 추가)
client = OpenAI(
    base_url=config.API_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://sudanghelp.co.kr",
        "X-Title": "Sudanghelp Unemployment Chat"
    }
)

# 메모리 기반 추적
calculator_users = {}  
daily_usage = defaultdict(lambda: {"date": None, "count": 0})
feedback_counts = defaultdict(lambda: {"like": 0, "dislike": 0})

# ===== FAQ 시스템 =====
TOKEN_RE = re.compile(r'[가-힣]{2,}|[A-Za-z]+|\d+')

def normalize_text(text):
    """텍스트 정규화"""
    return unicodedata.normalize('NFKC', text)

def tokenize(text):
    """간단한 토큰화"""
    text = normalize_text(text)
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return set(tokens)

def load_knowledge():
    """FAQ 데이터 로드"""
    try:
        with open('data/knowledge.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            faqs = data.get('faqs', [])
            
            # 토큰화 추가
            for faq in faqs:
                faq['_tokens'] = tokenize(faq['q'] + ' ' + faq.get('a_short', faq['a']))
            
            return faqs
    except FileNotFoundError:
        logger.warning("knowledge.json not found, using empty FAQ")
        return []

# FAQ 로드
FAQS = load_knowledge()

def retrieve_faq(query, max_faqs=2, max_tokens=150):
    """관련 FAQ 검색 (토큰 제한 증가)"""
    if not FAQS:
        return []
    
    q_tokens = tokenize(query)
    
    if len(q_tokens) < 2:
        return []
    
    scores = []
    for faq in FAQS:
        # 토큰 오버랩
        overlap = len(q_tokens & faq['_tokens'])
        
        # 키워드 보너스 (확장)
        bonus = 0
        keywords = ['권고사직', '자진퇴사', '임금체불', '계약만료', '재수급',
                   '반복수급', '4회', '5회', '구직활동', '65세', '66세',
                   '자영업', '폐업', '조기재취업']
        for kw in keywords:
            if kw in query and kw in faq['q']:
                bonus += 2
        
        score = overlap + bonus
        if score > 0:
            scores.append((score, faq))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # 임계값 체크
    if not scores or scores[0][0] < config.FAQ_CONFIG['min_threshold']:
        return []
    
    # 토큰 제한으로 선택
    results = []
    used_tokens = 0
    
    for i, (score, faq) in enumerate(scores[:max_faqs]):
        # 첫 번째 FAQ는 더 상세히, 두 번째는 짧게
        if i == 0:
            faq_text = faq.get('a', faq.get('a_short', ''))[:120]
        else:
            faq_text = faq.get('a_short', faq['a'][:80])
        
        faq_tokens = len(faq_text) // 3  # 한글 3자 = 1토큰 추정
        
        if used_tokens + faq_tokens > max_tokens:
            break
            
        results.append({
            'q': faq['q'][:30],
            'a': faq_text,
            'category': faq.get('category', '')
        })
        used_tokens += faq_tokens
    
    return results

# 금액 계산 의도 감지 (정규식 기반)
RX_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
ASK_AMT = re.compile(r"(얼마|금액|일액|일당|월급|상한|하한|수당|총액|받(?:나요|아|을까요)|나오(?:나요|니|게))")
HAS_NUMW = re.compile(fr"{RX_NUM}\s*(원|만원)")
VERB_CALC = re.compile(r"(계산|산정|얼추|대략)\s*(해|해줘|가능|방법)")
INFO_ONLY = re.compile(r"(상한|하한|기준|정의|의미|뭐[야|에요])")

def detect_amount_intent(q: str) -> str:
    """금액 계산 의도 감지"""
    t = unicodedata.normalize("NFKC", q).lower()
    hits = 0
    hits += 1 if ASK_AMT.search(t) else 0
    hits += 1 if HAS_NUMW.search(t) else 0
    hits += 1 if VERB_CALC.search(t) else 0
    
    # 정보성 질문은 제외
    if INFO_ONLY.search(t) and hits == 1:
        return None
    
    return "AMOUNT_CALC" if hits >= 2 or VERB_CALC.search(t) else None

# 기존 함수들
def get_user_keys(request, fingerprint):
    """IP, 쿠키, 지문 모든 조합 반환"""
    client_ip = request.remote_addr
    usage_cookie = request.cookies.get('usage_token')
    
    keys = {
        'ip': f"ip_{client_ip}",
        'fingerprint': f"fp_{client_ip}_{fingerprint}",
        'cookie': f"ck_{client_ip}_{usage_cookie}" if usage_cookie else None,
        'primary': None
    }
    
    if usage_cookie:
        keys['primary'] = keys['cookie']
    else:
        keys['primary'] = keys['fingerprint']
    
    return keys

def check_all_limits(keys, limit=3):
    """모든 키로 제한 체크"""
    today = date.today()
    
    if daily_usage[keys['ip']]["date"] == today and daily_usage[keys['ip']]["count"] >= limit:
        return False
    
    if daily_usage[keys['fingerprint']]["date"] == today and daily_usage[keys['fingerprint']]["count"] >= limit:
        return False
    
    if keys['cookie'] and daily_usage[keys['cookie']]["date"] == today and daily_usage[keys['cookie']]["count"] >= limit:
        return False
    
    return True

def increment_all_usage(keys):
    """모든 키의 사용 횟수 증가"""
    today = date.today()
    
    if daily_usage[keys['ip']]["date"] != today:
        daily_usage[keys['ip']] = {"date": today, "count": 0}
    daily_usage[keys['ip']]["count"] += 1
    
    if daily_usage[keys['fingerprint']]["date"] != today:
        daily_usage[keys['fingerprint']] = {"date": today, "count": 0}
    daily_usage[keys['fingerprint']]["count"] += 1
    
    if keys['cookie']:
        if daily_usage[keys['cookie']]["date"] != today:
            daily_usage[keys['cookie']] = {"date": today, "count": 0}
        daily_usage[keys['cookie']]["count"] += 1

def get_remaining_count(keys):
    """남은 횟수 계산"""
    today = date.today()
    remaining = 3
    
    if daily_usage[keys['ip']]["date"] == today:
        remaining = min(remaining, 3 - daily_usage[keys['ip']]["count"])
    
    if daily_usage[keys['fingerprint']]["date"] == today:
        remaining = min(remaining, 3 - daily_usage[keys['fingerprint']]["count"])
    
    if keys['cookie'] and daily_usage[keys['cookie']]["date"] == today:
        remaining = min(remaining, 3 - daily_usage[keys['cookie']]["count"])
    
    return max(0, remaining)

def check_calculator_usage(keys):
    """계산기 사용 체크"""
    if keys['ip'] in calculator_users:
        return True
    if keys['fingerprint'] in calculator_users:
        return True
    if keys['cookie'] and keys['cookie'] in calculator_users:
        return True
    return False

def mark_calculator_usage(keys):
    """모든 키에 계산기 사용 표시"""
    calculator_users[keys['ip']] = True
    calculator_users[keys['fingerprint']] = True
    if keys['cookie']:
        calculator_users[keys['cookie']] = True

def is_unemployment_related(question):
    """실업급여 관련 질문인지 체크"""
    return any(keyword in question.lower() for keyword in config.UNEMPLOYMENT_KEYWORDS)

def check_malicious_input(text):
    """악성 패턴 체크"""
    blocked = ['ignore previous', '무시하고', 'system:', 'assistant:', '<script']
    for pattern in blocked:
        if pattern in text.lower():
            return False
    return True

def validate_input_length(text):
    """입력 길이 체크"""
    return len(text) <= config.MAX_INPUT_LENGTH

def mask_personal_info(text):
    """개인정보 마스킹"""
    text = re.sub(r'\d{6}-\d{7}', 'XXX-XXXX', text)
    text = re.sub(r'010-\d{4}-\d{4}', '010-XXXX-XXXX', text)
    text = re.sub(r'\d{3,4}-\d{3,4}-\d{4}', 'XXXX-XXXX-XXXX', text)
    return text

def save_qa_with_user(question, answer, user_key):
    """사용자별로 구분해서 Q&A 저장"""
    user_id = hashlib.md5(user_key.encode()).hexdigest()[:8]
    
    filename = f"qa_{datetime.now().strftime('%Y_%m')}.csv"
    filepath = os.path.join('qa_logs', filename)
    
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['날짜시간', '사용자ID', '질문', '답변'])
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id,
            mask_personal_info(question),
            mask_personal_info(answer[:500])
        ])

def validate_answer(answer, question):
    """답변 검증 및 교정"""
    # 반복수급 관련 오류 체크
    if ("반복수급" in question or "네 번째" in question or "4회" in question):
        if "30%" in answer or "3회 이상" in answer:
            return config.FALLBACK_ANSWERS["반복수급_감액"]
    
    # 구직활동 횟수 오류 체크
    if ("구직활동" in question and ("4차" in question or "횟수" in question)):
        if "총 4번" in answer or "4차까지 4번" in answer:
            return config.FALLBACK_ANSWERS["구직활동_횟수"]
    
    # 하한액 오류 체크 (2025년 정확한 수치)
    if "63,816원" in answer:
        answer = answer.replace("63,816원", "64,192원")
    if "68,640원" in answer:
        answer = answer.replace("68,640원", "66,000원")
    
    # 조기재취업수당 오류 체크
    if "조기재취업" in question and ("50%" in answer or "1/2" in answer):
        return config.FALLBACK_ANSWERS["조기재취업수당"]
    
    # 비현실적 금액 차단
    MAX_DAILY = 66000
    MAX_TOTAL = MAX_DAILY * 270  # 17,820,000원
    
    if re.search(fr"{RX_NUM}\s*만\s*원", answer):
        nums = [int(x.replace(",","")) for x in re.findall(RX_NUM, answer)]
        if any(n > MAX_TOTAL * 1.1 for n in nums):
            return config.FALLBACK_ANSWERS["금액_계산_금지"]
    
    return answer

def generate_ai_answer(question, calc_data=None):
    """AI 답변 생성 (2025년 개선 버전)"""
    try:
        # 금액 계산 의도 차단
        if detect_amount_intent(question) == "AMOUNT_CALC":
            return config.FALLBACK_ANSWERS["금액_계산_금지"]
        
        # 180일 미만 근무 체크 (최우선 처리)
        month_match = re.search(r'(\d+)\s*개월', question)
        if month_match:
            months = int(month_match.group(1))
            if months < 6:
                return "고용보험 가입기간이 180일(6개월) 이상이어야 실업급여 수급이 가능합니다. 6개월 미만 근무시에는 수급 자격이 없습니다.\n\n자세한 상담: 고용노동부 1350"
        
        # 1. 특정 케이스는 바로 fallback
        if ("권고사직" in question and "사직서" in question):
            return config.FALLBACK_ANSWERS["권고사직_사직서"]
        
        # "자진퇴사 후" 맥락은 제외
        if ("자진퇴사" in question and "후" not in question and "회사" not in question) and "임금체불" not in question:
            return config.FALLBACK_ANSWERS["자진퇴사"]
        
        if "반복수급" in question and ("감액" in question or "깎" in question):
            return config.FALLBACK_ANSWERS["반복수급_감액"]
        
        if "구직활동" in question and ("몇 번" in question or "횟수" in question):
            return config.FALLBACK_ANSWERS["구직활동_횟수"]
        
        if "자영업" in question and ("폐업" in question or "실업급여" in question):
            return config.FALLBACK_ANSWERS["자영업자"]
        
        if "조기재취업" in question and not any(word in question for word in ["얼마", "깎", "계산", "반복", "4번"]):
            return config.FALLBACK_ANSWERS["조기재취업수당"]
        
        if "부정수급" in question:
            return config.FALLBACK_ANSWERS["부정수급"]
        
        # 2. FAQ 검색
        faqs = retrieve_faq(question)
        
        # 3. 시스템/유저 메시지 구성
        system_prompt = config.SYSTEM_PROMPT.format(
            current_info=config.CURRENT_INFO
        )
        
        user_msg = f"질문: {question}"
        
        # 계산기 데이터 활용
        if calc_data and calc_data.get('calculated'):
            user_msg += f"\n\n[계산기 사용 데이터]"
            user_msg += f"\n- 월 평균임금: {calc_data.get('salary', '미입력')}원"
            user_msg += f"\n- 연령: {calc_data.get('age', '미입력')}세"
            user_msg += f"\n- 예상 일 급여: {calc_data.get('daily_amount', '미계산')}원"
            user_msg += f"\n- 수급 일수: {calc_data.get('days', '미계산')}일"
        
        # FAQ 있으면 참고사례로 추가
        if faqs:
            case_text = "\n\n[참고 지식]\n"
            for faq in faqs:
                case_text += f"- {faq['q']}: {faq['a']}\n"
            case_text += "\n위는 일반 원칙입니다. 사용자의 구체적 상황(근무기간, 임금, 퇴사사유)을 180일, 상한/하한액 규칙에 직접 대입하여 답변하세요."
            user_msg += case_text

        # 컨텍스트 명확화 (쿠팡플렉스, 배달 등)
        if ("하는데" in question or "인데" in question) and "실업급여" in question:
            if "받으면서" not in question and "수급" not in question:
                user_msg += "\n\n⚠️ 중요: 질문자는 현재 해당 일을 하고 있으며, 퇴직 후 실업급여 자격을 묻는 것입니다. 수급 중 부업이 아닙니다!"
                user_msg += "\n답변 구조: 1) 해당 직종의 고용보험 가입 여부 2) 퇴직 후 수급 조건"

        # 특정 케이스 강조
        if "임금체불" in question:
            user_msg += "\n\n중요: 임금체불 2개월 이상시 자진퇴사도 실업급여 가능. 이 점을 반드시 언급하세요."

        if "65세" in question or "66세" in question:
            user_msg += "\n\n중요: 65세 이전부터 계속 근무한 경우만 가능. 65세 이후 신규 고용은 제외."

        # 여러 회사 언급시 마지막 이직사유 강조
        if ("회사" in question and "후" in question) or ("퇴사" in question and "다시" in question):
            user_msg += "\n\n중요: 실업급여는 마지막 직장의 이직사유만 판단합니다. 이전 직장은 180일 계산에만 사용."

        # 알바/근로 언급시 부정수급 경고
        if "알바" in question or "일하면서" in question:
            user_msg += "\n\n중요: 실업급여 수급 중 근로는 반드시 신고. 미신고시 5배 추징."        
        
        if "다시" in question or "현재" in question or "지금" in question:
            if any(word in question for word in ["일하고", "근무하고", "활동하고", "라이더로"]):
                user_msg += "\n\n⚠️ 매우 중요: 이미 새로운 일을 시작했다면 실업 상태가 아니므로 실업급여 신청 자체가 불가능합니다!"
            
        # 4. API 호출
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=config.MAX_OUTPUT_TOKENS
        )
        
        answer = response.choices[0].message.content
        
        # 5. 답변 검증
        answer = validate_answer(answer, question)
        
        # 6. 계산 관련 질문시 링크 추가 (a태그 형식)
        if any(word in question for word in ['얼마', '금액', '계산', '월급', '하한', '상한']):
            if "sudanghelp.co.kr" not in answer:
                answer += '\n\n<a href="https://sudanghelp.co.kr/unemployment/" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>'
        
        # 7. 후처리
        answer = postprocess_answer(answer)
        
        return answer
        
    except Exception as e:
        logger.error(f"API error: {e}")
        
        # API 실패시 fallback
        if "권고사직" in question:
            return config.FALLBACK_ANSWERS.get("권고사직_사직서", "고용노동부 상담센터 1350으로 문의하세요.")
        elif any(word in question for word in ['얼마', '금액', '계산']):
            return config.CALCULATION_GUIDE
        elif "자진퇴사" in question or "자발적" in question:
            return config.FALLBACK_ANSWERS.get("자진퇴사", "고용노동부 상담센터 1350으로 문의하세요.")
        
        return "일시적 오류가 발생했습니다. 고용노동부 상담센터 1350으로 문의하세요."

def postprocess_answer(answer):
    """답변 후처리 (계산기 링크 변환 포함)"""
    # 마크다운 제거
    answer = answer.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
    
    # 계산기 URL을 클릭 가능한 형태로 변환
    # 패턴 1: "계산기: URL" 형태
    answer = re.sub(
        r'계산기:\s*(https://sudanghelp\.co\.kr/unemployment/?)',
        r'<a href="\1" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>',
        answer
    )
    
    # 패턴 2: 단순 URL (이미 a태그가 아닌 경우만)
    answer = re.sub(
        r'(?<!href=")(?<!>)(https://sudanghelp\.co\.kr/unemployment/?)(?!</a>)',
        r'<a href="\1" target="_blank" style="background:#0066ff;color:white;padding:8px 16px;border-radius:4px;text-decoration:none;display:inline-block;margin:10px 0">📊 실업급여 계산기 바로가기</a>',
        answer
    )
    
    # 중복 줄바꿈 제거
    answer = re.sub(r'\n{3,}', '\n\n', answer).strip()
    
    return answer

@app.route("/health", methods=["GET"])
def health_check():
    """헬스체크 엔드포인트"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": config.MODEL_NAME,
        "version": "2025.08.28"
    })

@app.route("/api/mark-calculator-used", methods=["POST"])
def mark_calculator_used():
    """계산기 사용 표시"""
    fingerprint = request.json.get("fingerprint")
    if fingerprint:
        is_dev = fingerprint in config.MASTER_FINGERPRINTS or config.ENVIRONMENT == "development"
        
        if not is_dev:
            keys = get_user_keys(request, fingerprint)
            
            if not check_all_limits(keys, 3):
                return jsonify({"error": "일일 제한으로 계산기 사용 불가"}), 403
            
            mark_calculator_usage(keys)
        
        resp = make_response(jsonify({"status": "ok"}))
        if not request.cookies.get('usage_token'):
            new_token = str(uuid.uuid4())
            resp.set_cookie('usage_token', new_token, max_age=86400, httponly=True, samesite='Lax')
        return resp
    
    return jsonify({"error": "fingerprint required"}), 400

@app.route("/api/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def feedback():
    """좋아요/싫어요 피드백 처리"""
    try:
        data = request.json
        feedback_type = data.get("type")  # "like" or "dislike"
        answer_hash = hashlib.md5(data.get('answer', '').encode()).hexdigest()[:16]
        
        # 싫어요인 경우 패턴 분석용 로그
        if feedback_type == "dislike":
            logger.warning(f"Dislike feedback: {data.get('question')[:100]}")
        
        # 카운트 증가
        feedback_counts[answer_hash][feedback_type] += 1
        
        # CSV 저장
        feedback_file = 'qa_logs/feedback.csv'
        file_exists = os.path.exists(feedback_file)
        
        with open(feedback_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['시간', 'ID', '타입', '질문', '답변'])
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                hashlib.md5(data.get('fingerprint', '').encode()).hexdigest()[:8],
                feedback_type,
                data.get('question', '')[:200],
                data.get('answer', '')[:200]
            ])
        
        # 현재 카운트 반환
        return jsonify({
            "status": "ok",
            "counts": {
                "like": feedback_counts[answer_hash]["like"],
                "dislike": feedback_counts[answer_hash]["dislike"]
            }
        })
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"error": "failed"}), 500

@app.route("/api/feedback/count/<answer_hash>", methods=["GET"])
def get_feedback_count(answer_hash):
    """특정 답변의 좋아요/싫어요 수 조회"""
    return jsonify({
        "like": feedback_counts[answer_hash]["like"],
        "dislike": feedback_counts[answer_hash]["dislike"]
    })

@app.route("/api/reload-faq", methods=["POST"])
def reload_faq():
    """FAQ 리로드 (선택사항)"""
    global FAQS
    FAQS = load_knowledge()
    return jsonify({"status": "reloaded", "count": len(FAQS)})

@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute")
@limiter.limit("100 per hour", key_func=get_remote_address)
def chat():
    try:
        question = request.json.get("question", "")
        fingerprint = request.json.get("fingerprint", "")
        calc_data = request.json.get("calcData")
        
        # 개발자 체크
        is_dev = fingerprint in config.MASTER_FINGERPRINTS or config.ENVIRONMENT == "development"
        
        # User-Agent 체크 (봇 방지)
        user_agent = request.headers.get('User-Agent', '')
        if not user_agent or 'bot' in user_agent.lower():
            return jsonify({"error": "접근이 차단되었습니다"}), 403
        
        # 빈 질문 체크
        if not question:
            return jsonify({"error": "질문이 없습니다"}), 400
        
        # HTML 태그 제거 (XSS 방지)
        question = bleach.clean(question, tags=[], strip=True)
        
        # 입력 길이 체크
        if not validate_input_length(question):
            return jsonify({"error": f"질문은 {config.MAX_INPUT_LENGTH}자 이내로 작성해주세요"}), 400
        
        # 악성 패턴 체크
        if not check_malicious_input(question):
            return jsonify({"error": "허용되지 않는 입력입니다"}), 400
        
        # 실업급여 관련 체크
        if not is_unemployment_related(question):
            return jsonify({
                "answer": "실업급여 관련 질문만 답변 가능합니다. 문의: 고용노동부 상담센터 1350"
            })
        
        # 개발자가 아닐 때만 제한 체크
        if not is_dev:
            keys = get_user_keys(request, fingerprint)
            
            # 계산기 사용 체크 (선택사항 - 주석처리 가능)
            # if not check_calculator_usage(keys):
            #     return jsonify({
            #         "error": "계산기를 먼저 이용해주세요",
            #         "redirect": "https://sudanghelp.co.kr/unemployment/"
            #     })
            
            # 일일 3회 제한
            if not check_all_limits(keys, 3):
                return jsonify({
                    "error": "일일 3회 초과. 내일 다시 이용하세요",
                    "remaining": 0
                })
            
            increment_all_usage(keys)
            remaining = get_remaining_count(keys)
        else:
            remaining = 999
            keys = {'primary': f"dev_{fingerprint}"}
        
        # AI로 답변 생성
        answer = generate_ai_answer(question, calc_data)
        
        # 답변 해시 생성 (피드백용)
        answer_hash = hashlib.md5(answer.encode()).hexdigest()[:16]
        
        # Q&A 저장
        save_qa_with_user(question, answer, keys['primary'])
        
        # 로깅
        logger.info({
            "action": "chat_request",
            "user_id": keys['primary'][:8],
            "is_dev": is_dev,
            "remaining": remaining
        })
        
        # 응답 생성
        resp = make_response(jsonify({
            "answer": answer,
            "answer_hash": answer_hash,
            "sources": [],
            "remaining": remaining,
            "updated": "2025-08-28"
        }))
        
        # 쿠키 설정 (없는 경우에만)
        if not request.cookies.get('usage_token'):
            new_token = str(uuid.uuid4())
            resp.set_cookie('usage_token', new_token, max_age=86400, httponly=True, samesite='Lax')
        
        return resp
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"error": "서버 오류가 발생했습니다"}), 500

# 보안 헤더 추가
@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# 등록된 라우트 확인
print("REGISTERED ROUTES:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.endpoint}: {rule.rule}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)