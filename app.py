from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import requests
from dotenv import load_dotenv
import html

# 환경변수 로드
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# CORS 설정 (기존 사이트에서 접근 허용)
CORS(app, origins=["https://sudanghelp.co.kr", "http://localhost:3000", "http://localhost:5000", "http://127.0.0.1:5000"])

# Rate Limiting (IP당 하루 10회)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per day"]
)

# Perplexity API 설정
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

@app.route('/')
def home():
    return {"message": "AI 챗봇 API 서버가 실행중입니다!", "status": "success"}

@app.route('/api/health')
def health_check():
    return {"status": "healthy", "service": "chatbot-api"}

@app.route('/api/chat', methods=['POST'])
@limiter.limit("3 per day")  # 하루 3회 제한
def chat_with_ai():
    try:
        # 요청 데이터 받기
        data = request.get_json()
        
        if not data or 'message' not in data:
            return {"error": "메시지가 필요합니다"}, 400
        
        user_message = data['message']
        calc_context = data.get('calc_context', '')
        
        # XSS 방어: HTML 이스케이프
        user_message = html.escape(user_message)
        calc_context = html.escape(calc_context)
        
        # 메시지 길이 제한
        if len(user_message) > 1000:
            return {"error": "메시지가 너무 깁니다 (1000자 이내)"}, 400
        
        # 프롬프트 생성
        prompt = create_consultation_prompt(calc_context, user_message)
        
        # Perplexity API 호출 (일단 테스트용으로 더미 응답)
        response = call_perplexity_api(prompt)
        
        if response:
            return {
                "success": True,
                "response": response,
                "remaining_calls": "API 호출 완료"
            }
        else:
            return {"error": "AI 서비스가 일시적으로 이용불가합니다"}, 503
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": "서버 오류가 발생했습니다"}, 500

def create_consultation_prompt(calc_context, user_question):
    """상담용 프롬프트 생성"""
    prompt = f"""
당신은 한국의 금융 및 세무 전문가입니다. 2025년 8월 기준 최신 정보로 답변해주세요.

📊 계산기 사용 정보:
{calc_context}

💡 답변 가이드라인:
1. 계산 결과가 정확한지 확인
2. 구체적이고 실용적인 조언 제공  
3. 최신 정책 변경사항 반영
4. 주의사항도 함께 안내
5. 출처나 근거 명시

사용자 질문: {user_question}

답변은 친근하고 이해하기 쉽게 작성해주세요.
"""
    return prompt

def call_perplexity_api(prompt):
    """Perplexity API 호출"""
    try:
        # API 키가 설정되지 않았으면 테스트 응답
        if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY == 'your_api_key_here':
            return f"[테스트 모드] 질문을 잘 받았습니다!\n\n질문: {prompt[:100]}...\n\n실제 운영시에는 Perplexity API가 답변을 제공합니다."
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(
            PERPLEXITY_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Perplexity API Error: {str(e)}")
        return None

# 에러 핸들링
@app.errorhandler(429)
def ratelimit_handler(e):
    return {
        "error": "오늘 사용 횟수를 모두 사용했습니다. 내일 다시 이용해주세요!",
        "retry_after": "내일 00시에 초기화됩니다"
    }, 429

@app.errorhandler(404)
def not_found(e):
    return {"error": "API 엔드포인트를 찾을 수 없습니다"}, 404

@app.errorhandler(500)
def server_error(e):
    return {"error": "서버 내부 오류가 발생했습니다"}, 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5009)  # ← host를 127.0.0.1로, 포트도 변경
