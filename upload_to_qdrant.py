from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
import json
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# 환경변수에서 읽기 (안전)
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://4baad140-4533-4284-b2a0-5dabdf9918d4.us-east4-0.gcp.cloud.qdrant.io:6333",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 디버깅: API 키 확인
print(f"API Key loaded: {bool(QDRANT_API_KEY)}")
print(f"API Key length: {len(QDRANT_API_KEY) if QDRANT_API_KEY else 0}")
print(f"API Key starts with: {QDRANT_API_KEY[:10] if QDRANT_API_KEY else 'None'}")

if not QDRANT_API_KEY:
    print("❌ QDRANT_API_KEY 환경변수를 설정하세요!")
    exit(1)

# API 키 정리 (공백, 줄바꿈 제거)
QDRANT_API_KEY = QDRANT_API_KEY.strip()

# Qdrant Cloud 연결
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("✅ Qdrant 연결 성공")
except Exception as e:
    print(f"❌ Qdrant 연결 실패: {e}")
    exit(1)

# 1. 컬렉션 생성/확인
try:
    client.create_collection(
        collection_name="unemployment_rag",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("✅ 컬렉션 생성 완료")
except:
    print("컬렉션 이미 존재")

# 2. knowledge.json 읽기
with open("backend/data/knowledge.json", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

# 3. 문서를 포인트로 변환 부분 수정
points = []
for i, item in enumerate(knowledge_data.get("faqs", [])):  # faqs 배열 접근
    # 질문과 답변을 합쳐서 텍스트 생성
    text = f"질문: {item['q']}\n답변: {item['a']}"

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),  # UUID 자동 생성으로 변경
            vector=np.random.rand(1024).tolist(),
            payload={
                "text": text,
                "original_id": item["id"],  # 원래 ID는 payload에 저장
                "category": item.get("category", ""),
                "keywords": item.get("keywords", []),
                "priority": item.get("priority", 5),
                "q": item["q"],
                "a": item["a"],
                "source": "knowledge.json",
            },
        )
    )

print(f"처리할 FAQ 개수: {len(points)}")

# 4. 업로드
if points:
    client.upsert(collection_name="unemployment_rag", points=points)
    print(f"✅ {len(points)}개 문서 업로드 완료!")
else:
    print("업로드할 문서가 없습니다")

# 5. 확인
info = client.get_collection("unemployment_rag")
print(f"총 문서 개수: {info.points_count}")
