# backend/test_complex_queries.py
import os
import sys

sys.path.append(".")
os.environ["USE_REAL_EMBEDDING"] = "true"

from rag.vectorstore import QdrantVectorStore
from rag.embedder import BGEEmbedder

# 초기화
print("🔧 시스템 초기화...")
embedder = BGEEmbedder()
vector_store = QdrantVectorStore()

# 복잡한 테스트 질문들
complex_queries = [
    # 1. 복합 조건
    "주3일 알바 8개월 했는데 일수는 180일 넘는데 시간이 부족하면 어떻게 되나요?",
    # 2. 시나리오 기반
    "상사가 매일 욕하고 때려서 병원 진단서 받고 녹음도 했는데 회사는 자진퇴사 처리하려고 해요",
    # 3. 순차적 상황
    "육아휴직 1년 쓰고 복직했다가 2개월 뒤에 또 임신해서 출산휴가 쓰고 싶은데 실업급여는?",
    # 4. 예외 케이스
    "프리랜서로 3년 일하다가 정규직 전환 후 4개월만에 회사 망했어요",
    # 5. 복수 조건 충족
    "65세 넘었는데 장애인이고 최저임금 받으며 계약직인데 실업급여 얼마나 받을 수 있나?",
    # 6. 모호한 질문
    "저번달에 그만뒀는데 지금 신청하면 늦나요? 벌써 한달 지났는데...",
    # 7. 계산 필요
    "월 350만원 받다가 퇴사했는데 야근수당 빼면 기본급 250인데 실업급여는 뭘로 계산?",
    # 8. 중복 상황
    "권고사직 하면서 회사가 자진퇴사로 써달래요. 녹음은 있는데 문서는 없어요",
    # 9. 특수 직종
    "배민 라이더 2년했고 쿠팡플렉스도 같이했는데 둘 다 그만두면 실업급여 되나요?",
    # 10. 혼합 질문
    "2025년 1월에 4번째 실업급여인데 나이 60살이고 장애인이면 감액 안되나요?",
]

print("\n" + "=" * 70)
print("복잡도 높은 RAG 테스트")
print("=" * 70)

for i, query in enumerate(complex_queries, 1):
    print(f"\n[질문 {i}] {query}")
    print("-" * 70)

    # 하이브리드 검색
    query_embedding = embedder.embed_query(query)
    results = vector_store.hybrid_search(query, query_embedding, top_k=5)

    print(f"검색 결과 요약:")
    for j, result in enumerate(results[:3], 1):
        print(f"  {j}. Score: {result.get('hybrid_score', 0):.3f}")
        print(
            f"     BM25: {result.get('bm25_rrf', 0):.3f}, Vector: {result.get('vector_rrf', 0):.3f}"
        )
        print(f"     내용: {result['text'][:60]}...")

    # 상위 1위와 5위의 점수 차이 분석
    if len(results) >= 5:
        score_gap = results[0]["hybrid_score"] - results[4]["hybrid_score"]
        print(f"\n  📊 1위-5위 점수 차: {score_gap:.3f}")
        if score_gap < 0.05:
            print(f"  ⚠️ 점수 차이 작음 - 불확실한 검색 결과")
        elif score_gap > 0.1:
            print(f"  ✅ 명확한 관련 문서 찾음")

print("\n" + "=" * 70)
print("✅ 복잡도 테스트 완료!")
