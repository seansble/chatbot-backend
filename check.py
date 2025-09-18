#!/usr/bin/env python
"""
형태소 분석 → BM25/Vector 검색 → Parent-Child → LLM 전달 과정 확인
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "backend"))

from backend.rag.tokenizer import KiwiTokenizer
from backend.rag.vectorstore import QdrantVectorStore
from backend.rag.embedder import BGEEmbedder
from backend.rag.retriever import RAGRetriever


def test_full_flow():
    print("=" * 80)
    print("🔍 RAG Parent-Child 전체 흐름 테스트")
    print("=" * 80)

    # 테스트 질문들
    test_queries = [
        "조기재취업수당 받으려면 어떻게 해야 해요",  # 합성어 포함
        "권고사직 증거가 없으면 어떻게 되나요",     # FAQ 변형
        "2025년 실업급여 하한액이 얼마인가요"       # 새로운 질문
    ]

    for query_idx, test_query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"📝 테스트 {query_idx}: '{test_query}'")
        print("-" * 80)

        # 1. 형태소 분석
        print("\n[1단계] Kiwi 형태소 분석")
        print("-" * 40)
        tokenizer = KiwiTokenizer()
        tokens = tokenizer.tokenize(test_query)
        print(f"원문: {test_query}")
        print(f"토큰: {tokens}")
        print(f"토큰 수: {len(tokens)}개")

        # 2. 동의어 확장
        expanded = tokenizer.expand_query(tokens)
        print(f"동의어 확장: {expanded}")

        # 3. BM25 검색
        print("\n[2단계] BM25 키워드 검색")
        print("-" * 40)
        vector_store = QdrantVectorStore()
        vector_store._load_all_documents()  # BM25 초기화

        bm25_results = vector_store.bm25_search(test_query, top_k=3)
        print(f"BM25 검색 결과: {len(bm25_results)}개")
        for i, doc in enumerate(bm25_results, 1):
            print(f"\n  [{i}] BM25 Score: {doc['score']:.3f}")
            print(f"      Text: {doc['text'][:60]}...")

        # 4. Vector 검색
        print("\n[3단계] Vector 의미 검색")
        print("-" * 40)
        embedder = BGEEmbedder()
        query_embedding = embedder.embed_query(test_query)

        vector_results = vector_store.search(query_embedding, top_k=3)
        print(f"Vector 검색 결과: {len(vector_results)}개")
        for i, doc in enumerate(vector_results, 1):
            print(f"\n  [{i}] Vector Score: {doc['score']:.3f}")
            print(f"      Text: {doc['text'][:60]}...")

        # 5. Hybrid 검색 (최종)
        print("\n[4단계] Hybrid 검색 (BM25 + Vector 결합)")
        print("-" * 40)
        hybrid_results = vector_store.hybrid_search(test_query, query_embedding, top_k=3)
        print(f"Hybrid 검색 결과: {len(hybrid_results)}개")

        for i, doc in enumerate(hybrid_results, 1):
            print(f"\n  [{i}] Hybrid Score: {doc['hybrid_score']:.3f}")
            print(f"      BM25 RRF: {doc.get('bm25_rrf', 0):.3f}")
            print(f"      Vector RRF: {doc.get('vector_rrf', 0):.3f}")
            print(f"      Text (Child): {doc['text'][:60]}...")
            if "parent_text" in doc:
                print(f"      Parent Text: {doc['parent_text'][:100]}...")

        # 6. 최종 점수 판정
        print("\n[5단계] RAG 점수 판정")
        print("-" * 40)
        relevance_score = hybrid_results[0]["hybrid_score"] if hybrid_results else 0.0
        print(f"최고 점수: {relevance_score:.3f}")
        
        if relevance_score < 0.15:
            print("❌ RAG 점수 낮음 → LLM 단독 모드로 전환")
        elif relevance_score < 0.3:
            print("⚠️ RAG Lite 모드 (점수 보통)")
        else:
            print("✅ RAG Full 모드 (점수 높음)")

        print("\n" + "="*80)


if __name__ == "__main__":
    test_full_flow()