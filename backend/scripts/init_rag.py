#!/usr/bin/env python
import sys
import os

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline

def main():
    """Qdrant 초기화 스크립트"""
    print("🚀 RAG 데이터 초기화")
    pipeline = RAGPipeline()
    pipeline.run()
    print("✅ 완료!")

if __name__ == "__main__":
    main()