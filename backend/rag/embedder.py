# backend/rag/embedder.py
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGEEmbedder:
   def __init__(self, model_name: str = 'BAAI/bge-m3'):
       """더미 임베더 - Qdrant Cloud 사용시 불필요"""
       logger.info(f"Dummy BGE embedder initialized (model: {model_name})")
       logger.info("Using Qdrant Cloud - no local embedding needed")
       self.dimension = 1024  # BGE-M3 기본 차원
       logger.info(f"✅ Embedder ready! 차원: {self.dimension}")
       
   def embed_documents(self, texts: List[str]) -> np.ndarray:
       """문서 임베딩 - 더미 구현"""
       if not texts:
           return np.array([])
       
       # 랜덤 벡터 반환 (실제로는 사용 안 함)
       embeddings = np.random.random((len(texts), self.dimension))
       return embeddings
   
   def embed_query(self, query: str) -> np.ndarray:
       """질의 임베딩 - 더미 구현"""
       # 랜덤 벡터 반환 (실제로는 사용 안 함)
       embedding = np.random.random(self.dimension)
       return embedding