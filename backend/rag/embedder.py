from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGEEmbedder:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        """BGE-M3 임베딩 모델 초기화"""
        logger.info(f"Loading BGE-M3 model: {model_name}")
        logger.info("첫 실행시 모델 다운로드로 5-10분 걸릴 수 있습니다...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ BGE-M3 모델 로드 완료! 차원: {self.dimension}")
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """문서 임베딩"""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """질의 임베딩"""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding[0]