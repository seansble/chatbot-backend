from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict
import numpy as np
import logging
from rank_bm25 import BM25Okapi
import re
import os

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, collection_name: str = "unemployment_rag"):
        """Qdrant 클라이언트 초기화"""
        # 환경변수 체크 - 디버깅 추가
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        logger.info(f"Environment check - URL: {qdrant_url[:30] if qdrant_url else 'None'}")
        logger.info(f"Environment check - Key: {'Set' if qdrant_api_key else 'None'}")
        
        if qdrant_url and qdrant_api_key:
            # Railway/Production - Qdrant Cloud
            try:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                logger.info("✅ Qdrant Cloud 모드 연결 시도")
            except Exception as e:
                logger.error(f"Qdrant Cloud 연결 실패: {e}")
                # 폴백으로 로컬 시도
                self.client = QdrantClient(host="localhost", port=6333)
        else:
            # 로컬 개발 - Docker Qdrant
            self.client = QdrantClient(host="localhost", port=6333)
            logger.info("✅ Qdrant Docker 모드 연결 성공")