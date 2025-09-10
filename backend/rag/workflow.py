# backend/rag/workflow.py
"""통합 RAG 워크플로우 - 프롬프트 내장"""

from typing import Dict, List, TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
import logging
import re
import sys
from pathlib import Path

# config import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """워크플로우 상태"""

    # 입력
    query: str
    processed_query: str

    # 검색
    documents: List[Dict]
    relevance_score: float
    context: str

    # 평가
    hit_ratio: float
    overlap_score: float
    mode: str  # "llm_only" or "rag_lite"

    # 답변 생성
    raw_answer: str
    final_answer: str
    citation: str

    # 메타데이터
    confidence: float
    debug_path: List[str]


class SemanticRAGWorkflow:
    """통합 RAG 워크플로우"""

    # 시스템 프롬프트 통합
    SYSTEM_PROMPT_BASE = """당신은 한국 실업급여 전문 상담사입니다.

[2025년 확정 정보]
- 일 상한액: 66,000원 (절대 69,000원 아님)
- 일 하한액: 64,192원
- 가입조건: 18개월 중 180일 이상
- 지급률: 평균임금의 60%

[답변 스타일]
- 결론부터 명확히 (가능/불가능)
- 핵심은 **볼드체** 강조
- 친근한 어투로 자연스럽게
- 400자 내외
- 확실하지 않으면 "~로 알려져 있습니다" 표현"""

    SYSTEM_PROMPT_RAG = """당신은 한국 실업급여 전문 상담사입니다.

[검색 정보 활용 원칙]
1. 제공된 정보를 이해하고 분석하여 답변
2. Parent-Child 구조의 Q&A를 참고하여 맥락 파악
3. 검색 결과를 재구성하여 자연스럽게 설명
4. 여러 정보를 종합하여 포괄적 답변
5. 검색 정보에 없는 내용은 추가하지 않기

[답변 스타일]
- 단순 복붙 금지
- 질문에 맞게 정보 재구성
- 핵심은 **볼드체** 강조
- 친근한 어투
- 400자 내외"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """워크플로우 구성"""
        workflow = StateGraph(RAGState)

        # 노드 정의
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("rag_search", self.rag_search)
        workflow.add_node("simple_evaluate", self.simple_evaluate)
        workflow.add_node("generate_llm_only", self.generate_llm_only)
        workflow.add_node("generate_rag_lite", self.generate_rag_lite)
        workflow.add_node("format_final", self.format_final)

        # 엣지 정의
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "rag_search")
        workflow.add_edge("rag_search", "simple_evaluate")

        # 라우팅
        workflow.add_conditional_edges(
            "simple_evaluate",
            self.route_by_mode,
            {
                "llm_only": "generate_llm_only",
                "rag_lite": "generate_rag_lite",
            },
        )

        workflow.add_edge("generate_llm_only", "format_final")
        workflow.add_edge("generate_rag_lite", "format_final")
        workflow.add_edge("format_final", END)

        return workflow.compile()

    def analyze_query(self, state: RAGState) -> RAGState:
        """쿼리 전처리"""
        query = state["query"]
        state["debug_path"] = ["analyze_query"]

        # 구어체 정규화
        replacements = {
            "때려치": "자진퇴사",
            "짤렸": "해고",
            "얼마나": "얼마",
            "언제부터": "언제",
            "되나요": "가능",
            "받을 수 있": "수급 가능",
        }

        processed = query
        for old, new in replacements.items():
            processed = processed.replace(old, new)

        state["processed_query"] = processed
        logger.info(f"Query processed: {processed[:50]}...")
        return state

    def rag_search(self, state: RAGState) -> RAGState:
        """RAG 검색"""
        state["debug_path"].append("rag_search")
        query = state["processed_query"]

        # RAG retriever 호출
        results = self.retriever.retrieve(query, top_k=3)

        state["documents"] = results
        state["relevance_score"] = results[0]["score"] if results else 0.0

        # Parent-Child: parent_text 우선 사용
        context_parts = []
        for i, doc in enumerate(results[:2], 1):
            parent = doc.get("parent_text", doc["text"])
            context_parts.append(f"[정보 {i}]\n{parent}")

        state["context"] = "\n\n".join(context_parts)

        logger.info(
            f"Retrieved {len(results)} documents, top score: {state['relevance_score']:.3f}"
        )
        return state

    def simple_evaluate(self, state: RAGState) -> RAGState:
        """키워드 매칭 평가 개선"""
        state["debug_path"].append("simple_evaluate")

        # 디버그 로깅 추가
        logger.info(
            f"RAG scores - rel: {state.get('relevance_score', 0):.3f}, "
            f"overlap: {state.get('overlap_score', 0):.3f}, "
            f"hit: {state.get('hit_ratio', 0):.3f}"
        )

        # Kiwi 토크나이저 사용
        try:
            from ..tokenizer import KiwiTokenizer

            tokenizer = KiwiTokenizer()
            query_tokens = set(tokenizer.tokenize(state["processed_query"]))
            context_tokens = set(tokenizer.tokenize(state.get("context", "")))
        except:
            # 폴백: 단순 split
            query_tokens = set(state["processed_query"].lower().split())
            context_tokens = set(state.get("context", "").lower().split())

        # 중요 키워드 체크
        important_keywords = {
            "실업급여",
            "권고사직",
            "자격",
            "금액",
            "기간",
            "조건",
            "반복수급",
            "조기재취업",
            "구직활동",
            "180일",
            "하한액",
            "상한액",
        }
        query_important = query_tokens & important_keywords
        context_important = context_tokens & important_keywords

        # hit_ratio 계산
        if query_important:
            state["hit_ratio"] = len(query_important & context_important) / len(
                query_important
            )
        else:
            state["hit_ratio"] = 0.0

        # overlap_score 계산
        if query_tokens:
            state["overlap_score"] = len(query_tokens & context_tokens) / len(
                query_tokens
            )
        else:
            state["overlap_score"] = 0.0

        # 직접 답변 가능 체크
        if (
            state.get("relevance_score", 0) > 0.15
            and state.get("documents")
            and "answer" in state["documents"][0]
        ):
            state["mode"] = "rag_lite"
            logger.info("Direct answer available in document")
        elif (
            state["hit_ratio"] >= 0.1
            or state["overlap_score"] >= 0.2
            or state["relevance_score"] >= 0.05
        ):
            state["mode"] = "rag_lite"
        else:
            state["mode"] = "llm_only"

        state["citation"] = ""  # 근거 표시 제거

        logger.info(
            f"Evaluation - hit_ratio: {state['hit_ratio']:.2f}, "
            f"overlap: {state['overlap_score']:.2f}, "
            f"relevance: {state['relevance_score']:.3f}, "
            f"mode: {state['mode']}"
        )
        return state

    def route_by_mode(self, state: RAGState) -> str:
        """모드에 따라 라우팅"""
        return state["mode"]

    def generate_llm_only(self, state: RAGState) -> RAGState:
        """LLM only 모드"""
        state["debug_path"].append("generate_llm_only")

        # FALLBACK_ANSWERS 체크
        try:
            import config

            query_lower = state["query"].lower()

            for keyword, answer in config.FALLBACK_ANSWERS.items():
                if keyword in query_lower:
                    state["raw_answer"] = answer
                    state["confidence"] = 0.75
                    return state
        except:
            pass

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # 결론 우선 프롬프트
            user_prompt = f"""질문: {state['query']}

[답변 작성 규칙]
1. 첫 문장: 핵심 답변 (가능/불가능/금액)
2. 둘째 문장: 주요 조건이나 근거
3. 셋째 문장: 주의사항
4. 반드시 400자 이내로 완결

답변:"""

            messages = [
                {"role": "system", "content": "실업급여 전문가. 간결하고 명확하게."},
                {"role": "user", "content": user_prompt},
            ]

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=config.MODEL_TEMPERATURE,
                max_tokens=300,
                stop=["\n\n\n", "이상입니다", "참고하세요"],
                timeout=10,
            )

            raw_answer = completion.choices[0].message.content

            # 강제 길이 제한
            if len(raw_answer) > 400:
                sentences = raw_answer.split(". ")
                result = []
                current_len = 0
                for sent in sentences:
                    if current_len + len(sent) < 380:
                        result.append(sent)
                        current_len += len(sent)
                    else:
                        break
                raw_answer = ". ".join(result)
                if not raw_answer.endswith("."):
                    raw_answer += "."

            state["raw_answer"] = raw_answer
            state["confidence"] = 0.7

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            state["raw_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            state["confidence"] = 0.0

        return state

    def truncate_safely(self, text: str, max_len: int = 1000) -> str:
        """문장 단위로 안전하게 자르기"""
        if len(text) <= max_len:
            return text

        # 문장 단위로 분리
        sentences = text.split(". ")
        result = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) + 2 <= max_len:
                result.append(sent)
                current_len += len(sent) + 2
            else:
                break

        return ". ".join(result) + "." if result else text[:max_len]

    def generate_rag_lite(self, state: RAGState) -> RAGState:
        """RAG lite 모드 - 정보 보완 전문가"""
        state["debug_path"].append("generate_rag_lite")

        # RAG 품질 판단
        high_confidence = state.get("relevance_score", 0) > 0.12

        # 기본 정보 추출
        base_answer = ""
        if state["documents"]:
            doc = state["documents"][0]
            base_answer = doc.get("answer", "")
            parent_text = doc.get("parent_text", "")

        # Context 준비
        context = self.truncate_safely(state.get("context", ""), 400)

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            if high_confidence and base_answer:
                # 높은 품질: 정보 보완자 역할
                user_prompt = f"""[핵심 정보 - 정확함]
    {base_answer}

    [추가 컨텍스트]
    {parent_text if parent_text else context}

    [사용자 질문]
    {state['query']}

    [당신의 역할: 정보 보완 전문가]
    1. 핵심 정보는 100% 정확하므로 그대로 사용
    2. 질문의 맥락에 맞게 자연스럽게 설명 추가
    3. 부족한 부분만 보충하되 각 문장이 자연스럽게 이어지게 해줘
    4. 400자 이내로 완성 

    답변:"""
                system_msg = (
                    "실업급여 전문 상담사. 제공된 핵심 정보를 바탕으로 친절하게 설명."
                )
                temp = 0.2

            else:
                # 낮은 품질: 참고만 하고 전문가 답변
                user_prompt = f"""[참고 정보 - 신뢰도 낮음]
    {context}

    [질문]
    {state['query']}

    [당신의 역할: 실업급여 전문가]
    1. 주어진 질문의 맥락에 맞게 자연스럽게 설명
    2. 각 문장이 자연스럽게 이어지게 해주되 간결하게
    3. 400자 이내로 완성

    [2025년 정확한 정보]
    - 반복수급: 3회 10%, 4회 25%, 5회 40%, 6회 이상 50%
    - 구직활동: 4주마다 1회 이상
    - 상한액: 66,000원, 하한액: 64,192원
    - 조건: 18개월 중 180일

    전문가로서 정확한 답변:"""
                system_msg = "실업급여 전문가. 2025년 최신 정보로 정확히."
                temp = 0.3

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ]

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=300,
                stop=["\n\n\n", "이상입니다"],
                timeout=15,
            )

            raw_answer = completion.choices[0].message.content

            # 길이 체크
            if len(raw_answer) > 400:
                sentences = raw_answer.split(". ")
                result = []
                current_len = 0
                for sent in sentences:
                    if current_len + len(sent) < 380:
                        result.append(sent)
                        current_len += len(sent)
                    else:
                        break
                raw_answer = ". ".join(result) + "."

            state["raw_answer"] = raw_answer
            state["confidence"] = 0.9 if high_confidence else 0.7

        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            # 폴백: 직접 답변 사용
            if base_answer:
                state["raw_answer"] = (
                    f"{base_answer}\n\n자세한 내용은 고용센터 1350으로 문의하세요."
                )
            else:
                state["raw_answer"] = "답변 생성 중 오류가 발생했습니다."
            state["confidence"] = 0.5

        return state

    def format_final(self, state: RAGState) -> RAGState:
        """최종 포맷팅"""
        state["debug_path"].append("format_final")

        answer = state.get("raw_answer", "관련 정보를 찾을 수 없습니다.")

        # 후처리: 틀린 정보 교정
        try:
            import config

            for wrong, correct in config.COMMON_MISTAKES.items():
                answer = answer.replace(wrong, correct)
        except:
            pass

        # 최종 길이 체크 (400자)
        if len(answer) > 400:
            # 마지막 온전한 문장까지만
            sentences = answer.split(". ")
            result = []
            length = 0
            for sent in sentences:
                if length + len(sent) <= 380:
                    result.append(sent)
                    length += len(sent)
                else:
                    break
            answer = ". ".join(result) + "."

        # citation 제거 (근거: 없음 안 붙임)
        state["final_answer"] = answer

        logger.info(f"Workflow complete: {' → '.join(state['debug_path'])}")
        return state

    def run(self, query: str) -> Dict:
        """워크플로우 실행"""
        initial_state = {
            "query": query,
            "processed_query": "",
            "documents": [],
            "relevance_score": 0.0,
            "context": "",
            "hit_ratio": 0.0,
            "overlap_score": 0.0,
            "mode": "",
            "raw_answer": "",
            "final_answer": "",
            "citation": "",
            "confidence": 0.0,
            "debug_path": [],
        }

        result = self.workflow.invoke(initial_state)

        return {
            "answer": result.get("final_answer", ""),
            "confidence": result.get("confidence", 0.0),
            "method": result.get("mode", "unknown"),
            "coverage_score": result.get("overlap_score", 0.0),
            "documents": result.get("documents", []),
            "debug": {
                "path": result.get("debug_path", []),
                "hit_ratio": result.get("hit_ratio", 0.0),
                "overlap_score": result.get("overlap_score", 0.0),
                "relevance_score": result.get("relevance_score", 0.0),
            },
        }


# 기존 클래스 대체
RAGWorkflow = SemanticRAGWorkflow
ImprovedRAGWorkflow = SemanticRAGWorkflow
