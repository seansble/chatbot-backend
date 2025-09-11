# backend/rag/workflow.py
"""통합 RAG 워크플로우 - 변수 추출 중심"""

from typing import Dict, List, TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
import logging
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from .unemployment_logic import unemployment_logic

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """워크플로우 상태"""
    query: str
    processed_query: str
    extracted_variables: Dict[str, Any]
    calculation_result: Dict[str, Any]
    documents: List[Dict]
    relevance_score: float
    context: str
    hit_ratio: float
    overlap_score: float
    mode: str
    raw_answer: str
    final_answer: str
    citation: str
    confidence: float
    debug_path: List[str]


class SemanticRAGWorkflow:
    """통합 RAG 워크플로우"""

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
- 400자 내외"""

    SYSTEM_PROMPT_FINAL = """당신은 친절한 실업급여 상담사입니다.

[역할]
이미 계산된 결과를 자연스럽고 친근하게 설명하기
추가 계산 절대 금지 (이미 완료됨)
제공된 숫자 그대로 사용

400자 이내, 친근한 존댓말"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """워크플로우 구성"""
        workflow = StateGraph(RAGState)

        # 노드 정의
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("extract_variables", self.extract_variables)
        workflow.add_node("calculate_benefit", self.calculate_benefit)
        workflow.add_node("rag_search", self.rag_search)
        workflow.add_node("simple_evaluate", self.simple_evaluate)
        workflow.add_node("generate_llm_only", self.generate_llm_only)
        workflow.add_node("generate_rag_lite", self.generate_rag_lite)
        workflow.add_node("generate_two_stage", self.generate_two_stage)
        workflow.add_node("format_final", self.format_final)

        # 엣지 정의
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "extract_variables")
        workflow.add_edge("extract_variables", "calculate_benefit")
        workflow.add_edge("calculate_benefit", "rag_search")
        workflow.add_edge("rag_search", "simple_evaluate")

        workflow.add_conditional_edges(
            "simple_evaluate",
            self.route_by_mode,
            {
                "llm_only": "generate_llm_only",
                "rag_lite": "generate_rag_lite",
                "two_stage": "generate_two_stage",
            },
        )

        workflow.add_edge("generate_llm_only", "format_final")
        workflow.add_edge("generate_rag_lite", "format_final")
        workflow.add_edge("generate_two_stage", "format_final")
        workflow.add_edge("format_final", END)

        return workflow.compile()

    def analyze_query(self, state: RAGState) -> RAGState:
        """쿼리 전처리"""
        query = state["query"]
        state["debug_path"] = ["analyze_query"]

        replacements = {
            "때려치": "자진퇴사",
            "짤렸": "해고",
            "잘렸": "해고",
            "얼마나": "얼마",
            "언제부터": "언제",
            "되나요": "가능",
            "받을 수 있": "수급 가능",
            "그만뒀": "퇴사",
            "그만둬": "퇴사",
        }

        processed = query
        for old, new in replacements.items():
            processed = processed.replace(old, new)

        processed = processed.replace('"', "").replace("'", "")

        state["processed_query"] = processed
        logger.info(f"Query processed: {processed[:50]}...")
        return state

    def extract_variables(self, state: RAGState) -> RAGState:
        """변수 추출 - 모든 질문에서 시도"""
        state["debug_path"].append("extract_variables")

        try:
            variables = unemployment_logic.extract_variables_with_llm(
                state["processed_query"]
            )
            state["extracted_variables"] = variables
            logger.info(f"Variables extracted: {variables}")
        except Exception as e:
            logger.error(f"Variable extraction failed: {e}")
            state["extracted_variables"] = {}

        return state

    def calculate_benefit(self, state: RAGState) -> RAGState:
        """실업급여 계산"""
        state["debug_path"].append("calculate_benefit")

        variables = state.get("extracted_variables", {})
        
        if not variables or not variables.get("eligible_months"):
            state["calculation_result"] = {}
            logger.info("No valid variables for calculation")
            return state

        try:
            result = unemployment_logic.calculate_total_benefit(variables)
            state["calculation_result"] = result
            logger.info(f"Calculation complete: eligible={result.get('eligible')}")
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            state["calculation_result"] = {}

        return state

    def rag_search(self, state: RAGState) -> RAGState:
        """RAG 검색"""
        state["debug_path"].append("rag_search")
        query = state["processed_query"]

        results = self.retriever.retrieve(query, top_k=3)

        state["documents"] = results
        state["relevance_score"] = results[0]["score"] if results else 0.0

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
        """평가 및 모드 결정 - 계산 결과 우선"""
        state["debug_path"].append("simple_evaluate")

        # 계산 결과가 있으면 무조건 two_stage
        calc_result = state.get("calculation_result", {})
        if calc_result and "eligible" in calc_result:
            state["mode"] = "two_stage"
            state["hit_ratio"] = 1.0
            state["overlap_score"] = 1.0
            logger.info("Mode: two_stage (has calculation)")
            return state

        # 계산 결과 없을 때만 RAG 점수로 판단
        try:
            from ..tokenizer import KiwiTokenizer
            tokenizer = KiwiTokenizer()
            query_tokens = set(tokenizer.tokenize(state["processed_query"]))
            context_tokens = set(tokenizer.tokenize(state.get("context", "")))
        except:
            query_tokens = set(state["processed_query"].lower().split())
            context_tokens = set(state.get("context", "").lower().split())

        important_keywords = {
            "실업급여", "권고사직", "자격", "금액", "기간", "조건",
            "반복수급", "조기재취업", "구직활동", "180일", "하한액", "상한액",
        }
        
        query_important = query_tokens & important_keywords
        context_important = context_tokens & important_keywords

        if query_important:
            state["hit_ratio"] = len(query_important & context_important) / len(query_important)
        else:
            state["hit_ratio"] = 0.0

        if query_tokens:
            state["overlap_score"] = len(query_tokens & context_tokens) / len(query_tokens)
        else:
            state["overlap_score"] = 0.0

        # 모드 결정
        if state.get("relevance_score", 0) > 0.05:
            state["mode"] = "rag_lite"
        else:
            state["mode"] = "llm_only"

        state["citation"] = ""

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

    def generate_two_stage(self, state: RAGState) -> RAGState:
        """계산 결과 + RAG 통합 생성"""
        state["debug_path"].append("generate_two_stage")

        calc_result = state.get("calculation_result", {})
        
        if not calc_result or "eligible" not in calc_result:
            logger.warning("No calculation result, falling back to rag_lite")
            return self.generate_rag_lite(state)

        formatted_result = unemployment_logic.format_calculation_result(calc_result)

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            user_prompt = f"""질문: {state['query']}

{formatted_result}

RAG 정보: {state.get('context', '')}

[지시사항]
- RAG 정보를 확인하고, 위 계산 결과가 포함되어 있지 않다면 보완하세요
- 계산 결과는 완전한 문장이므로 그대로 활용 가능
- 자연스럽게 통합하여 400자 이내로 작성

답변:"""

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT_FINAL},
                {"role": "user", "content": user_prompt},
            ]

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
                timeout=15,
            )

            raw_answer = completion.choices[0].message.content

            if len(raw_answer) > 400:
                raw_answer = self.truncate_safely(raw_answer, 400)

            state["raw_answer"] = raw_answer
            state["confidence"] = 0.95

        except Exception as e:
            logger.error(f"Two-stage generation failed: {e}")
            state["raw_answer"] = formatted_result
            state["confidence"] = 0.8

        return state

    def generate_llm_only(self, state: RAGState) -> RAGState:
        """LLM only 모드"""
        state["debug_path"].append("generate_llm_only")

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

            if len(raw_answer) > 400:
                raw_answer = self.truncate_safely(raw_answer, 400)

            state["raw_answer"] = raw_answer
            state["confidence"] = 0.7

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            state["raw_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            state["confidence"] = 0.0

        return state

    def generate_rag_lite(self, state: RAGState) -> RAGState:
        """RAG lite 모드"""
        state["debug_path"].append("generate_rag_lite")

        high_confidence = state.get("relevance_score", 0) > 0.12

        base_answer = ""
        parent_text = ""
        if state.get("documents"):
            doc = state["documents"][0]
            base_answer = doc.get("answer", "")
            parent_text = doc.get("parent_text", "")

        context = self.truncate_safely(state.get("context", ""), 400)

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            if high_confidence and base_answer:
                user_prompt = f"""[핵심 정보]
{base_answer}

[추가 컨텍스트]
{parent_text if parent_text else context}

[사용자 질문]
{state['query']}

질문의 맥락에 맞게 자연스럽게 설명하고 400자 이내로 답변:"""
                
                temp = 0.2

            else:
                user_prompt = f"""[참고 정보]
{context}

[질문]
{state['query']}

[2025년 정확한 정보]
- 상한액: 66,000원, 하한액: 64,192원
- 조건: 18개월 중 180일

전문가로서 정확한 답변 (400자 이내):"""
                
                temp = 0.3

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT_BASE},
                {"role": "user", "content": user_prompt},
            ]

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=300,
                timeout=15,
            )

            raw_answer = completion.choices[0].message.content

            if len(raw_answer) > 400:
                raw_answer = self.truncate_safely(raw_answer, 400)

            state["raw_answer"] = raw_answer
            state["confidence"] = 0.9 if high_confidence else 0.7

        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            if base_answer:
                state["raw_answer"] = f"{base_answer}\n\n자세한 내용은 고용센터 1350으로 문의하세요."
            else:
                state["raw_answer"] = "답변 생성 중 오류가 발생했습니다."
            state["confidence"] = 0.5

        return state

    def truncate_safely(self, text: str, max_len: int = 400) -> str:
        """문장 단위로 안전하게 자르기"""
        if len(text) <= max_len:
            return text

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

    def format_final(self, state: RAGState) -> RAGState:
        """최종 포맷팅"""
        state["debug_path"].append("format_final")

        answer = state.get("raw_answer", "관련 정보를 찾을 수 없습니다.")

        try:
            import config
            for wrong, correct in config.COMMON_MISTAKES.items():
                answer = answer.replace(wrong, correct)
        except:
            pass

        if len(answer) > 400:
            answer = self.truncate_safely(answer, 400)

        state["final_answer"] = answer

        logger.info(f"Workflow complete: {' → '.join(state['debug_path'])}")
        return state

    def run(self, query: str) -> Dict:
        """워크플로우 실행"""
        initial_state = {
            "query": query,
            "processed_query": "",
            "extracted_variables": {},
            "calculation_result": {},
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
                "has_calculation": bool(result.get("calculation_result")),
                "has_variables": bool(result.get("extracted_variables")),
            },
        }


# 기존 클래스 대체
RAGWorkflow = SemanticRAGWorkflow
ImprovedRAGWorkflow = SemanticRAGWorkflow