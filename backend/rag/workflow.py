"""LangGraph RAG 워크플로우 - LLM 직접 평가"""

from typing import Dict, List, TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
import logging
import re
import json

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """워크플로우 상태"""

    # 입력
    query: str

    # 전처리
    processed_query: str
    query_type: str

    # 검색
    documents: List[Dict]
    relevance_score: float
    context: str

    # 평가
    coverage_details: Dict[str, Any]
    coverage_score: float
    missing_parts: List[str]

    # 답변 생성
    answer_method: str
    raw_answer: str
    final_answer: str

    # 메타데이터
    confidence: float
    debug_path: List[str]


class SemanticRAGWorkflow:
    def __init__(self, retriever):
        self.retriever = retriever
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """워크플로우 구성"""
        workflow = StateGraph(RAGState)

        # 노드 정의
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("rag_search", self.rag_search)
        workflow.add_node("llm_evaluate_coverage", self.llm_evaluate_coverage)
        workflow.add_node("generate_from_rag", self.generate_from_rag)
        workflow.add_node("enhance_missing", self.enhance_missing)
        workflow.add_node("regenerate_full", self.regenerate_full)
        workflow.add_node("format_final", self.format_final)

        # 엣지 정의
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "rag_search")
        workflow.add_edge("rag_search", "llm_evaluate_coverage")

        # 평가 후 라우팅
        workflow.add_conditional_edges(
            "llm_evaluate_coverage",
            self.route_by_coverage,
            {
                "complete": "generate_from_rag",
                "partial": "enhance_missing",
                "insufficient": "regenerate_full",
            },
        )

        workflow.add_edge("generate_from_rag", "format_final")
        workflow.add_edge("enhance_missing", "format_final")
        workflow.add_edge("regenerate_full", "format_final")
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
        state["query_type"] = self._classify_query_type(processed)

        logger.info(f"Query processed: {processed[:50]}...")
        return state

    def rag_search(self, state: RAGState) -> RAGState:
        """RAG 검색"""
        state["debug_path"].append("rag_search")
        query = state["processed_query"]

        # 하이브리드 검색
        results = self.retriever.retrieve(query, top_k=5)

        state["documents"] = results
        state["relevance_score"] = results[0]["score"] if results else 0.0
        state["context"] = "\n".join([doc["text"] for doc in results[:3]])

        logger.info(f"Retrieved {len(results)} documents")
        return state

    def llm_evaluate_coverage(self, state: RAGState) -> RAGState:
        """LLM을 사용한 커버리지 평가"""
        state["debug_path"].append("llm_evaluate")

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # RAG 결과로 질문에 답변 가능한지 평가
            prompt = f"""[질문] {state['query']}

[RAG 검색 결과]
{state['context']}

다음 기준으로 평가하세요:
1. 질문의 핵심 요소(금액, 조건, 기간 등)가 RAG에 모두 포함되어 있는가?
2. RAG 정보가 최신(2025년 기준)인가?
3. RAG와 충돌할 수 있는 일반 상식 정보가 필요한가?

답변 형식 (JSON):
{{
 "coverage_score": 0.0~1.0,
 "missing_parts": ["부분1", "부분2"],
 "has_conflict": false,
 "intent": "질문 의도"
}}"""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            response_text = completion.choices[0].message.content.strip()

            # JSON 파싱 시도
            try:
                eval_result = json.loads(response_text)
                state["coverage_score"] = eval_result.get("coverage_score", 0.5)
                state["missing_parts"] = eval_result.get(
                    "missing_parts", []
                )  # 이 줄 추가 필요
                state["coverage_details"] = eval_result  # 이 줄도 추가 필요

            except:
                # JSON 파싱 실패시 relevance_score 기반 추정
                if state.get("relevance_score", 0) > 0.7:
                    state["coverage_score"] = 0.7  # RAG 검색 점수 높으면 높게
                elif state.get("relevance_score", 0) > 0.4:
                    state["coverage_score"] = 0.5  # 중간
                else:
                    state["coverage_score"] = 0.3  # 낮게
                state["missing_parts"] = []
                state["coverage_details"] = {"intent": "unknown"}

            logger.info(f"Coverage score: {state['coverage_score']}")

        except Exception as e:
            logger.error(f"Coverage evaluation failed: {str(e)}")
            # 에러시 기본값
            state["coverage_score"] = 0.5
            state["coverage_details"] = {"intent": "unknown"}
            state["missing_parts"] = []

        return state

    def route_by_coverage(self, state: RAGState) -> str:
        """커버리지 점수에 따라 라우팅"""
        score = state.get("coverage_score", 0.0)

        if score >= 0.7:
            return "complete"  # RAG로 충분
        elif score >= 0.3:
            return "partial"  # LLM 보완 필요
        else:
            return "insufficient"  # 전체 재생성

    def _extract_key_facts(self, context: str) -> Dict[str, Any]:
        facts = {}

        # 더 정확한 패턴 매칭
        if re.search(r"18개월[\s]*중[\s]*180일", context):
            facts["min_days"] = "180일"
            facts["period"] = "18개월"
        elif re.search(r"(?<!\d)180일(?!\d)", context):
            facts["min_days"] = "180일"

        # 금액 정확히 추출
        if match := re.search(r"(?<!\d)66,?000원", context):
            facts["daily_max"] = "66,000원"
        if match := re.search(r"(?<!\d)64,?192원", context):
            facts["daily_min"] = "64,192원"
        if "1년 이내" in context or "12개월 이내" in context:
            facts["claim_period"] = "1년 이내"
        if any(word in context for word in ["권고사직", "해고", "계약만료"]):
            facts["eligible_reasons"] = "권고사직, 해고, 계약만료 등"

        return facts

    def generate_from_rag(self, state: RAGState) -> RAGState:
        """RAG 결과만으로 답변 생성"""
        state["debug_path"].append("generate_from_rag")

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # RAG 정보 구조화
            facts = self._extract_key_facts(state["context"])

            prompt = f"""[공식 정보 - 절대 변경 금지]
{json.dumps(facts, ensure_ascii=False, indent=2)}

[사용자 질문]
{state['query']}

[지침]
1. 위 공식 정보만 사용하세요
2. 정보를 왜곡하거나 추가하지 마세요
3. 친절하고 자연스러운 말투로 200자 내외
4. 이모지 1개만 사용
5. "예상", "아마" 같은 모호한 표현 금지

답변:"""

            completion = client.chat.completions.create(
                model=config.EVAL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 한국 실업급여 전문 상담사입니다. 제공된 공식 정보만을 정확히 전달하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=400,
            )

            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "rag_complete"
            state["confidence"] = 0.9

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            # 폴백
            state["raw_answer"] = self._extract_relevant_answer(
                state["query"], state["documents"]
            )
            state["answer_method"] = "rag_direct"
            state["confidence"] = 0.7

        return state

    def enhance_missing(self, state: RAGState) -> RAGState:
        """RAG 정보 기반 + LLM 보완"""
        state["debug_path"].append("enhance_missing")

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # RAG 정보 구조화
            facts = self._extract_key_facts(state["context"])

            prompt = f"""[공식 정보 - 절대 변경 금지]
{json.dumps(facts, ensure_ascii=False, indent=2)}

[절대 금지 정보]
- "24개월 중 18개월" ❌ → "18개월 중 180일" ✅
- "24개월 중 12개월" ❌ → "18개월 중 180일" ✅
- "68,000원" ❌ → "66,000원 (2025년)" ✅
- "63,816원" ❌ → "64,192원 (2025년)" ✅
- "2년간 지급" ❌ → "최대 240일" ✅
- "매월 지급" ❌ → "4주마다 인정" ✅
- "180일 이상 근무" ❌ → "180일 이상 가입" ✅

[사용자 질문]
{state['query']}

[지침]
1. 위 [공식 정보]는 반드시 정확히 사용
2. [절대 금지 정보]는 절대 포함하지 마세요
3. 공식 정보에 없는 부분만 "일반적으로" 보완
4. 200-300자, 이모지 1-2개
5. 친근하고 정확하게

답변:"""

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 한국 실업급여 전문 상담사입니다.
제공된 공식 정보는 절대 변경하지 마세요.
특히 숫자, 기간, 조건은 정확히 유지하세요.""",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )

            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "enhanced"
            state["confidence"] = 0.85

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            # 폴백
            state["raw_answer"] = self._extract_relevant_answer(
                state["query"], state["documents"]
            )
            state["answer_method"] = "error"
            state["confidence"] = 0.5

        return state

    def regenerate_full(self, state: RAGState) -> RAGState:
        state["debug_path"].append("regenerate_full")

        # FALLBACK_ANSWERS 먼저 체크
        try:
            import config

            query_lower = state["query"].lower()

            for keyword, answer in config.FALLBACK_ANSWERS.items():
                if keyword in query_lower:
                    state["raw_answer"] = (
                        f"{answer}\n\n※ 일반적인 기준으로 안내드립니다."
                    )
                    state["answer_method"] = "fallback"
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

            prompt = f"""[사용자 질문]
{state['query']}

[부족한 RAG 정보]
{state['context'] if state['context'] else '관련 정보 없음'}

[지침]
1. 2025년 실업급여 기준으로 답변
2. 다음 형식으로 답변:
  - 핵심 답변 (일반적 기준)
  - "※ 참고: 일반적인 기준으로 안내드립니다."
  - "정확한 상담은 고용센터 1350으로 문의하세요."
3. 200-300자, 이모지 1개

[절대 금지 정보]
- "24개월 중 18개월" ❌
- "68,000원" ❌

답변:"""

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 한국 실업급여 전문 상담사입니다. 정확한 정보가 부족할 때는 일반적 기준을 안내하되, 반드시 고용센터 문의를 권하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            state["raw_answer"] = completion.choices[0].message.content
            state["answer_method"] = "regenerated"
            state["confidence"] = 0.7

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            state[
                "raw_answer"
            ] = """죄송합니다. 답변 생성 중 오류가 발생했습니다.
고용센터 1350으로 직접 문의해주세요."""
            state["answer_method"] = "error"
            state["confidence"] = 0.0

        return state

    def format_final(self, state: RAGState) -> RAGState:
        """최종 포맷팅"""
        state["debug_path"].append("format_final")

        answer = state.get("raw_answer", "관련 정보를 찾을 수 없습니다.")

        # 답변 끝에 메타 정보 추가 (개발용)
        if logger.level == logging.DEBUG:
            debug_info = f"\n\n[디버그: 충족도 {state.get('coverage_score', 0):.1%}, 방법: {state.get('answer_method', 'unknown')}]"
            answer += debug_info

        state["final_answer"] = answer

        # coverage_score 보정
        if state.get("coverage_score") == 0 and state.get("answer_method") != "error":
            logger.warning("Coverage score was 0, using confidence instead")
            state["coverage_score"] = state.get("confidence", 0.5)

        logger.info(f"Workflow complete: {' → '.join(state['debug_path'])}")
        return state

    def _classify_query_type(self, query: str) -> str:
        """질문 유형 분류"""
        if any(word in query for word in ["얼마", "금액", "계산"]):
            return "calculation"
        elif any(word in query for word in ["자격", "조건", "가능"]):
            return "qualification"
        elif any(word in query for word in ["어떻게", "방법", "절차"]):
            return "procedure"
        else:
            return "general"

    def _extract_relevant_answer(self, query: str, documents: List[Dict]) -> str:
        """문서에서 관련 답변 추출"""
        query_lower = query.lower()

        # 키워드 매칭으로 가장 관련된 답변 찾기
        for doc in documents[:2]:
            text = doc.get("text", "")

            # "질문:" "답변:" 패턴으로 분리
            if "답변:" in text:
                parts = text.split("질문:")
                for part in parts:
                    # 질문의 키워드가 포함된 부분 찾기
                    if any(
                        keyword in query_lower for keyword in ["조건", "자격", "가능"]
                    ):
                        if any(
                            keyword in part.lower()
                            for keyword in ["조건", "자격", "가능"]
                        ):
                            if "답변:" in part:
                                answer = (
                                    part.split("답변:")[1].split("질문:")[0].strip()
                                )
                                return answer

                # 첫 번째 답변 반환 (폴백)
                if "답변:" in text:
                    answer = text.split("답변:")[1].split("질문:")[0].strip()
                    return answer

        # 기본 답변
        return """실업급여 수급을 위한 기본 조건:
1. 이직일 이전 18개월간 고용보험 가입기간 180일 이상
2. 비자발적 퇴직 (권고사직, 계약만료 등)
3. 근로 의사와 능력이 있고 적극적 구직활동 중
4. 이직 후 12개월 이내 신청

자세한 상담은 고용센터 1350으로 문의하세요."""

    def run(self, query: str) -> Dict:
        """워크플로우 실행"""
        initial_state = {
            "query": query,
            "processed_query": "",
            "query_type": "",
            "documents": [],
            "relevance_score": 0.0,
            "context": "",
            "coverage_details": {},
            "coverage_score": 0.0,
            "missing_parts": [],
            "answer_method": "",
            "raw_answer": "",
            "final_answer": "",
            "confidence": 0.0,
            "debug_path": [],
        }

        result = self.workflow.invoke(initial_state)

        return {
            "answer": result.get("final_answer", ""),
            "confidence": result.get("confidence", 0.0),
            "method": result.get("answer_method", "unknown"),
            "coverage_score": result.get("coverage_score", 0.0),
            "documents": result.get("documents", []),
            "debug": {
                "path": result.get("debug_path", []),
                "coverage_details": result.get("coverage_details", {}),
                "missing_parts": result.get("missing_parts", []),
            },
        }


# 기존 클래스 대체
RAGWorkflow = SemanticRAGWorkflow
ImprovedRAGWorkflow = SemanticRAGWorkflow
