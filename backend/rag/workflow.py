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
        """RAG 검색 - 우리 시스템 사용"""
        state["debug_path"].append("rag_search")
        query = state["processed_query"]

        # 우리 RAG retriever 호출
        results = self.retriever.retrieve(query, top_k=5)

        state["documents"] = results
        state["relevance_score"] = results[0]["score"] if results else 0.0
        state["context"] = "\n".join([doc["text"] for doc in results[:3]])

        logger.info(
            f"Retrieved {len(results)} documents, top score: {state['relevance_score']:.3f}"
        )
        return state

    def llm_evaluate_coverage(self, state: RAGState) -> RAGState:
        """LLM을 사용한 커버리지 평가 - 우리 RAG 점수 기준 반영"""
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
                temperature=0.1,  # 평가는 일관성이 중요
                max_tokens=200,
            )

            response_text = completion.choices[0].message.content.strip()

            # JSON 파싱 시도
            try:
                eval_result = json.loads(response_text)
                state["coverage_score"] = eval_result.get("coverage_score", 0.5)
                state["missing_parts"] = eval_result.get("missing_parts", [])
                state["coverage_details"] = eval_result

            except:
                # JSON 파싱 실패시 우리 RAG 점수 기준으로 추정 (수정된 로직)
                relevance = state.get("relevance_score", 0)

                # 더 세밀한 구간 설정
                if relevance > 0.20:
                    state["coverage_score"] = 0.7
                elif relevance > 0.15:
                    state["coverage_score"] = 0.5
                else:
                    state["coverage_score"] = 0.2  # 대부분 insufficient로

                state["missing_parts"] = []
                state["coverage_details"] = {
                    "intent": "unknown",
                    "relevance_based": True,
                }

            # 평가 디버깅 로그 추가
            logger.info(
                f"""
[평가 디버그]
- 질문: {state['query'][:50]}
- RAG 점수: {state.get('relevance_score', 0):.3f}
- Coverage: {state.get('coverage_score', 0):.2f}
- 경로: {self.route_by_coverage(state)}
- RAG 문서 수: {len(state.get('documents', []))}
"""
            )

        except Exception as e:
            logger.error(f"Coverage evaluation failed: {str(e)}")
            # 에러시 relevance_score 기반 판단 (수정된 로직)
            relevance = state.get("relevance_score", 0)
            if relevance > 0.20:
                state["coverage_score"] = 0.7
            elif relevance > 0.15:
                state["coverage_score"] = 0.5
            else:
                state["coverage_score"] = 0.2

            state["coverage_details"] = {"intent": "unknown", "error": True}
            state["missing_parts"] = []

        return state

    def route_by_coverage(self, state: RAGState) -> str:
        """커버리지 점수에 따라 라우팅 (수정된 기준)"""
        score = state.get("coverage_score", 0.0)

        if score >= 0.8:  # 0.7 → 0.8 (더 엄격)
            return "complete"  # RAG로 충분
        elif score >= 0.4:  # 0.3 → 0.4
            return "partial"  # LLM 보완 필요
        else:
            return "insufficient"  # 전체 재생성 (LLM 주로 사용)

    def _extract_key_facts(self, context: str) -> Dict[str, Any]:
        """RAG 컨텍스트에서 핵심 사실 추출"""
        facts = {}

        # 더 정확한 패턴 매칭
        if re.search(r"18개월[\s]*중[\s]*180일", context):
            facts["min_days"] = "180일"
            facts["period"] = "18개월"
        elif re.search(r"(?<!\d)180일(?!\d)", context):
            facts["min_days"] = "180일"

        # 금액 정확히 추출
        if re.search(r"(?<!\d)66,?000원", context):
            facts["daily_max"] = "66,000원"
        if re.search(r"(?<!\d)64,?192원", context):
            facts["daily_min"] = "64,192원"
        if "1년 이내" in context or "12개월 이내" in context:
            facts["claim_period"] = "1년 이내"
        if any(word in context for word in ["권고사직", "해고", "계약만료"]):
            facts["eligible_reasons"] = "권고사직, 해고, 계약만료 등"

        return facts

    def generate_from_rag(self, state: RAGState) -> RAGState:
        """RAG 결과만으로 답변 생성 - RAG 내용 절대 우선"""
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

            prompt = f"""[RAG 검색 결과]
{state['context']}

[추출된 핵심 정보]
{json.dumps(facts, ensure_ascii=False, indent=2)}

[사용자 질문]
{state['query']}

[중요 지침]
1. RAG 검색 결과의 숫자, 조건, 기간은 절대 변경 금지
2. RAG와 충돌하는 정보가 있다면 무조건 RAG 우선
3. 친절하고 자연스러운 말투로 답변
4. 답변은 500자 이내로 작성하고, 넘으면 핵심만 요약하세요
5. 이모지 1개만 사용

답변:"""

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 한국 실업급여 전문 상담사입니다. RAG 정보를 정확히 전달하는 것이 최우선입니다.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # 일관성 우선
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
        """RAG 정보 기반 + LLM 보완 - 계층형 프롬프트 적용"""
        state["debug_path"].append("enhance_missing")

        try:
            from openai import OpenAI
            import config

            client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=config.OPENROUTER_API_KEY,
            )

            # 1. 기본 시스템 프롬프트
            base_system = config.BASE_SYSTEM_PROMPT

            # 2. 질문 분석
            query = state["query"]
            needs_calculation = "얼마" in query or "금액" in query
            needs_eligibility = "조건" in query or "자격" in query
            needs_period = "기간" in query or "언제" in query

            # 3. 필요한 정보만 동적 주입
            dynamic_context = []

            if needs_calculation:
                facts = config.UNEMPLOYMENT_FACTS["amounts"]
                dynamic_context.append(
                    f"""[금액 정보]
- 일 상한액: {facts['daily_max']}
- 일 하한액: {facts['daily_min']}
- 지급률: {facts['rate']}"""
                )

            if needs_eligibility:
                facts = config.UNEMPLOYMENT_FACTS["eligibility"]
                dynamic_context.append(
                    f"""[자격 조건]
- {facts['insurance_period']}
- {facts['resignation_type']}"""
                )

            if needs_period:
                facts = config.UNEMPLOYMENT_FACTS["periods"]
                dynamic_context.append(
                    f"""[수급 기간]
- 50세 미만: {facts['30_to_50']}
- 50세 이상: {facts['50_plus']}"""
                )

            # 4. RAG 정보 구조화
            rag_facts = self._extract_key_facts(state["context"])

            # 5. 메시지 구성
            messages = [{"role": "system", "content": base_system}]

            if dynamic_context:
                messages.append(
                    {
                        "role": "system",
                        "content": "[핵심 정보]\n" + "\n".join(dynamic_context),
                    }
                )

            # RAG 정보 추가
            messages.append(
                {
                    "role": "system",
                    "content": f"[RAG 정보 - 절대 우선]\n{state['context'][:500]}",
                }
            )

            user_prompt = f"""질문: {query}

RAG 정보를 우선하되, 부족한 부분은 보완하세요.
400자 이내로 구체적이고 실용적으로 답변."""

            messages.append({"role": "user", "content": user_prompt})

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=400,
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
        """RAG 정보 불충분시 전체 재생성 - 계층형 프롬프트 적용"""
        state["debug_path"].append("regenerate_full")

        # FALLBACK_ANSWERS 먼저 체크
        try:
            import config

            query_lower = state["query"].lower()

            for keyword, answer in config.FALLBACK_ANSWERS.items():
                if keyword in query_lower:
                    state["raw_answer"] = answer
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

            # 1. 기본 시스템 프롬프트 (짧게)
            base_system = config.BASE_SYSTEM_PROMPT

            # 2. 질문 분석
            query = state["query"]
            needs_calculation = "얼마" in query or "금액" in query or "계산" in query
            needs_eligibility = "조건" in query or "자격" in query or "가능" in query
            needs_period = "기간" in query or "언제" in query or "얼마나" in query
            needs_job_seeking = "구직" in query or "실업인정" in query
            needs_reduction = "반복" in query or "감액" in query

            # 3. 필요한 정보만 동적 주입
            dynamic_context = []

            if needs_calculation:
                facts = config.UNEMPLOYMENT_FACTS["amounts"]
                dynamic_context.append(
                    f"""[금액 정보]
- 일 상한액: {facts['daily_max']}
- 일 하한액: {facts['daily_min']}
- 지급률: {facts['rate']}
- 최저임금: 시간당 {facts['min_wage_hourly']}"""
                )

            if needs_eligibility:
                facts = config.UNEMPLOYMENT_FACTS["eligibility"]
                dynamic_context.append(
                    f"""[자격 조건]
- 가입기간: {facts['insurance_period']}
- 이직사유: {facts['resignation_type']}
- 연령제한: {facts['age_limit']}
- 신청기한: {facts['claim_deadline']}"""
                )

            if needs_period:
                facts = config.UNEMPLOYMENT_FACTS["periods"]
                dynamic_context.append(
                    f"""[수급 기간]
- 30세 미만: {facts['under_30']}
- 30~50세: {facts['30_to_50']}
- 50세 이상: {facts['50_plus']}
- 50세 이상 장애인: {facts['disabled_50_plus']}"""
                )

            if needs_job_seeking:
                facts = config.UNEMPLOYMENT_FACTS["job_seeking"]
                dynamic_context.append(
                    f"""[구직활동]
- 빈도: {facts['frequency']}
- 5차 이후: {facts['5th_onwards']}
- 인정활동: {facts['activities']}"""
                )

            if needs_reduction:
                facts = config.UNEMPLOYMENT_FACTS["reduction"]
                dynamic_context.append(
                    f"""[반복수급 감액]
- 5년 내 3회: {facts['3_times']}
- 5년 내 4회: {facts['4_times']}
- 5년 내 5회: {facts['5_times']}
- 5년 내 6회: {facts['6_times']}"""
                )

            # 4. 금지 정보 추가
            forbidden_context = f"""[절대 금지 정보]
다음은 잘못된 정보이므로 절대 사용하지 마세요:
{', '.join(config.COMMON_MISTAKES[:3])}"""

            # 5. 메시지 구성
            messages = [{"role": "system", "content": base_system}]

            if dynamic_context:
                messages.append(
                    {
                        "role": "system",
                        "content": "[2025년 핵심 정보]\n" + "\n".join(dynamic_context),
                    }
                )

            messages.append({"role": "system", "content": forbidden_context})

            user_prompt = f"""질문: {query}

위 질문에 대해 2025년 기준으로 정확하게 답변해주세요.
구체적인 금액, 기간, 조건을 포함하되 400자 이내로 작성하세요."""

            messages.append({"role": "user", "content": user_prompt})

            completion = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=0.2,  # 일관성 중요
                max_tokens=400,
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
        if not documents:
            return "관련 정보를 찾을 수 없습니다."

        # 첫 번째 문서 사용
        first_doc = documents[0]
        text = first_doc.get("text", "")

        # 메타데이터에서 답변 추출 시도
        metadata = first_doc.get("metadata", {})
        if metadata.get("answer"):
            return metadata["answer"]

        # 텍스트 그대로 반환
        return text[:300] if len(text) > 300 else text

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
