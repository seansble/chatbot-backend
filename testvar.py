# testvar.py - v3 (복잡한 복합 케이스 10개)

import json
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from backend.rag.unemployment_logic import unemployment_logic

class IntegratedRAGTest:
    """RAG + 변수추출 + 발화 통합 테스트"""
    
    def __init__(self):
        self.test_cases = [
            {
                "id": "다중경력_체불_노동청",
                "query": "첫 직장 8개월 이백만원, 두번째 일년반 삼백이십만원 받다가 마지막 3개월 체불로 노동청 신고하고 나왔어요. 서른여섯살.",
                "expected_vars": {
                    "age": 36,
                    "eligible_months": 26,  # 8 + 18
                    "monthly_salary": 3200000,  # 마지막 직장
                    "resignation_category": "정당한자발적",
                    "special_reason": "임금체불"
                },
                "expected_passages": ["임금체불", "정당한자발적", "노동청"],
                "expected_response_keywords": ["수급 가능", "26개월", "임금체불", "210일"]
            },
            {
                "id": "모호시간_반복5회_감액",
                "query": "꽤 오래전부터 일했는데 얼마 안 돼요. 다섯번째 받으려고 해요. 월급은 적당히 받았고 회사가 망했어요.",
                "expected_vars": {
                    "eligible_months": 18,  # 모호한 표현 추정값
                    "monthly_salary": 2500000,  # 모호한 표현 추정값
                    "resignation_category": "비자발적",
                    "special_reason": "회사폐업",
                    "repetition_count": 5
                },
                "expected_passages": ["비자발적", "반복수급", "감액"],
                "expected_response_keywords": ["5회차", "40% 감액", "회사폐업"]
            },
            {
                "id": "청년특례_3개월_계약종료",
                "query": "스물셋 알바생인데 3개월 단기 계약 끝나서 나왔어요. 시급 만원으로 하루 8시간 주5일 일했어요.",
                "expected_vars": {
                    "age": 23,
                    "eligible_months": 3,
                    "monthly_salary": 1760000,  # 10000*8*22
                    "resignation_category": "비자발적",
                    "special_reason": "계약만료",
                    "employment_type": "파트타임"
                },
                "expected_passages": ["청년", "3개월", "비자발적"],
                "expected_response_keywords": ["청년 특례", "3개월", "수급 가능"]
            },
            {
                "id": "장기근속_21년_권고사직",
                "query": "쉰둘입니다. 이십일년 근무한 회사에서 권고사직 받았어요. 연봉 육천만원이었습니다.",
                "expected_vars": {
                    "age": 52,
                    "eligible_months": 252,  # 21년
                    "monthly_salary": 5000000,  # 연봉/12
                    "resignation_category": "비자발적",
                    "special_reason": "권고사직"
                },
                "expected_passages": ["장기근속", "권고사직", "비자발적"],
                "expected_response_keywords": ["장기근속", "30일 추가", "270일", "권고사직"]
            },
            {
                "id": "장애_통근_복합사유",
                "query": "장애 3급이고 마흔여덟살. 편도 2시간 반에 상사 갑질까지 겹쳐서 14개월만에 퇴사. 이백팔십만원.",
                "expected_vars": {
                    "age": 48,
                    "eligible_months": 14,
                    "monthly_salary": 2800000,
                    "resignation_category": "정당한자발적",
                    "special_reason": "통근곤란",  # 또는 직장내괴롭힘
                    "disability": True
                },
                "expected_passages": ["장애", "통근", "괴롭힘", "정당한자발적"],
                "expected_response_keywords": ["14개월", "정당한", "180일"]
            },
            {
                "id": "특고_프리랜서_전환",
                "query": "정규직 2년하다가 프리랜서로 6개월, 다시 특고로 8개월 일했어요. 마지막 특고 월 삼백오십. 서른일곱.",
                "expected_vars": {
                    "age": 37,
                    "eligible_months": 32,  # 24 + 6 + 8 (단, 특고/프리 기간 제한 가능)
                    "monthly_salary": 3500000,
                    "employment_types": ["정규직", "프리랜서", "특고"]
                },
                "expected_passages": ["특고", "프리랜서", "고용보험"],
                "expected_response_keywords": ["특고", "가입기간", "확인 필요"]
            },
            {
                "id": "연속체불_부도_복합",
                "query": "작년 10월부터 올해 4월까지 일했는데 1월부터 4개월 월급 못받고 회사 부도났어요. 사백이십만원씩 받기로 했었어요.",
                "expected_vars": {
                    "age": 35,  # 기본값
                    "eligible_months": 7,  # 작년10월~올해4월
                    "monthly_salary": 4200000,
                    "resignation_category": "비자발적",  # 부도가 우선
                    "special_reason": "회사폐업"
                },
                "expected_passages": ["부도", "임금체불", "비자발적"],
                "expected_response_keywords": ["7개월", "회사폐업", "120일"]
            },
            {
                "id": "육아_간병_동시사유",
                "query": "마흔셋 여성입니다. 아이 돌봄과 부모님 간병이 겹쳐서 2년 3개월 다닌 직장 그만뒀어요. 삼백삼십만원.",
                "expected_vars": {
                    "age": 43,
                    "eligible_months": 27,
                    "monthly_salary": 3300000,
                    "resignation_category": "정당한자발적",
                    "special_reason": "가족돌봄"
                },
                "expected_passages": ["육아", "간병", "정당한자발적"],
                "expected_response_keywords": ["27개월", "가족돌봄", "210일"]
            },
            {
                "id": "한자어_복합_반복3회",
                "query": "오십구세. 삼년 육개월 근무. 구조조정. 월 사백팔십. 세번째 수급.",
                "expected_vars": {
                    "age": 59,
                    "eligible_months": 42,  # 3년 6개월
                    "monthly_salary": 4800000,
                    "resignation_category": "비자발적",
                    "special_reason": "구조조정",
                    "repetition_count": 3
                },
                "expected_passages": ["구조조정", "비자발적", "반복수급"],
                "expected_response_keywords": ["3회차", "10% 감액", "구조조정", "240일"]
            },
            {
                "id": "예술인_시즌제_복합경력",
                "query": "서른 뮤지컬 배우. 첫 시즌 4개월 이백만, 두번째 시즌 5개월 이백오십만 후 시즌 종료. 예술인 고용보험 가입.",
                "expected_vars": {
                    "age": 30,
                    "eligible_months": 9,  # 4 + 5
                    "monthly_salary": 2500000,  # 마지막
                    "resignation_category": "비자발적",
                    "special_reason": "계약만료",
                    "employment_type": "예술인"
                },
                "expected_passages": ["예술인", "시즌", "비자발적"],
                "expected_response_keywords": ["9개월", "예술인", "120일", "시즌 종료"]
            }
        ]
    
    def safe_calculate_benefit(self, variables: Dict) -> Dict:
        """안전 계산"""
        safe = variables.copy()
        safe.setdefault("age", 35)
        safe.setdefault("monthly_salary", 0)
        safe.setdefault("eligible_months", 0)
        safe.setdefault("resignation_category", "unknown")
        safe.setdefault("disability", False)
        return unemployment_logic.calculate_total_benefit(safe)
    
    def test_full_pipeline(self, test_case: Dict) -> Dict:
        results = {"id": test_case["id"], "query": test_case["query"], "steps": {}}
        try:
            # 1) 추출
            extracted = unemployment_logic.extract_variables_with_llm(test_case["query"])
            print(f"  추출된 원본: {extracted}")
            results["steps"]["extraction"] = {
                "extracted": extracted,
                "expected": test_case["expected_vars"],
                "accuracy": self._calculate_accuracy(extracted, test_case["expected_vars"])
            }
            # 2) 계산
            calculation = self.safe_calculate_benefit(extracted)
            results["steps"]["calculation"] = calculation
            # 3) RAG 모의
            search_query = self._build_search_query(extracted, test_case["query"])
            passages = self._mock_search_relevant_docs(search_query)
            results["steps"]["rag_search"] = {
                "search_query": search_query,
                "found_passages": len(passages),
                "relevant": self._check_passage_relevance(passages, test_case.get("expected_passages", []))
            }
            # 4) 응답
            context = {"user_query": test_case["query"], "extracted_vars": extracted,
                       "calculation": calculation, "relevant_passages": passages}
            final_response = self._generate_proper_response(context)
            results["steps"]["response"] = {
                "text": final_response,
                "contains_keywords": self._check_keywords(final_response, test_case.get("expected_response_keywords", [])),
                "length": len(final_response)
            }
            results["total_score"] = self._calculate_total_score(results)
        except Exception as e:
            print(f"  오류 발생: {e}")
            import traceback; traceback.print_exc()
            results["error"] = str(e); results["total_score"] = 0
        return results
    
    def _generate_proper_response(self, context: Dict) -> str:
        calc = context["calculation"]; vars = context["extracted_vars"]
        if not calc.get("eligible"):
            return f"수급 불가. 사유: {calc.get('reason')}"
        parts = []
        
        # 특례 적용 체크
        if calc.get("is_youth"):
            parts.append("청년 특례 적용으로")
        elif vars.get("eligible_months") and vars["eligible_months"] >= 240:
            parts.append("장기근속 특례로")
            
        if vars.get("special_reason"):
            parts.append(f"{vars['special_reason']}으로 인한 퇴사는")
        if vars.get("resignation_category") == "정당한자발적":
            parts.append("정당한 자발적 퇴사로")
        elif vars.get("resignation_category") == "비자발적":
            parts.append("비자발적 퇴사로")
        parts.append("실업급여 수급 가능.")
        
        details = []
        if calc.get("age"): details.append(f"{calc['age']}세")
        details += [f"{calc['eligible_months']}개월 가입", f"{calc['benefit_days']}일",
                    f"일 급여 {calc['daily_benefit']:,}원", f"총 {calc['total_amount']:,}원"]
        parts.append(" ".join(details))
        
        if calc.get("reduction_info"): 
            parts.append(f"({calc['reduction_info']})")
        if vars.get("repetition_count") and vars['repetition_count'] >= 3:
            parts.append(f"{vars['repetition_count']}회차 반복수급 감액 적용")
        if vars.get("employment_type") in ["특고", "프리랜서", "예술인"]:
            parts.append(f"{vars['employment_type']} 고용보험 가입기간 확인 필요")
            
        return " ".join(parts)
    
    def _mock_search_relevant_docs(self, query: str) -> List[str]:
        mock_passages = {
            "정당한자발적": [
                "정당한 자발적 퇴사 인정 사유: 임금체불, 직장내 괴롭힘, 통근시간 장기, 가족돌봄 등."
            ],
            "비자발적": [
                "권고사직, 해고, 계약만료, 회사 폐업, 구조조정은 비자발적 퇴사로 수급 가능."
            ],
            "임금체불": [
                "임금체불은 정당한 이직사유. 노동청 신고로 체당금 및 실업급여 병행 신청 가능."
            ],
            "통근": [
                "편도 2시간 반 이상 등 장거리 통근은 센터 판단으로 정당사유 가능."
            ],
            "장애": [
                "장애 등록자는 수급일수 연장 가능. 장애 + 만 50세 이상은 270일 적용."
            ],
            "괴롭힘": [
                "직장 내 괴롭힘, 갑질은 정당한 사유의 자발적 이직에 해당."
            ],
            "청년": [
                "만 18-34세 청년은 가입기간 3개월부터 수급 가능. 수급액 10% 가산."
            ],
            "장기근속": [
                "20년 이상 장기근속자는 수급일수 30일 추가 지급."
            ],
            "반복수급": [
                "5년 내 반복수급시 3회 10%, 4회 25%, 5회 40%, 6회 이상 50% 감액."
            ],
            "예술인": [
                "예술인 고용보험 가입자는 시즌제, 프로젝트 단위 계약 종료시 수급 가능."
            ],
            "특고": [
                "특고·프리랜서는 특수형태근로종사자 고용보험 가입 기간만 인정."
            ],
            "육아": [
                "육아, 간병 등 가족돌봄은 정당한 이직사유로 인정 가능."
            ],
            "부도": [
                "회사 부도, 폐업은 비자발적 퇴사. 임금체불 동시 발생시 체당금 신청 가능."
            ],
            "구조조정": [
                "구조조정, 정리해고는 비자발적 퇴사로 즉시 수급 가능."
            ],
            "시즌": [
                "시즌제 근로자의 시즌 종료는 계약만료로 비자발적 퇴사 인정."
            ]
        }
        relevant = []
        for key, passages in mock_passages.items():
            if key in query.lower():
                relevant.extend(passages)
        return relevant[:5] if relevant else ["실업급여 일반 지침"]
    
    def _build_search_query(self, vars: Dict, original: str) -> str:
        parts = []
        if vars.get("resignation_category"): parts.append(vars["resignation_category"])
        if vars.get("special_reason"): parts.append(vars["special_reason"])
        if vars.get("disability"): parts.append("장애")
        if vars.get("repetition_count") and vars["repetition_count"] >= 3: parts.append("반복수급")
        if vars.get("age") and 18 <= vars["age"] <= 34: parts.append("청년")
        if vars.get("eligible_months") and vars["eligible_months"] >= 240: parts.append("장기근속")
        if vars.get("employment_type"): parts.append(vars["employment_type"])
        return " ".join(parts)
    
    def _calculate_accuracy(self, extracted: Dict, expected: Dict) -> float:
        if not expected: return 100.0
        correct = 0; total = len(expected)
        for key, exp in expected.items():
            act = extracted.get(key)
            # 특별 처리: 근사값 허용
            if key == "eligible_months" and act and exp:
                if abs(act - exp) <= 2:  # ±2개월 오차 허용
                    correct += 1
                elif act == exp:
                    correct += 1
            elif key == "monthly_salary" and act and exp:
                if abs(act - exp) <= 100000:  # ±10만원 오차 허용
                    correct += 1
                elif act == exp:
                    correct += 1
            elif key == "employment_types":
                # 리스트 비교: 포함 여부 체크
                if act and exp and all(e in act for e in exp):
                    correct += 1
            elif act == exp:
                correct += 1
            elif key == "repetition" and extracted.get("repetition_count") == exp:
                correct += 1
        return (correct / total) * 100 if total > 0 else 0
    
    def _check_passage_relevance(self, passages: List, keywords: List) -> Dict:
        if not keywords: return {"found_keywords": [], "coverage": 100.0}
        found = []
        for kw in keywords:
            for p in passages:
                if kw in p: found.append(kw); break
        return {"found_keywords": found, "coverage": len(found) / len(keywords) * 100 if keywords else 100}
    
    def _check_keywords(self, response: str, keywords: List) -> Dict:
        if not keywords: return {"found": [], "missing": [], "coverage": 100.0}
        found = [kw for kw in keywords if kw in response]
        return {"found": found, "missing": [kw for kw in keywords if kw not in found],
                "coverage": len(found) / len(keywords) * 100 if keywords else 100}
    
    def _calculate_total_score(self, results: Dict) -> float:
        extraction = results["steps"]["extraction"]["accuracy"]
        rag = results["steps"]["rag_search"]["relevant"]["coverage"]
        response = results["steps"]["response"]["contains_keywords"]["coverage"]
        return extraction * 0.5 + rag * 0.2 + response * 0.3
    
    def run_all_tests(self):
        print("=" * 60); print("RAG + LLM 통합 테스트 (복잡한 케이스)"); print("=" * 60)
        all_results = []
        for i, tc in enumerate(self.test_cases, 1):
            print(f"\n테스트 {i} [{tc['id']}]")
            print(f"질문: {tc['query'][:80]}...")
            res = self.test_full_pipeline(tc)
            if "error" not in res:
                print(f"├─ 변수 추출: {res['steps']['extraction']['accuracy']:.1f}%")
                print(f"├─ RAG 검색: {res['steps']['rag_search']['relevant']['coverage']:.1f}%")
                print(f"├─ 응답 품질: {res['steps']['response']['contains_keywords']['coverage']:.1f}%")
                print(f"└─ 통합 점수: {res['total_score']:.1f}%")
                ext = res['steps']['extraction']['extracted']
                print(f"\n주요 변수:")
                print(f"  - 나이: {ext.get('age', 'None')}")
                print(f"  - 근무: {ext.get('eligible_months', 'None')}개월")
                print(f"  - 월급: {ext.get('monthly_salary', 0):,}원")
                print(f"  - 퇴사: {ext.get('resignation_category', 'None')}")
                if ext.get('repetition_count'):
                    print(f"  - 반복: {ext['repetition_count']}회차")
                if ext.get('employment_type'):
                    print(f"  - 고용: {ext['employment_type']}")
                if res['steps']['calculation'].get('eligible'):
                    print(f"\n응답: {res['steps']['response']['text'][:220]}...")
            all_results.append(res)
        valid = [r for r in all_results if "error" not in r]
        if valid:
            avg = sum(r["total_score"] for r in valid) / len(valid)
            print(f"\n{'=' * 60}")
            print(f"전체 평균 점수: {avg:.1f}%")
            print(f"성공: {len(valid)}/{len(all_results)}")
        with open("integrated_test_results_complex.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        return all_results

if __name__ == "__main__":
    tester = IntegratedRAGTest()
    results = tester.run_all_tests()