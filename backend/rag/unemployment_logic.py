# backend/rag/unemployment_logic.py
"""실업급여 통합 로직 모듈"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from openai import OpenAI
import sys
from pathlib import Path

# config import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class UnemploymentLogic:
    """실업급여 계산 및 판단 로직"""

    # 수급일수 테이블 (나이별, 가입기간별) - 구조 수정
    BENEFIT_DAYS_TABLE = {
        # (나이_최소, 나이_최대): [(최소개월, 최대개월, 수급일수), ...]
        (0, 30): [
            (0, 12, 90),
            (12, 36, 90),
            (36, 60, 120),
            (60, 120, 150),
            (120, 999, 180),
        ],
        (30, 50): [
            (0, 12, 90),
            (12, 36, 120),
            (36, 60, 150),
            (60, 120, 180),
            (120, 999, 210),
        ],
        (50, 999): [
            (0, 12, 120),
            (12, 36, 150),
            (36, 60, 180),
            (60, 120, 210),
            (120, 999, 240),
        ],
    }

    # 나머지 상수들은 동일...
    DISABLED_50_PLUS_DAYS = 270

    RESIGNATION_TYPES = {
        "비자발적": [
            "권고사직",
            "해고",
            "계약만료",
            "회사폐업",
            "정리해고",
            "경영악화",
        ],
        "정당한자발적": [
            "임금체불",
            "괴롭힘",
            "질병",
            "성희롱",
            "통근곤란",
            "가족간병",
        ],
        "단순자발적": ["이직", "개인사정", "자진퇴사", "그만둠"],
    }

    EMPLOYMENT_TYPES = {
        "정규직": {"insurance_rate": 1.0, "min_days": 180},
        "계약직": {"insurance_rate": 1.0, "min_days": 180},
        "프리랜서": {"insurance_rate": 0.0, "min_income": 800000},
        "특고": {"insurance_rate": 0.0, "min_income": 800000},
        "일용직": {"insurance_rate": 1.0, "min_days_per_month": 15},
        "예술인": {"insurance_rate": 0.5, "min_days": 90},
        "초단시간": {"insurance_rate": 0.0, "weekly_hours": 15},
    }

    REPETITION_PENALTY = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.75, 5: 0.6, 6: 0.5}

    def __init__(self):
        """초기화"""
        self.client = OpenAI(
            base_url=config.API_BASE_URL, api_key=config.TOGETHER_API_KEY
        )

    def extract_variables_with_llm(self, query: str) -> Dict[str, Any]:
        """LLM을 사용한 변수 추출 (Qwen2.5-7B-Turbo)"""

        prompt = f"""질문: {query}

다음 정보를 추출하고 판단하세요:

[필수 추출 항목]
1. 나이 또는 생년월일
2. 월급 (세전 기준)
3. 근무 이력 (각 직장별)
4. 퇴사 사유
5. 반복수급 여부

[퇴사사유 분류]
- 비자발적: 권고사직, 해고, 계약만료, 회사폐업
- 정당한자발적: 임금체불(2개월↑), 괴롭힘, 질병, 성희롱
- 단순자발적: 이직, 개인사정

[근무유형 분류]
- 정규직/계약직: 고용보험 100%
- 프리랜서/특고: 월 80만원 이상시만
- 일용직: 월 15일 이상 근무시
- 예술인: 특별기준 적용

JSON 형식으로만 답변:
{{
  "age": 숫자 또는 null,
  "birth_year": 숫자 또는 null,
  "monthly_salary": 숫자 또는 null,
  "work_history": [
    {{
      "company": "회사명 또는 A/B/C",
      "duration_months": 숫자,
      "employment_type": "정규직/계약직/프리랜서/일용직/예술인",
      "resignation_type": "권고사직/해고/자진퇴사 등",
      "resignation_category": "비자발적/정당한자발적/단순자발적",
      "wage_delayed": true/false,
      "insurance_covered": true/false
    }}
  ],
  "total_months": 총근무개월,
  "eligible_months": 실업급여인정개월,
  "repetition_count": 반복수급횟수 또는 null,
  "disability": true/false,
  "special_notes": "육아휴직, 병역 등"
}}"""

        try:
            completion = self.client.chat.completions.create(
                model=config.EVAL_MODEL,  # Qwen2.5-7B-Turbo
                messages=[
                    {"role": "system", "content": "정보 추출 전문가. JSON만 출력."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                # response_format 제거 - Together AI 미지원
            )

            response = completion.choices[0].message.content

            # JSON 추출 (```json 블록 처리)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            variables = json.loads(response.strip())

            # 검증 및 보정
            variables = self._validate_variables(variables)
            logger.info(f"Variables extracted: {variables}")
            return variables

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}, Response: {response[:200]}")
            return self._extract_with_regex(query)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._extract_with_regex(query)

    def _extract_with_regex(self, query: str) -> Dict[str, Any]:
        """정규식 기반 폴백 추출"""
        variables = {
            "age": None,
            "monthly_salary": None,
            "work_history": [],
            "total_months": 0,
            "eligible_months": 0,
            "repetition_count": None,
            "disability": False,
            "special_notes": "",
        }

        # 나이 추출
        age_match = re.search(r"(\d{2,3})\s*살", query)
        if age_match:
            variables["age"] = int(age_match.group(1))

        # 월급 추출
        salary_patterns = [
            r"(\d{3,4})\s*만\s*원",
            r"월\s*(\d{3,4})",
            r"월급\s*(\d{3,4})",
        ]
        for pattern in salary_patterns:
            match = re.search(pattern, query)
            if match:
                variables["monthly_salary"] = int(match.group(1)) * 10000
                break

        # 근무기간 추출
        month_matches = re.findall(r"(\d{1,2})\s*개월", query)
        if month_matches:
            variables["total_months"] = sum(int(m) for m in month_matches)
            variables["eligible_months"] = variables["total_months"]

            # 간단한 work_history 생성
            if variables["total_months"] > 0:
                variables["work_history"] = [
                    {
                        "company": "회사",
                        "duration_months": variables["total_months"],
                        "employment_type": "정규직",
                        "resignation_type": (
                            "권고사직" if "권고사직" in query else "자진퇴사"
                        ),
                        "resignation_category": (
                            "비자발적"
                            if "권고사직" in query or "해고" in query
                            else "단순자발적"
                        ),
                        "insurance_covered": True,
                    }
                ]

        # 반복수급
        repetition_patterns = [r"(\d)\s*번째", r"(\d)회차", r"(\d)번\s*받"]
        for pattern in repetition_patterns:
            match = re.search(pattern, query)
            if match:
                variables["repetition_count"] = int(match.group(1))
                break

        # 장애인 여부
        if "장애" in query:
            variables["disability"] = True

        return variables

    def _validate_variables(self, variables: Dict) -> Dict:
        """추출된 변수 검증 및 보정"""
        # 기본값 설정
        if variables is None:
            return self._extract_with_regex("")

        # 나이 계산 (생년월일에서)
        if not variables.get("age") and variables.get("birth_year"):
            current_year = datetime.now().year
            variables["age"] = current_year - variables["birth_year"] + 1

        # work_history가 없으면 기본값
        if not variables.get("work_history"):
            variables["work_history"] = []

        # 적격 개월수 계산
        eligible_months = 0
        for job in variables.get("work_history", []):
            if job.get("insurance_covered", False):
                if job.get("resignation_category") in ["비자발적", "정당한자발적"]:
                    eligible_months += job.get("duration_months", 0)
        variables["eligible_months"] = eligible_months

        return variables

    def calculate_benefit_days(
        self, age: int, insurance_months: int, disability: bool = False
    ) -> int:
        """수급일수 계산 - 수정된 로직"""
        # 장애인 50세 이상 특별 규정
        if disability and age >= 50:
            return self.DISABLED_50_PLUS_DAYS

        # 일반 계산
        for (min_age, max_age), periods in self.BENEFIT_DAYS_TABLE.items():
            if min_age <= age < max_age:
                for min_months, max_months, days in periods:  # 리스트로 변경
                    if min_months <= insurance_months < max_months:
                        return days

        # 기본값
        return 90

    def calculate_daily_amount(self, monthly_salary: int) -> Dict[str, int]:
        """일급 계산"""
        if not monthly_salary:
            return {"daily_base": 0, "daily_benefit": 0, "applied": "계산불가"}

        # 평균일급 계산 (월급 / 30)
        daily_wage = monthly_salary / 30

        # 실업급여 = 평균임금의 60%
        benefit_60 = daily_wage * 0.6

        # 상하한 적용
        daily_max = 66000
        daily_min = 64192

        if benefit_60 > daily_max:
            return {
                "daily_base": int(daily_wage),
                "daily_benefit": daily_max,
                "applied": "상한액",
            }
        elif benefit_60 < daily_min:
            return {
                "daily_base": int(daily_wage),
                "daily_benefit": daily_min,
                "applied": "하한액",
            }
        else:
            return {
                "daily_base": int(daily_wage),
                "daily_benefit": int(benefit_60),
                "applied": "60%",
            }

    def check_eligibility(self, variables: Dict) -> Dict[str, Any]:
        """수급자격 종합 판단"""
        result = {"eligible": False, "reasons": [], "warnings": [], "suggestions": []}

        # 1. 근무일수 체크 (180일)
        total_days = variables.get("eligible_months", 0) * 22
        if total_days >= 180:
            result["reasons"].append(f"✅ 근무일수 충족 ({total_days}일 ≥ 180일)")
        else:
            result["reasons"].append(f"❌ 근무일수 부족 ({total_days}일 < 180일)")
            result["suggestions"].append("180일 충족시까지 더 근무 필요")
            return result

        # 2. 퇴사사유 체크
        if variables.get("work_history"):
            last_job = (
                variables["work_history"][-1] if variables["work_history"] else {}
            )
            resignation = last_job.get("resignation_category", "")

            if resignation == "비자발적":
                result["reasons"].append("✅ 비자발적 이직 (수급 가능)")
            elif resignation == "정당한자발적":
                result["reasons"].append("✅ 정당한 사유의 자발적 이직")
            elif resignation == "단순자발적":
                result["reasons"].append("❌ 단순 자발적 이직 (수급 불가)")
                result["suggestions"].append("비자발적 이직 또는 정당한 사유 필요")
                return result

        # 3. 나이 체크
        age = variables.get("age", 0)
        if age >= 65:
            result["warnings"].append("65세 이상은 특별 요건 필요")

        # 4. 반복수급 체크
        repetition = variables.get("repetition_count", 0)
        if repetition >= 3:
            penalty = (1 - self.REPETITION_PENALTY.get(repetition, 0.5)) * 100
            result["warnings"].append(f"반복수급 {penalty:.0f}% 감액 적용")

        # 최종 판단
        has_error = any("❌" in r for r in result["reasons"])
        if not has_error:
            result["eligible"] = True

        return result

    def apply_repetition_penalty(self, base_amount: int, count: int) -> int:
        """반복수급 감액 적용"""
        if count <= 2:
            return base_amount

        penalty_rate = self.REPETITION_PENALTY.get(count, 0.5)
        return int(base_amount * penalty_rate)

    def handle_complex_work_history(self, work_history: List[Dict]) -> Dict:
        """복합 근무이력 처리"""
        result = {
            "total_months": 0,
            "eligible_months": 0,
            "gaps": [],
            "overlaps": [],
            "within_18months": 0,
        }

        # 18개월 기준일
        cutoff_date = datetime.now() - timedelta(days=548)

        for i, job in enumerate(work_history):
            months = job.get("duration_months", 0)
            result["total_months"] += months

            # 자격 있는 개월만
            if job.get("insurance_covered") and job.get("resignation_category") in [
                "비자발적",
                "정당한자발적",
            ]:
                result["eligible_months"] += months

        # 18개월 내 근무만
        result["within_18months"] = min(result["eligible_months"], 18)

        return result

    def handle_special_cases(self, variables: Dict) -> List[str]:
        """특수 케이스 처리"""
        special_notes = []

        # 65세 이상
        if variables.get("age", 0) >= 65:
            special_notes.append("65세 이상: 65세 이전부터 계속 근무시만 가능")

        # 장애인
        if variables.get("disability"):
            special_notes.append("장애인: 50세 이상시 최대 270일")

        # 임신/출산
        if "임신" in variables.get("special_notes", "") or "출산" in variables.get(
            "special_notes", ""
        ):
            special_notes.append("임신/출산: 수급기간 90일 연장 가능")

        # 육아휴직
        if "육아휴직" in variables.get("special_notes", ""):
            special_notes.append("육아휴직: 기간 제외하고 계산")

        # 프리랜서
        for job in variables.get("work_history", []):
            if job.get("employment_type") == "프리랜서":
                special_notes.append("프리랜서: 월 80만원 이상시만 고용보험 적용")
                break

        return special_notes

    def calculate_total_benefit(self, variables: Dict) -> Dict[str, Any]:
        """종합 실업급여 계산"""

        # 1. 변수 확인
        age = variables.get("age", 30)
        salary = variables.get("monthly_salary", 0)
        eligible_months = variables.get("eligible_months", 0)
        repetition = variables.get("repetition_count", 1)
        disability = variables.get("disability", False)

        # 2. 수급자격 체크
        eligibility = self.check_eligibility(variables)

        if not eligibility["eligible"]:
            return {
                "eligible": False,
                "reasons": eligibility.get("reasons", []),
                "suggestions": eligibility.get("suggestions", []),
            }

        # 3. 수급일수 계산
        benefit_days = self.calculate_benefit_days(age, eligible_months, disability)

        # 4. 일급 계산
        daily_amount = self.calculate_daily_amount(salary)

        # 5. 반복수급 감액
        final_daily_amount = daily_amount["daily_benefit"]
        if repetition and repetition >= 3:
            final_daily_amount = self.apply_repetition_penalty(
                daily_amount["daily_benefit"], repetition
            )

        # 6. 총액 계산
        total_amount = final_daily_amount * benefit_days

        # 7. 특수 케이스
        special_notes = self.handle_special_cases(variables)

        return {
            "eligible": True,
            "benefit_days": benefit_days,
            "daily_amount": final_daily_amount,
            "total_amount": total_amount,
            "calculation_details": {
                "age": age,
                "salary": salary,
                "eligible_months": eligible_months,
                "eligible_days": eligible_months * 22,
                "repetition_count": repetition,
                "penalty_rate": (
                    self.REPETITION_PENALTY.get(repetition, 1.0) if repetition else 1.0
                ),
            },
            "reasons": eligibility.get("reasons", []),
            "warnings": eligibility.get("warnings", []),
            "special_notes": special_notes,
        }

    def format_calculation_result(self, result: Dict) -> str:
        """계산 결과를 구조화된 텍스트로 변환"""
        if not result.get("eligible"):
            reasons = result.get("reasons", [])
            suggestions = result.get("suggestions", [])

            text = "[수급 불가]\n"
            if reasons:
                text += "\n".join(reasons)
            if suggestions:
                text += "\n\n제안사항:\n"
                text += "\n".join(suggestions)
            return text

        # 수급 가능한 경우
        text = f"""[계산 결과]
- 수급자격: 충족
- 수급기간: {result['benefit_days']}일
- 일 급여액: {result['daily_amount']:,}원
- 예상 총액: {result['total_amount']:,}원"""

        if result.get("reasons"):
            text += "\n\n[세부사항]\n"
            text += "\n".join(result["reasons"])

        if result.get("warnings"):
            text += "\n\n[주의사항]\n"
            text += "\n".join(result["warnings"])

        if result.get("special_notes"):
            text += "\n\n[특수사항]\n"
            text += "\n".join(result["special_notes"])

        return text


# 싱글톤 인스턴스
unemployment_logic = UnemploymentLogic()
