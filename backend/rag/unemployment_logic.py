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

    # 수급일수 테이블 (나이별, 가입기간별)
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
        """강화된 LLM 변수 추출"""
        
        prompt = f"""질문을 깊이 이해하고 문맥을 파악하여 실업급여 계산에 필요한 모든 변수를 추출하세요.
아래 예시의 동의어, 유사표현, 변형된 표현도 모두 찾아주세요.

질문: "{query}"

찾아야 할 변수:
1. 나이: 살/세/년생/서른/마흔/쉰 → 숫자
   예: "32살"→32, "서른"→30, "93년생"→32
2. 급여: 만원/월급/연봉/시급/일당 → 원단위
   예: "280만원"→2800000, "연봉 3천"→2500000, "시급 만원"→1730000
3. 기간: 개월/년/달/반년 → 개월수
   예: "8개월"→8, "반년"→6, "1년반"→18
4. 직종: 정규직/계약직/프리랜서/알바/일용직
5. 퇴사: 권고사직/해고/짤림/그만둠 → 비자발적/자발적
   예: "권고사직"→비자발적, "그만뒀어"→자발적
6. 특수: 장애/임신/반복수급횟수

발견한 표현과 변환값을 JSON으로:
{{
  "age": 숫자 또는 null,
  "monthly_salary": 원단위 또는 null,
  "eligible_months": 개월수 또는 null,
  "employment_type": "정규직/계약직/프리랜서" 또는 null,
  "resignation_category": "비자발적/정당한자발적/단순자발적" 또는 null,
  "repetition_count": 숫자 또는 null,
  "disability": true/false
}}"""

        try:
            completion = self.client.chat.completions.create(
                model=config.EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "변수 추출 전문가. JSON만 출력."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            response = completion.choices[0].message.content

            # JSON 추출
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            variables = json.loads(response.strip())
            variables = self._validate_variables(variables)
            
            logger.info(f"Variables extracted: {variables}")
            return variables

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._extract_with_regex(query)

    def _extract_with_regex(self, query: str) -> Dict[str, Any]:
        """정규식 기반 폴백 추출"""
        variables = {
            "age": None,
            "monthly_salary": None,
            "eligible_months": 0,
            "resignation_category": None,
            "repetition_count": None,
            "disability": False,
        }

        # 나이
        age_patterns = [
            (r'(\d{2,3})\s*살', lambda x: int(x)),
            (r'서른', lambda x: 30),
            (r'마흔', lambda x: 40),
            (r'쉰', lambda x: 50),
        ]
        for pattern, converter in age_patterns:
            if isinstance(pattern, str):
                if pattern in query:
                    variables['age'] = converter(None)
                    break
            else:
                match = re.search(pattern, query)
                if match:
                    variables['age'] = converter(match.group(1))
                    break

        # 월급
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

        # 근무기간
        month_patterns = [
            (r"(\d+)\s*개월", lambda x: int(x)),
            (r"(\d+)\s*년", lambda x: int(x) * 12),
            (r"반년", lambda x: 6),
            (r"일년반", lambda x: 18),
        ]
        total_months = 0
        for pattern, converter in month_patterns:
            if isinstance(pattern, str):
                if pattern in query:
                    total_months += converter(None)
            else:
                matches = re.findall(pattern, query)
                for match in matches:
                    total_months += converter(match)
        
        variables["eligible_months"] = total_months

        # 퇴사사유
        if any(word in query for word in ["권고사직", "해고", "짤렸", "잘렸"]):
            variables["resignation_category"] = "비자발적"
        elif any(word in query for word in ["임금체불", "괴롭힘"]):
            variables["resignation_category"] = "정당한자발적"
        elif any(word in query for word in ["그만", "퇴사", "이직"]):
            variables["resignation_category"] = "단순자발적"

        # 반복수급
        repetition_patterns = [r"(\d)\s*번째", r"(\d)회차", r"(\d)번\s*받"]
        for pattern in repetition_patterns:
            match = re.search(pattern, query)
            if match:
                variables["repetition_count"] = int(match.group(1))
                break

        # 장애
        if "장애" in query:
            variables["disability"] = True

        return variables

    def _validate_variables(self, variables: Dict) -> Dict:
        """변수 검증 및 보정"""
        if variables is None:
            return {}

        # 나이 계산
        if not variables.get("age") and variables.get("birth_year"):
            current_year = datetime.now().year
            variables["age"] = current_year - variables["birth_year"] + 1

        # 기본값 처리
        if not variables.get("eligible_months"):
            variables["eligible_months"] = 0

        return variables

    def calculate_benefit_days(
        self, age: int, insurance_months: int, disability: bool = False
    ) -> int:
        """수급일수 계산"""
        if disability and age >= 50:
            return self.DISABLED_50_PLUS_DAYS

        for (min_age, max_age), periods in self.BENEFIT_DAYS_TABLE.items():
            if min_age <= age < max_age:
                for min_months, max_months, days in periods:
                    if min_months <= insurance_months < max_months:
                        return days

        return 90

    def calculate_daily_amount(self, monthly_salary: int) -> Dict[str, int]:
        """일급 계산"""
        if not monthly_salary:
            return {"daily_base": 0, "daily_benefit": 0, "applied": "계산불가"}

        daily_wage = monthly_salary / 30
        benefit_60 = daily_wage * 0.6

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

    def calculate_total_benefit(self, variables: Dict) -> Dict[str, Any]:
        """종합 실업급여 계산"""
        
        age = variables.get("age", 30)
        salary = variables.get("monthly_salary", 0)
        eligible_months = variables.get("eligible_months", 0)
        resignation = variables.get("resignation_category", "")
        repetition = variables.get("repetition_count", 1)
        disability = variables.get("disability", False)

        # 수급자격 체크
        if eligible_months < 6:
            return {
                "eligible": False,
                "reason": f"고용보험 {eligible_months}개월 < 최소 6개월",
            }

        if resignation == "단순자발적":
            return {
                "eligible": False,
                "reason": "단순 자발적 퇴사는 수급 불가",
            }

        # 수급일수 계산
        benefit_days = self.calculate_benefit_days(age, eligible_months, disability)

        # 일급 계산
        daily_amount = self.calculate_daily_amount(salary)

        # 반복수급 감액
        if repetition and repetition >= 3:
            penalty = self.REPETITION_PENALTY.get(repetition, 0.5)
            benefit_days = int(benefit_days * penalty)
            reduction_info = f"{repetition}회차 {int((1-penalty)*100)}% 감액"
        else:
            reduction_info = ""

        # 총액 계산
        total_amount = daily_amount["daily_benefit"] * benefit_days

        return {
            "eligible": True,
            "age": age,
            "monthly_salary": salary,
            "eligible_months": eligible_months,
            "resignation_type": resignation,
            "daily_benefit": daily_amount["daily_benefit"],
            "applied_limit": daily_amount["applied"],
            "benefit_days": benefit_days,
            "total_amount": total_amount,
            "reduction_info": reduction_info,
        }

    def format_calculation_result(self, result: Dict) -> str:
        """계산 결과를 원인-결과 문장으로"""
        
        if not result.get("eligible"):
            return f"수급 불가: {result.get('reason')}"

        resignation_text = {
            "비자발적": "권고사직은 비자발적 퇴사로 인정되어",
            "정당한자발적": "정당한 사유의 자발적 퇴사로",
            "단순자발적": "단순 자발적 퇴사는"
        }.get(result.get('resignation_type', ''), "퇴사 사유로")

        return f"""계산 완료:
- {resignation_text} 실업급여 수급이 가능합니다.
- {result['age']}세이고 {result['eligible_months']}개월 가입하여 수급기간은 {result['benefit_days']}일입니다.
- 월급 {result['monthly_salary']:,}원의 60%를 적용하면 일 급여는 {result['daily_benefit']:,}원입니다.
- 일 급여 {result['daily_benefit']:,}원을 {result['benefit_days']}일간 받으므로 총 수급액은 {result['total_amount']:,}원입니다.
{f"- {result['reduction_info']}" if result.get('reduction_info') else ''}"""


# 싱글톤 인스턴스
unemployment_logic = UnemploymentLogic()