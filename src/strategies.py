"""L2M-CoT 전략 패턴 모듈."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

from src.prompts import build_common_task_block
from src.utils import call_model, extract_final_answer


# =========================================================
# Strategy Framework
# =========================================================
@dataclass
class StrategyResult:
    """전략 실행 결과를 담는 데이터 클래스."""

    raw_output: str
    final_answer: str
    meta_info: Dict[str, object] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_time: float = 0.0  # 초 단위


class SolverStrategy(ABC):
    """문제 풀이 전략의 추상 베이스 클래스."""

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름을 반환."""
        ...

    @abstractmethod
    def solve(self, question: str, use_fewshot: bool, n_words: int) -> StrategyResult:
        """문제를 풀고 결과를 반환."""
        ...


# =========================================================
# ✅ Baseline = base class (1-call)
# - 자식은 extra_instructions만 override해서 prompt만 바뀜
# =========================================================
class BaselineStrategy(SolverStrategy):
    """기본 전략 (단일 LLM 호출)."""

    @property
    def name(self) -> str:
        return "Baseline"

    def extra_instructions(self, n_words: int) -> str:
        """추가 지시사항을 반환. 하위 클래스에서 오버라이드."""
        return """
Strategy (Baseline):
- Solve the task directly.
- Do NOT show any working.
""".strip()

    def build_prompt(self, question: str, use_fewshot: bool, n_words: int) -> str:
        """최종 프롬프트를 빌드."""
        common = build_common_task_block(use_fewshot)
        extra = self.extra_instructions(n_words).strip()
        extra_block = f"\n\n{extra}" if extra else ""

        global_constraints = f"""
Global output constraints:
- Do NOT write the phrase "Final answer" anywhere except the very last line.
- The very last line must be exactly: Final answer: <string>
- After the final line, output nothing.
- <string> must contain only lowercase letters a-z, with NO spaces or punctuation.
- <string> length MUST be exactly {n_words}.
- Do not include any other lines that contain the substring "final answer".
""".strip()

        return f"""{common}
{extra_block}

Now solve the following instance:
{question}

{global_constraints}
""".strip()

    def solve(self, question: str, use_fewshot: bool, n_words: int) -> StrategyResult:
        """문제를 풀고 결과를 반환."""
        prompt = self.build_prompt(question, use_fewshot, n_words)
        start = time.perf_counter()
        response = call_model(prompt)
        elapsed = time.perf_counter() - start
        final = extract_final_answer(response.text, n_words)
        return StrategyResult(
            raw_output=response.text,
            final_answer=final,
            meta_info={"num_calls": 1},
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            elapsed_time=elapsed,
        )


# =========================================================
# CoT: (정확도 목적) 내부적으로 step-by-step, 출력은 최소
# =========================================================
class CoTStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "CoT"

    def extra_instructions(self, n_words: int) -> str:
        # 변경: 추론 과정을 보여주도록 지시해야 성능이 향상됨
        return """
Strategy (CoT):
- Think step by step internally, but do NOT write any reasoning.
- Do NOT write intermediate steps, notes, or explanations.
- Output ONLY the final line in the required format.
""".strip()


# =========================================================
# L2M: least-to-most 작업을 "엄격한 포맷"으로 출력
# =========================================================
class L2MStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "L2M"

    def decomposition_prompt(self) -> str:
        """Decomposition 단계 프롬프트를 반환."""
        return """
Decomposition (strict):
	•	Output exactly ONE line starting with "Decomposition:".
	•	After it, write a list of subtasks separated by " ; ".
	•	Each subtask must be necessary to solve the overall task, atomic (one operation/decision), unambiguous, and executable on its own.
	•	Do NOT solve anything. No extra lines.
""".strip()

    def subproblem_solving_prompt(self) -> str:
        """Sub-problem solving 단계 프롬프트를 반환."""
        return """
Sub-problem solving (strict):
	•	Output exactly one line per subtask.
	•	Line i must be EXACTLY: Solve: <result_i>
	•	<result_i> must be the direct output for subtask i.
	•	Do NOT add any other text.
""".strip()

    def extra_instructions(self, n_words: int) -> str:
        return f"""
Strategy (L2M)

{self.decomposition_prompt()}

{self.subproblem_solving_prompt()}
""".strip()


# =========================================================
# L2M-DV: decomposition + verification을 "한 번만", 루프 금지
# =========================================================
class L2MDVStrategy(L2MStrategy):
    @property
    def name(self) -> str:
        return "L2M-DV"

    def decomposition_prompt(self) -> str:
        """L2M의 decomposition 프롬프트 뒤에 verification criteria를 추가."""
        base = super().decomposition_prompt()
        additional = """
	•	Each item must be EXACTLY in this format: Step: <subtask> || Criteria: <verification_criteria>
	•	<verification_criteria> must be a concrete checklist-style condition that can be used to judge whether Step was correctly completed (must be specific and testable, not vague).
""".strip()
        return f"{base}\n{additional}"

    def subproblem_solving_prompt(self) -> str:
        """L2M의 subproblem solving 프롬프트 뒤에 verification을 추가."""
        base = super().subproblem_solving_prompt()
        additional = """
	•	Line i must be EXACTLY in this format: Solve: <result_i> || Check: <pass_or_fail>
	•	<pass_or_fail> MUST be exactly either "PASS" or "FAIL".
	•	Check must be determined ONLY by applying Criteria to <result_i>.
	•	If Check is "FAIL", <result_i> must be the best attempt for Step anyway (do not skip), and do not revise other steps.
""".strip()
        return f"{base}\n{additional}"

    def extra_instructions(self, n_words: int) -> str:
        return f"""
Strategy (L2M-DV)

{self.decomposition_prompt()}

{self.subproblem_solving_prompt()}
""".strip()
