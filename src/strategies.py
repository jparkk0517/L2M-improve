"""L2M-CoT 전략 패턴 모듈."""

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
        raw = call_model(prompt)
        final = extract_final_answer(raw, n_words)
        return StrategyResult(
            raw_output=raw, final_answer=final, meta_info={"num_calls": 1}
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
- You MUST Think step by step.
- Write down which word you are processing and what its last letter is.
- Concatenate them step by step.
- Finally, write the Final answer line.
""".strip()


# =========================================================
# L2M: least-to-most 작업을 "엄격한 포맷"으로 출력
# =========================================================
class L2MStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "L2M"

    def extra_instructions(self, n_words: int) -> str:
        return f"""
Strategy (L2M)

Decomposition (strict):
- Write exactly one line that starts with "Decomposition:".
- After "Decomposition:", write exactly {n_words} subproblems separated by " | ".
- Each subproblem must be a short, meaningful step (concept-level), not token-level.
- Steps must be ordered from easiest/most basic to hardest/final.
- Do NOT solve anything.
- Do NOT include the character "|" inside any subproblem text.
- Do NOT add any extra lines.

Sub-problem solving (strict):
- Write exactly {n_words} lines, nothing else.
- Line i must start with "Solve<i>:" where <i> is 1..{n_words}.
- Each line must solve decomposition[i] in the same order.
- Each line must be a single line (no line breaks), concise and complete.
- You MUST use results from previous Solve lines when relevant.
- Do NOT restate the subproblem text. Do NOT add extra commentary.

Final (strict):
- After the {n_words} Solve lines, output exactly one final line:
  Final answer: <answer>
- Do NOT add anything after the final line.
""".strip()


# =========================================================
# L2M-DV: decomposition + verification을 "한 번만", 루프 금지
# =========================================================
class L2MDVStrategy(L2MStrategy):
    @property
    def name(self) -> str:
        return "L2M-DV"

    def extra_instructions(self, n_words: int) -> str:
        return f"""
Strategy (L2M-DV: Decomposition + Verification, no loops)

Decomposition (strict):
- Write exactly one line that starts with "Decomposition:".
- After "Decomposition:", write exactly {n_words} subproblems separated by " | ".
- Each subproblem MUST be in this exact format:
  <subproblem> {{criteria: <criteria>}}
- <subproblem> must be concept-level, short, and ordered least-to-most.
- <criteria> must be a concrete, checkable condition for success (not vague).
- Do NOT solve anything.
- Do NOT include the character "|" inside <subproblem> or <criteria>.
- Do NOT add any extra lines.

Sub-problem solving + verification (strict):
- Write exactly {n_words} lines, nothing else.
- Line i must be EXACTLY in this format:
  Solve<i>: <solution> || Check<i>: PASS|FAIL || Why<i>: <short reason>
- Solve<i> must solve the i-th decomposed subproblem (same order).
- Check<i> must judge whether <solution> meets the criteria written for step i.
- If FAIL, do NOT retry, do NOT revise earlier steps (no loops). Continue to the next step with best effort.
- You MUST use results from previous Solve lines when relevant.
- No extra commentary, no bullets, no numbering other than the required <i>.

Final (strict):
- After the {n_words} Solve lines, output exactly one final line:
  Final answer: <answer>
- Do NOT add anything after the final line.
""".strip()
