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

    def extra_instructions(self, n_words: int) -> str:
        return f"""
Strategy (L2M)

Decomposition (strict):
- Output exactly ONE line starting with "Decomposition:".
- After it, write exactly {n_words} items separated by " ; ".
- Item i must be EXACTLY:
  Wi: get the last letter of the i-th word
- Do NOT solve anything. No extra lines.

Sub-problem solving (strict):
- Output exactly {n_words} lines.
- Line i must be EXACTLY:
  Solve<i>: <c>
- <c> MUST be exactly ONE lowercase letter a-z (no spaces, no punctuation).
- Do NOT add any other text.

Final (strict):
- Output exactly one final line:
  Final answer: <answer>
- <answer> must be exactly {n_words} lowercase letters a-z.
- Output nothing after the final line.
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
- Output exactly ONE line starting with "Decomposition:".
- After it, write exactly {n_words} items separated by " ; " (semicolon).
- Item i must be:
  Wi=<the i-th word> {{criteria: output exactly 1 lowercase letter which is the last character of Wi}}
- Do NOT solve anything else. No extra lines.

Solving + verification (strict):
- Output exactly {n_words} lines.
- Line i must be EXACTLY:
  Solve<i>: <c> || Check<i>: P|F
- <c> MUST be exactly ONE lowercase letter a-z (no spaces).
- Check<i> is P if <c> meets the criteria, otherwise F.
- Do NOT add Why. Do NOT add any extra text.

Final (strict):
- After the {n_words} lines, output exactly one final line:
  Final answer: <answer>
- <answer> must be exactly {n_words} lowercase letters a-z.
- Output nothing after the final line.
""".strip()
