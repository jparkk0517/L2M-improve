from dotenv import load_dotenv

load_dotenv()

import random
import re
from dataclasses import dataclass, field
from typing import List, Dict
from abc import ABC, abstractmethod

from openai import OpenAI


# =========================================================
# 설정
# =========================================================
MODEL_NAME = "gpt-4o-mini"
MAX_WORDS = 15
TRIALS_PER_LENGTH = 1
RANDOM_SEED = 42
TEMPERATURE = 0
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # LM Studio는 보통 아무 문자열이어도 됨
)


# =========================================================
# (평가용) utilities  --- ✅ 모델 풀이에 쓰지 않음
# =========================================================
def get_last_alpha_letter(word: str) -> str:
    letters = re.findall(r"[A-Za-z]", word)
    return letters[-1].lower() if letters else ""


def gold_answer(words: List[str]) -> str:
    # ✅ gold label 생성용(rule-based). 평가에만 사용.
    return "".join(get_last_alpha_letter(w) for w in words)


def normalize_answer(ans: str) -> str:
    ans = ans.lower()
    return re.sub(r"[^a-z]", "", ans)


def format_question(words: List[str]) -> str:
    return "Words: " + " ".join(words)


def call_model(prompt: str) -> str:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=TEMPERATURE,
    )
    return resp.output_text.strip()


# =========================================================
# ✅ Robust answer extraction
# - reasoning이 섞여도 마지막 "Final answer:" 라인을 뽑아냄
# - 혹시 format이 깨져도 n_words 기준으로 후보를 추출하는 fallback 제공
# =========================================================
FINAL_PAT = re.compile(r"[Ff]inal\s*[Aa]nswer\s*:?\s*(.+)", re.IGNORECASE)


def _strip_finalanswer_runs(s: str) -> str:
    # normalize된 s(알파벳만)에서 'finalanswer'가 연속으로 등장하는 부분 제거
    # 1) 어디에 있든 모두 제거(가장 단순/강력)
    # return s.replace("finalanswer", "")

    # 2) 더 보수적으로: 앞에 붙은 경우만 반복 제거(추천)
    while s.startswith("finalanswer"):
        s = s[len("finalanswer") :]
    return s


def extract_final_answer(text: str, n_words: int) -> str:
    matches = FINAL_PAT.findall(text)
    if matches:
        cand = normalize_answer(matches[-1])
        cand = _strip_finalanswer_runs(cand)

        if len(cand) == n_words:
            return cand
        return cand  # 길이 mismatch면 평가에서 걸러짐

    # fallback: 길이 n_words의 영문 연속 문자열 후보를 뒤에서부터 탐색
    pattern = re.compile(rf"([A-Za-z]{{{n_words}}})")
    all_cands = pattern.findall(text)
    if all_cands:
        cand = normalize_answer(all_cands[-1])
        cand = _strip_finalanswer_runs(cand)
        return cand

    cand = normalize_answer(text)
    cand = _strip_finalanswer_runs(cand)
    return cand


# =========================================================
# Few-shot
# =========================================================
FEWSHOT_1 = ["apple", "banana", "pear"]
FEWSHOT_2 = ["dog", "cat", "pig"]


# =========================================================
# Dataset pool
# =========================================================
COMPLEX_WORDS_POOL = [
    "interoperability",
    "internationalization",
    "electroencephalography",
    "magnetoencephalography",
    "spectrophotometry",
    "chromatography",
    "electrophoresis",
    "thermodynamically",
    "electromagnetism",
    "photoacclimation",
    "microarchitecture",
    "neurotransmitter",
    "neuroplasticity",
    "psychophysiology",
    "neuropsychology",
    "immunohistochemistry",
    "immunofluorescence",
    "phosphorylation",
    "dephosphorylation",
    "glycosyltransferase",
    "deoxyribonucleotide",
    "ribonucleoprotein",
    "mitochondrion",
    "endoplasmicreticulum",
    "cytoskeleton",
    "extracellularmatrix",
    "microenvironment",
    "epigenetically",
    "transcriptionally",
    "posttranslational",
    "metalloprotease",
    "allosterically",
    "stoichiometric",
    "heteroscedasticity",
    "autocorrelation",
    "multicollinearity",
    "nonstationarity",
    "overparameterization",
    "regularization",
    "generalization",
    "initialization",
    "reparameterization",
    "quantization",
    "binarization",
    "vectorization",
    "normalization",
    "standardization",
    "tokenization",
    "lemmatization",
    "stemming",
    "disambiguation",
    "coreference",
    "entailment",
    "paraphrastic",
    "syntactically",
    "semantically",
    "pragmatically",
    "ontologically",
    "phenomenology",
    "epistemology",
    "metaphysical",
    "dialectically",
    "teleological",
    "deontological",
    "utilitarianism",
    "consequentialism",
    "incompatibilism",
    "compatibilism",
    "deterministically",
    "indeterminacy",
    "counterfactual",
    "interdependence",
    "incommensurable",
    "incontrovertible",
    "indistinguishable",
    "uncharacteristically",
    "intercontinental",
    "circumstantiality",
    "mischaracterization",
    "disproportionately",
    "unconstitutionality",
    "jurisprudential",
    "extrajudicial",
    "interlocutory",
    "justiciability",
    "inapplicability",
    "irrevocability",
    "inviolability",
    "indefeasibility",
    "unavailability",
    "intercalibration",
    "reproducibility",
    "repeatability",
    "traceability",
    "verifiability",
    "falsifiability",
    "interdisciplinary",
    "multidisciplinary",
    "transdisciplinary",
    "interinstitutional",
    "interdepartmental",
    "interoperable",
    "misinterpretation",
    "misrepresentation",
    "miscommunication",
    "overgeneralization",
    "underestimation",
    "overestimation",
    "misclassification",
    "reclassification",
    "redistribution",
    "reconfiguration",
    "reconciliation",
    "recontextualization",
    "decontextualization",
    "contextualization",
    "characterization",
    "decharacterization",
    "conceptualization",
    "operationalization",
    "institutionalization",
    "compartmentalization",
    "instrumentalization",
    "sensationalization",
    "professionalization",
    "deprofessionalization",
    "interpersonal",
    "intrapersonal",
    "interoception",
    "proprioception",
    "exteroception",
    "interoceptive",
    "proprioceptive",
    "exteroceptive",
    "indistinctness",
    "inconclusiveness",
    "incompleteness",
    "inconsistency",
    "incompatibility",
]


# =========================================================
# ✅ 공통 Task 블록
# - "reasoning 허용 + 마지막 줄 Final answer 강제" 로 바꿈
# =========================================================
def build_common_task_block(use_fewshot: bool) -> str:
    base = f"""
Task: last letter concatenation

Task definition:
- Input line starts with "Words:" followed by tokens separated by SINGLE SPACES.
- Each token is ONE word as-is. Do NOT split tokens on hyphens. Example: "error-analysis" is one word.
- For each word token, take EXACTLY ONE character:
  the LAST alphabetic letter (A-Z or a-z) within that token.
- Convert that letter to lowercase.
- Concatenate these letters in order into one string with no spaces.

Sanity constraints:
- The output string length MUST equal the number of word tokens.
- Each output character corresponds to exactly one input word.

Output rule:
- You MAY output reasoning or working above.
- But the VERY LAST LINE MUST be exactly!!!!:
```Final answer: <string>```
- <string> must contain only lowercase letters a-z (no spaces).

Two examples:

Example 1
{format_question(FEWSHOT_1)}
Final answer: {gold_answer(FEWSHOT_1)}

Example 2
{format_question(FEWSHOT_2)}
Final answer: {gold_answer(FEWSHOT_2)}
""".strip()

    if use_fewshot:
        return base

    base_no_fs = base.split("\n\nTwo examples:", 1)[0].strip()
    return base_no_fs


# =========================================================
# Strategy Framework
# =========================================================
@dataclass
class StrategyResult:
    raw_output: str
    final_answer: str
    meta_info: Dict[str, object] = field(default_factory=dict)


class SolverStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def solve(
        self, question: str, use_fewshot: bool, n_words: int
    ) -> StrategyResult: ...


# =========================================================
# ✅ Baseline = base class (1-call)
# - 자식은 extra_instructions만 override해서 prompt만 바뀜
# =========================================================
class BaselineStrategy(SolverStrategy):
    @property
    def name(self) -> str:
        return "Baseline"

    def extra_instructions(self, n_words: int) -> str:
        return ""

    def build_prompt(self, question: str, use_fewshot: bool, n_words: int) -> str:
        common = build_common_task_block(use_fewshot)
        extra = self.extra_instructions(n_words).strip()
        extra_block = f"\n\n{extra}" if extra else ""
        return f"""{common}
{extra_block}

Now solve the following instance:
{question}

Reminder: final answer length must be {n_words}.
Make sure your VERY LAST LINE is 'Final answer: <string>'.
""".strip()

    def solve(self, question: str, use_fewshot: bool, n_words: int) -> StrategyResult:
        prompt = self.build_prompt(question, use_fewshot, n_words)
        raw = call_model(prompt)
        final = extract_final_answer(raw, n_words)
        return StrategyResult(
            raw_output=raw, final_answer=final, meta_info={"num_calls": 1}
        )


# =========================================================
# CoT: reasoning을 실제로 "출력"하도록 유도 (마지막 줄만 Final)
# =========================================================
class CoTStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "CoT"

    def extra_instructions(self, n_words: int) -> str:
        return """
Strategy (CoT):
- Think step by step and SHOW your reasoning above.
- Keep reasoning concise but explicit.
- The VERY LAST LINE must be: Final answer: <string>
""".strip()


# =========================================================
# L2M: 한 번의 응답 안에서 least-to-most 진행을 "표현"하게 함
# =========================================================
class L2MStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "L2M"

    def extra_instructions(self, n_words: int) -> str:
        return """
Strategy (L2M, single-call):
- Solve in a least-to-most style:
  1) list each token and its last alphabetic letter
  2) then concatenate
- Show this working above.
- The VERY LAST LINE must be: Final answer: <string>
""".strip()


# =========================================================
# L2M-DV: decomposition + criteria + verification을 "출력"하도록 유도
# =========================================================
class L2MDVStrategy(BaselineStrategy):
    @property
    def name(self) -> str:
        return "L2M-DV"

    def extra_instructions(self, n_words: int) -> str:
        return """
Strategy (L2M-DV, single-call):
- First, decompose into sub-problems per token.
- For each token, write:
  - sub-problem: get last alphabetic letter and lowercase it
  - criteria: result must be exactly one lowercase a-z AND must be the LAST alphabetic letter in the token
- Then solve each sub-problem.
- For each solved letter, explicitly verify it satisfies the criteria; if not, correct it.
- For Each solved letter, if the response of this step is not satisfied criteria then repeat the process until it is satisfied.
- Show this plan + verification above.
- The VERY LAST LINE must be: Final answer: <string>
""".strip()


# =========================================================
# Dataset
# =========================================================
def sample_words(n: int, rng: random.Random) -> List[str]:
    return rng.sample(COMPLEX_WORDS_POOL, k=n)


def build_dataset(
    max_len: int, trials_per_len: int, seed: int
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    dataset = []
    idx = 0
    for n in range(1, max_len + 1):
        for _ in range(trials_per_len):
            idx += 1
            words = sample_words(n, rng)
            dataset.append(
                {
                    "id": idx,
                    "n_words": n,
                    "words": words,
                    "question": format_question(words),
                    "answer": gold_answer(words),
                }
            )
    return dataset


# =========================================================
# Evaluation
# =========================================================
@dataclass
class Result:
    problem_id: int
    n_words: int
    question: str
    gold: str

    baseline_pred: str
    cot_pred: str
    l2m_pred: str
    l2mdv_pred: str

    baseline_correct: bool
    cot_correct: bool
    l2m_correct: bool
    l2mdv_correct: bool


def is_correct(pred: str, gold: str, n_words: int) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    return (p == g) and (len(p) == n_words)


def log_accuracy(label: str, gold: str, pred: str) -> None:
    g = normalize_answer(gold)
    p = normalize_answer(pred)
    compared = min(len(g), len(p))
    correct = sum(1 for i in range(compared) if g[i] == p[i])
    total = len(g)
    acc = (correct / total) * 100.0 if total else 0.0
    note = ""
    if len(p) != len(g):
        note = f" | len_mismatch(gold={len(g)}, pred={len(p)})"
    print(
        f"{label:8s} | pred='{p}' | pos_acc={correct}/{total} ({acc:.1f}%) | compared={compared}{note}"
    )


from concurrent.futures import ThreadPoolExecutor, as_completed


def evaluate(
    dataset: List[Dict[str, object]],
    strategies: List[SolverStrategy],
    use_fewshot: bool,
) -> List[Result]:
    results: List[Result] = []

    for item in dataset:
        pid = int(item["id"])
        n_words = int(item["n_words"])
        q = str(item["question"])
        gold = str(item["answer"])

        print("\n" + "=" * 70)
        print(f"Problem {pid} | num_words = {n_words}")
        print("Question:", q)
        print("Gold answer:", gold)

        # 병렬 실행
        outs: Dict[str, StrategyResult] = {}
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            future_to_strategy = {
                executor.submit(s.solve, q, use_fewshot, n_words): s for s in strategies
            }
            for future in as_completed(future_to_strategy):
                s = future_to_strategy[future]
                res = future.result()
                outs[s.name] = res
                log_accuracy(s.name, gold, res.final_answer)

        base = outs["Baseline"]
        cot = outs["CoT"]
        l2m = outs["L2M"]
        l2mdv = outs["L2M-DV"]

        results.append(
            Result(
                problem_id=pid,
                n_words=n_words,
                question=q,
                gold=gold,
                baseline_pred=base.final_answer,
                cot_pred=cot.final_answer,
                l2m_pred=l2m.final_answer,
                l2mdv_pred=l2mdv.final_answer,
                baseline_correct=is_correct(base.final_answer, gold, n_words),
                cot_correct=is_correct(cot.final_answer, gold, n_words),
                l2m_correct=is_correct(l2m.final_answer, gold, n_words),
                l2mdv_correct=is_correct(l2mdv.final_answer, gold, n_words),
            )
        )

    return results


def position_acc_stats(gold: str, pred: str):
    g = normalize_answer(gold)
    p = normalize_answer(pred)

    total = len(g)
    compared = min(len(g), len(p))
    correct = sum(1 for i in range(compared) if g[i] == p[i])

    # 분모는 gold 길이 (pred가 짧으면 자동으로 불리)
    pct = (correct / total * 100.0) if total else 0.0
    return correct, total, pct


# ===== 평균 정확도 (Position-wise) =====
def macro_micro(results, attr_name: str):
    # attr_name: "baseline_pred" / "cot_pred" / "l2m_pred" / "l2mdv_pred"
    pcts = []
    correct_sum = 0
    total_sum = 0

    for r in results:
        pred = getattr(r, attr_name)
        correct, total, pct = position_acc_stats(r.gold, pred)
        pcts.append(pct)
        correct_sum += correct
        total_sum += total

    macro = sum(pcts) / len(pcts) if pcts else 0.0
    micro = (correct_sum / total_sum * 100.0) if total_sum else 0.0
    return macro, micro


def summarize(results: List[Result], max_len: int) -> None:
    base_macro, base_micro = macro_micro(results, "baseline_pred")
    cot_macro, cot_micro = macro_micro(results, "cot_pred")
    l2m_macro, l2m_micro = macro_micro(results, "l2m_pred")
    dv_macro, dv_micro = macro_micro(results, "l2mdv_pred")

    print("-" * 70)
    print("Average position-wise accuracy (Macro / Micro)")
    print(f"Baseline : {base_macro:.2f}% / {base_micro:.2f}%")
    print(f"CoT      : {cot_macro:.2f}% / {cot_micro:.2f}%")
    print(f"L2M      : {l2m_macro:.2f}% / {l2m_micro:.2f}%")
    print(f"L2M-DV   : {dv_macro:.2f}% / {dv_micro:.2f}%")


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    dataset = build_dataset(MAX_WORDS, TRIALS_PER_LENGTH, RANDOM_SEED)

    strategies: List[SolverStrategy] = [
        BaselineStrategy(),
        CoTStrategy(),
        L2MStrategy(),
        L2MDVStrategy(),
    ]

    results = evaluate(dataset, strategies=strategies, use_fewshot=True)
    summarize(results, MAX_WORDS)
