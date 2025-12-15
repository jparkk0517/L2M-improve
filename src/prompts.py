"""L2M-CoT 프롬프트 및 데이터 풀 모듈."""

from src.utils import format_question, gold_answer


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
    """공통 태스크 블록을 생성."""
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
