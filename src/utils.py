"""L2M-CoT 유틸리티 함수 모듈."""

import re
from typing import List

from src.config import client, MODEL_NAME, TEMPERATURE


# =========================================================
# (평가용) utilities  --- ✅ 모델 풀이에 쓰지 않음
# =========================================================
def get_last_alpha_letter(word: str) -> str:
    """단어에서 마지막 알파벳 문자를 소문자로 반환."""
    letters = re.findall(r"[A-Za-z]", word)
    return letters[-1].lower() if letters else ""


def gold_answer(words: List[str]) -> str:
    """✅ gold label 생성용(rule-based). 평가에만 사용."""
    return "".join(get_last_alpha_letter(w) for w in words)


def normalize_answer(ans: str) -> str:
    """답변을 소문자 알파벳만 남기도록 정규화."""
    ans = ans.lower()
    return re.sub(r"[^a-z]", "", ans)


def format_question(words: List[str]) -> str:
    """단어 리스트를 질문 형식으로 포맷팅."""
    return "Words: " + " ".join(words)


def call_model(prompt: str) -> str:
    """LLM에 프롬프트를 전달하고 응답을 반환."""
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
    """normalize된 s(알파벳만)에서 'finalanswer'가 연속으로 등장하는 부분 제거."""
    while s.startswith("finalanswer"):
        s = s[len("finalanswer") :]
    return s


def extract_final_answer(text: str, n_words: int) -> str:
    """텍스트에서 최종 답변을 추출."""
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
