"""L2M-CoT 데이터셋 생성 모듈."""

import random
from typing import List, Dict

from src.prompts import COMPLEX_WORDS_POOL
from src.utils import format_question, gold_answer


def sample_words(n: int, rng: random.Random) -> List[str]:
    """단어 풀에서 n개의 단어를 무작위 샘플링."""
    return rng.sample(COMPLEX_WORDS_POOL, k=n)


def build_dataset(
    max_len: int, trials_per_len: int, seed: int
) -> List[Dict[str, object]]:
    """지정된 파라미터로 데이터셋을 생성."""
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
