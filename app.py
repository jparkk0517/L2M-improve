"""L2M-CoT 메인 진입점.

이 스크립트는 Last Letter Concatenation 문제에 대해
다양한 prompting 전략(Baseline, CoT, L2M, L2M-DV)을 평가합니다.
"""

from typing import List

from src.config import (
    MAX_WORDS,
    TRIALS_PER_LENGTH,
    RANDOM_SEED,
    BATCH_SIZE,
    MAX_CONCURRENT_API,
)
from src.dataset import build_dataset
from src.evaluation import (
    evaluate_batch,
    summarize,
    print_accuracy_table,
    plot_accuracy_comparison,
)
from src.strategies import (
    SolverStrategy,
    BaselineStrategy,
    CoTStrategy,
    L2MStrategy,
    L2MDVStrategy,
)


def main():
    """메인 함수: 데이터셋 생성 → 전략 평가 → 결과 요약."""
    dataset = build_dataset(MAX_WORDS, TRIALS_PER_LENGTH, RANDOM_SEED)

    strategies: List[SolverStrategy] = [
        BaselineStrategy(),
        CoTStrategy(),
        L2MStrategy(),
        L2MDVStrategy(),
    ]

    # 배치 처리 방식 평가
    results = evaluate_batch(
        dataset,
        strategies=strategies,
        use_fewshot=True,
        batch_size=BATCH_SIZE,
        max_concurrent=MAX_CONCURRENT_API,
    )

    # 기본 요약 출력
    summarize(results, MAX_WORDS)

    # 단어 갯수별 정확도 테이블 출력
    print_accuracy_table(results)

    # 그래프 생성 및 저장
    plot_accuracy_comparison(results, save_path="accuracy_comparison.png", show=True)


if __name__ == "__main__":
    main()
