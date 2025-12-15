"""L2M-CoT 평가 로직 모듈."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict

from src.strategies import SolverStrategy, StrategyResult
from src.utils import normalize_answer


@dataclass
class Result:
    """평가 결과를 담는 데이터 클래스."""

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
    """예측값이 정답인지 확인."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    return (p == g) and (len(p) == n_words)


def log_accuracy(label: str, gold: str, pred: str) -> None:
    """정확도를 로그로 출력."""
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


def evaluate(
    dataset: List[Dict[str, object]],
    strategies: List[SolverStrategy],
    use_fewshot: bool,
) -> List[Result]:
    """데이터셋에 대해 전략들을 평가."""
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
    """위치별 정확도 통계를 계산."""
    g = normalize_answer(gold)
    p = normalize_answer(pred)

    total = len(g)
    compared = min(len(g), len(p))
    correct = sum(1 for i in range(compared) if g[i] == p[i])

    # 분모는 gold 길이 (pred가 짧으면 자동으로 불리)
    pct = (correct / total * 100.0) if total else 0.0
    return correct, total, pct


def macro_micro(results: List[Result], attr_name: str):
    """Macro/Micro 평균 정확도를 계산."""
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
    """결과를 요약하여 출력."""
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
# ✅ 분석 및 시각화 함수
# =========================================================
STRATEGY_NAMES = ["Baseline", "CoT", "L2M", "L2M-DV"]
STRATEGY_PRED_ATTRS = ["baseline_pred", "cot_pred", "l2m_pred", "l2mdv_pred"]


def get_accuracy_by_n_words(results: List[Result]) -> Dict[int, Dict[str, float]]:
    """단어 갯수별 각 전략의 정확도를 계산.

    Returns:
        Dict[int, Dict[str, float]]: {n_words: {strategy_name: accuracy%}}
        예: {1: {"Baseline": 100.0, "CoT": 100.0, ...}, 2: {...}, ...}
    """
    from collections import defaultdict

    # n_words별로 결과 그룹화
    by_n_words: Dict[int, List[Result]] = defaultdict(list)
    for r in results:
        by_n_words[r.n_words].append(r)

    accuracy_table: Dict[int, Dict[str, float]] = {}

    for n_words in sorted(by_n_words.keys()):
        group = by_n_words[n_words]
        accuracy_table[n_words] = {}

        for name, attr in zip(STRATEGY_NAMES, STRATEGY_PRED_ATTRS):
            # 해당 그룹에서 각 전략의 position-wise accuracy 계산
            correct_sum = 0
            total_sum = 0
            for r in group:
                pred = getattr(r, attr)
                correct, total, _ = position_acc_stats(r.gold, pred)
                correct_sum += correct
                total_sum += total

            acc = (correct_sum / total_sum * 100.0) if total_sum else 0.0
            accuracy_table[n_words][name] = acc

    return accuracy_table


def get_summary_stats(results: List[Result]) -> Dict[str, Dict[str, float]]:
    """전체 평균 정확도 통계를 반환.

    Returns:
        Dict[str, Dict[str, float]]: {strategy_name: {"macro": %, "micro": %}}
    """
    stats = {}
    for name, attr in zip(STRATEGY_NAMES, STRATEGY_PRED_ATTRS):
        macro, micro = macro_micro(results, attr)
        stats[name] = {"macro": macro, "micro": micro}
    return stats


def print_accuracy_table(results: List[Result]) -> None:
    """단어 갯수별 정확도를 테이블 형식으로 출력."""
    acc_table = get_accuracy_by_n_words(results)
    summary = get_summary_stats(results)

    # 헤더 출력
    print("\n" + "=" * 70)
    print("Position-wise Accuracy by Number of Words (%)")
    print("=" * 70)
    header = f"{'n_words':>8} | " + " | ".join(f"{name:>10}" for name in STRATEGY_NAMES)
    print(header)
    print("-" * 70)

    # 각 n_words별 정확도 출력
    for n_words in sorted(acc_table.keys()):
        row_data = acc_table[n_words]
        row = f"{n_words:>8} | " + " | ".join(
            f"{row_data[name]:>10.2f}" for name in STRATEGY_NAMES
        )
        print(row)

    # 전체 평균 출력
    print("-" * 70)
    micro_row = f"{'Micro':>8} | " + " | ".join(
        f"{summary[name]['micro']:>10.2f}" for name in STRATEGY_NAMES
    )
    macro_row = f"{'Macro':>8} | " + " | ".join(
        f"{summary[name]['macro']:>10.2f}" for name in STRATEGY_NAMES
    )
    print(micro_row)
    print(macro_row)
    print("=" * 70)


def plot_accuracy_comparison(
    results: List[Result], save_path: str = None, show: bool = True
) -> None:
    """단어 갯수별 전략 정확도를 그래프로 시각화.

    Args:
        results: 평가 결과 리스트
        save_path: 저장할 파일 경로 (None이면 저장 안 함)
        show: 그래프를 화면에 표시할지 여부
    """
    import matplotlib.pyplot as plt

    acc_table = get_accuracy_by_n_words(results)
    summary = get_summary_stats(results)

    n_words_list = sorted(acc_table.keys())

    # 각 전략별 데이터 준비
    strategy_data = {name: [] for name in STRATEGY_NAMES}
    for n in n_words_list:
        for name in STRATEGY_NAMES:
            strategy_data[name].append(acc_table[n][name])

    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 라인 그래프: 단어 갯수별 정확도 추이
    ax1 = axes[0]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    markers = ["o", "s", "^", "D"]

    for i, name in enumerate(STRATEGY_NAMES):
        ax1.plot(
            n_words_list,
            strategy_data[name],
            label=name,
            color=colors[i],
            marker=markers[i],
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("Number of Words", fontsize=12)
    ax1.set_ylabel("Position-wise Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy by Number of Words", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(n_words_list)

    # 2. 막대 그래프: 전체 평균 비교
    ax2 = axes[1]
    x = range(len(STRATEGY_NAMES))
    width = 0.35

    macro_vals = [summary[name]["macro"] for name in STRATEGY_NAMES]
    micro_vals = [summary[name]["micro"] for name in STRATEGY_NAMES]

    bars1 = ax2.bar(
        [i - width / 2 for i in x],
        macro_vals,
        width,
        label="Macro Avg",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        micro_vals,
        width,
        label="Micro Avg",
        color="#e74c3c",
        alpha=0.8,
    )

    ax2.set_xlabel("Strategy", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Overall Average Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(STRATEGY_NAMES)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 105)

    # 막대 위에 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n그래프가 저장되었습니다: {save_path}")

    if show:
        plt.show()
