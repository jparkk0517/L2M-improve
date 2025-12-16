"""실험 결과 저장 모듈.

실험 실행 시마다 타임스탬프 기반 폴더를 생성하고,
실험 조건, 로그, 결과 그래프를 저장합니다.
"""

import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from src.config import (
    MODEL_NAME,
    MAX_WORDS,
    TRIALS_PER_LENGTH,
    BATCH_SIZE,
    MAX_CONCURRENT_API,
    TEMPERATURE,
)


class ExperimentLogger:
    """실험 결과를 저장하는 로거 클래스."""

    def __init__(self, base_dir: str = "experiments"):
        """
        Args:
            base_dir: 실험 결과를 저장할 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.experiment_dir: Optional[Path] = None
        self._log_buffer = StringIO()
        self._original_stdout = None

    def start_experiment(self, experiment_name: Optional[str] = None) -> Path:
        """새 실험 시작 및 폴더 생성.

        Args:
            experiment_name: 실험 이름 (None이면 타임스탬프 사용)

        Returns:
            생성된 실험 폴더 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{experiment_name}" if experiment_name else timestamp

        self.experiment_dir = self.base_dir / folder_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 실험 조건 저장
        self._save_config()

        # stdout 캡처 시작
        self._start_log_capture()

        return self.experiment_dir

    def _save_config(self) -> None:
        """실험 조건을 JSON 파일로 저장."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "max_words": MAX_WORDS,
            "trials_per_length": TRIALS_PER_LENGTH,
            "batch_size": BATCH_SIZE,
            "max_concurrent_api": MAX_CONCURRENT_API,
            "temperature": TEMPERATURE,
        }

        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"📁 실험 폴더: {self.experiment_dir}")
        print(f"📄 실험 조건 저장: {config_path}")

    def _start_log_capture(self) -> None:
        """stdout 캡처 시작."""
        self._original_stdout = sys.stdout
        self._log_buffer = StringIO()
        sys.stdout = _TeeOutput(self._original_stdout, self._log_buffer)

    def get_results_path(self, filename: str = "accuracy_comparison.png") -> str:
        """결과 그래프 저장 경로 반환."""
        if self.experiment_dir is None:
            raise RuntimeError("start_experiment()를 먼저 호출하세요.")
        return str(self.experiment_dir / filename)

    def finish_experiment(self) -> None:
        """실험 종료 및 로그 저장."""
        if self.experiment_dir is None:
            return

        # stdout 복원
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout

        # 로그 저장
        log_path = self.experiment_dir / "experiment.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(self._log_buffer.getvalue())

        print(f"\n📝 실험 로그 저장: {log_path}")
        print("✅ 실험 완료!")


class _TeeOutput:
    """stdout을 캡처하면서 동시에 원래 stdout에도 출력."""

    def __init__(self, original: Any, buffer: StringIO):
        self.original = original
        self.buffer = buffer

    def write(self, data: str) -> int:
        self.original.write(data)
        self.buffer.write(data)
        return len(data)

    def flush(self) -> None:
        self.original.flush()
        self.buffer.flush()


# 전역 인스턴스
_logger: Optional[ExperimentLogger] = None


def get_logger() -> ExperimentLogger:
    """전역 로거 인스턴스 반환."""
    global _logger
    if _logger is None:
        _logger = ExperimentLogger()
    return _logger
