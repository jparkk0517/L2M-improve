"""L2M-CoT 설정 및 OpenAI 클라이언트 모듈."""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# 설정
# =========================================================
MODEL_NAME = "openai/gpt-oss-20b"
# MODEL_NAME = "gpt-4o"
MAX_WORDS = 15
TRIALS_PER_LENGTH = 20  # BATCH_SIZE와 동일하게 설정 권장
RANDOM_SEED = 42
TEMPERATURE = 0

# 배치 처리 설정
BATCH_SIZE = 20  # 각 num_words당 동시 실행할 문제 수
MAX_CONCURRENT_API = 4  # 동시 API 호출 제한 (Rate Limit 방지)
base_url = "http://localhost:1234/v1" if MODEL_NAME == "openai/gpt-oss-20b" else None
api_key = "lm-studio" if MODEL_NAME == "openai/gpt-oss-20b" else None
client = OpenAI(base_url=base_url, api_key=api_key)
