"""L2M-CoT 설정 및 OpenAI 클라이언트 모듈."""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
