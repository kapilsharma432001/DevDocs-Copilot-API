import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL, temperature=0
    )  # temperature 0 for stable and factual answers
