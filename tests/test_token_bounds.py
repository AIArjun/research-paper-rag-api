"""Reservation bound verified against the real, pinned tiktoken encoding.

Skipped only where tiktoken (or its encoding, offline without the build-time
cache) is unavailable; the real image caches the encodings at build time.
"""

import os

import pytest

tiktoken = pytest.importorskip("tiktoken")

from app.config import MAX_QUESTION_CHARS  # noqa: E402
from app.tokens import (  # noqa: E402
    FRAMING_TOKENS, ByteLengthBound, TiktokenBound, TokenBoundError, token_bound_for,
)

SAMPLES = [
    "What accuracy did the model achieve on the benchmark?",
    "🙂" * 50,
    "日本語のテキストを含む質問です。注意機構とは何ですか。",
    "ما هي آلية الانتباه في الشبكات العصبية؟",
    "é" * 40 + " combining marks " + "‍​" * 10,
    "Mixed: attention 注意 внимание 🙂 \t\t\n\n   spaced   out   ",
    "<|endoftext|> inside user text <|im_start|>",
    "Ünïcödé " * 100,
    "a" * 3000,
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝖘𝖞𝖒𝖇𝖔𝖑𝖘 " * 30,
]


@pytest.fixture(scope="module")
def encoding():
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as error:
        if os.environ.get("TIKTOKEN_CACHE_DIR"):
            # The real image bakes the tables at build time; a miss there is a real failure, never a skip.
            raise AssertionError("The baked tiktoken cache did not provide o200k_base") from error
        pytest.skip(f"tiktoken encoding unavailable here: {type(error).__name__}")


def test_pinned_tokenizer_is_the_one_under_test(encoding):
    assert isinstance(tiktoken.__version__, str) and tiktoken.__version__
    assert encoding.name == "o200k_base"
    assert TiktokenBound("gpt-4o-mini").name == "tiktoken/o200k_base"


@pytest.mark.parametrize("sample", SAMPLES)
def test_reservation_equals_the_exact_count_plus_framing_and_output_cap(encoding, sample):
    bound = TiktokenBound("gpt-4o-mini")
    exact = len(encoding.encode(sample, disallowed_special=()))
    assert bound.count(sample) == exact
    assert bound.reservation(sample, 400) == exact + FRAMING_TOKENS + 400
    # The byte bound used for Ollama dominates the real tokenizer on every sample.
    assert ByteLengthBound().count(sample) >= exact


def test_the_replaced_character_heuristic_under_reserved_non_ascii_text(encoding):
    for sample in ("🙂" * 500, "日本語のテキスト" * 50, "Ünïcödé " * 100):
        exact = len(encoding.encode(sample))
        assert exact > len(sample) // 3 + 1, sample[:12]
        assert TiktokenBound("gpt-4o-mini").count(sample) == exact
    # Reviewer reproduction: 2000 CJK characters or 2000 emoji tokenize to about 2060
    # input tokens, where the old heuristic reserved about 762 for the whole prompt.
    for question in ("注" * MAX_QUESTION_CHARS, "🙂" * MAX_QUESTION_CHARS):
        prompt = "Context:\n" + "evidence " * 40 + "\n\nQuestion: " + question + "\n\nAnswer:"
        exact = len(encoding.encode(prompt))
        assert exact >= MAX_QUESTION_CHARS
        assert len(prompt) // 3 + 1 < exact
        assert TiktokenBound("gpt-4o-mini").reservation(prompt, 400) == exact + FRAMING_TOKENS + 400


def test_worst_case_question_and_context_reservation_is_finite_and_covered(encoding):
    prompt = "Context:\n" + "🙂" * 6000 + "\n\nQuestion: " + "🙂" * MAX_QUESTION_CHARS + "\n\nAnswer:"
    bound = TiktokenBound("gpt-4o-mini")
    reservation = bound.reservation(prompt, 400)
    assert reservation == len(encoding.encode(prompt)) + FRAMING_TOKENS + 400
    assert reservation <= ByteLengthBound().reservation(prompt, 400)


def test_unmapped_models_special_tokens_and_providers(encoding):
    with pytest.raises(TokenBoundError):
        TiktokenBound("not-a-real-model-name-xyz")
    with pytest.raises(TokenBoundError):
        TiktokenBound("   ")
    assert TiktokenBound("gpt-4o-mini").count("<|endoftext|>") > 0
    assert token_bound_for("openai", "gpt-4o-mini").name == "tiktoken/o200k_base"
    assert token_bound_for("ollama", "llama3").name == "utf8-bytes"
    with pytest.raises(TokenBoundError):
        token_bound_for("demo", "gpt-4o-mini")
