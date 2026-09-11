"""
Model-supported token bounds
============================
A pre-call reservation must never be smaller than what the provider can bill
for that call. A character heuristic cannot promise that: emoji, CJK and other
non-ASCII text tokenize to far more than one token per three characters. The
reservation is therefore computed from an explicit bound per provider:

- OpenAI chat models: the exact encoding tiktoken maps to the configured model
  (o200k_base for the gpt-4o family), plus a fixed framing margin for the chat
  message envelope, plus the configured output cap that the provider enforces
  through ``max_tokens``.
- Ollama: the UTF-8 byte length of the prompt. Byte-level BPE tokenizers and
  SentencePiece tokenizers with byte fallback (Llama 3, Mistral) never emit
  more tokens than input bytes, so bytes are a demonstrable upper bound; real
  usage is much lower and no measured usage is reported on that path.

An OpenAI model that tiktoken cannot map, an encoding that cannot be loaded
(for example offline without the build-time cache) or a missing tiktoken
package fails closed: the engine reports the category and never calls the
provider.
"""

from typing import Protocol

# OpenAI's published num_tokens_from_messages counts 3 tokens per message plus
# 3 priming tokens for the assistant reply: 6 for the single message sent here.
# 16 leaves margin for envelope changes without hiding a real overshoot.
FRAMING_TOKENS = 16


class TokenBoundError(RuntimeError):
    """No model-supported token bound is available for this configuration."""


class TokenBound(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def reservation(self, prompt: str, max_output_tokens: int) -> int: ...


class TiktokenBound:
    """Exact tiktoken count for the configured OpenAI model, plus framing and the output cap."""

    def __init__(self, model: str):
        import tiktoken  # ImportError propagates: the engine reports missing_dependency

        if not isinstance(model, str) or not model.strip():
            raise TokenBoundError("The model name is empty.")
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError as error:
            raise TokenBoundError("tiktoken has no encoding for the configured model.") from error
        except Exception as error:
            raise TokenBoundError("The tiktoken encoding could not be loaded.") from error
        self.name = f"tiktoken/{self._encoding.name}"

    def count(self, text: str) -> int:
        # Special-token strings inside user text are ordinary text here; never raise on them.
        return len(self._encoding.encode(text, disallowed_special=()))

    def reservation(self, prompt: str, max_output_tokens: int) -> int:
        return self.count(prompt) + FRAMING_TOKENS + max_output_tokens


class ByteLengthBound:
    """UTF-8 bytes as an upper bound for byte-level tokenizers (Ollama models)."""

    name = "utf8-bytes"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def reservation(self, prompt: str, max_output_tokens: int) -> int:
        return self.count(prompt) + FRAMING_TOKENS + max_output_tokens


def token_bound_for(provider: str, model: str) -> TokenBound:
    """Resolve the bound for a supported provider; anything else fails closed."""
    if provider == "openai":
        return TiktokenBound(model)
    if provider == "ollama":
        return ByteLengthBound()
    raise TokenBoundError("No token bound is defined for this provider.")
