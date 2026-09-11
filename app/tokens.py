"""
Model-supported token bounds
============================
A pre-call reservation must never be smaller than what the provider can bill
for that call. A character heuristic cannot promise that: emoji, CJK and other
non-ASCII text tokenize to far more than one token per three characters. The
reservation is therefore computed only from an explicit, model-supported bound:

- OpenAI chat models: the exact encoding tiktoken maps to the configured model
  (o200k_base for the gpt-4o family), plus a fixed framing margin for the chat
  message envelope, plus the configured output cap that the provider enforces
  through ``max_tokens``. The request carries exactly the prompt built here,
  so counting that prompt bounds the billed input.
- Ollama: no supported bound. The server wraps the supplied prompt in the
  model's Modelfile TEMPLATE and SYSTEM text (https://docs.ollama.com/modelfile),
  which this demo cannot see, count or bound from the client side, and the
  ``raw`` mode of https://docs.ollama.com/api/generate is not used here. Until
  a model/template-specific bound exists, Ollama fails closed as
  ``token_bound_unavailable`` and no local model is ever invoked.

An OpenAI model that tiktoken cannot map, an encoding that cannot be loaded
(for example offline without the build-time cache) or a missing tiktoken
package fails closed the same way: the engine reports the category and never
calls the provider.
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


def token_bound_for(provider: str, model: str) -> TokenBound:
    """Resolve the bound for a supported provider; every other configuration fails closed."""
    if provider == "openai":
        return TiktokenBound(model)
    if provider == "ollama":
        raise TokenBoundError(
            "Ollama applies a server-side Modelfile TEMPLATE/SYSTEM outside the supplied "
            "prompt, so the input cannot be bounded from this client; unsupported."
        )
    raise TokenBoundError("No token bound is defined for this provider.")
