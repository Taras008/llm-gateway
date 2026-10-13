class LLMError(Exception):
    """Base class for all LLM provider errors."""


class LLMUnavailableError(LLMError):
    """LLM server is not reachable."""


class LLMTimeoutError(LLMError):
    """LLM server did not respond in time."""


class LLMBadResponseError(LLMError):
    """LLM server returned an error or a response we cannot use."""
