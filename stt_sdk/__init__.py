"""
stt_sdk: Async client(s) for your STT service.
"""

from .client import (
    STTClientConfig,
    STTClient,
    STTResult,
    STTError,
    STTHTTPError,
    STTPool,
)

__all__ = [
    "STTClientConfig",
    "STTClient",
    "STTResult",
    "STTError",
    "STTHTTPError",
    "STTPool",
]

# Single source of truth for version (used by pyproject via setuptools dynamic attr)
__version__ = "0.1.0"