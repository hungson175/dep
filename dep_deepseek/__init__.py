"""Typed decisions from DeepSeek Flash; no model-specific setup required."""
from .client import DeepSeek
from deepseek_flash import APIError, DistributionError, TransportError

__version__ = "0.1.0"
__all__ = ["DeepSeek", "APIError", "DistributionError", "TransportError"]
