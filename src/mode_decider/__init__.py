"""MoDe: small mixture-of-experts models that answer typed decision questions with a probability for every option."""

from mode_decider.model import MoDe, Unsupported
from mode_decider.schema import InputTooLong, InvalidRequest

__all__ = ["InputTooLong", "InvalidRequest", "MoDe", "Unsupported"]
__version__ = "0.1.0"
