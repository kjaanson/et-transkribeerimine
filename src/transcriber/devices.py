from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "mps") -> str:
    preference = preference.lower().strip()

    if preference == "cpu":
        return "cpu"

    if preference == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        logger.warning("MPS requested but unavailable. Falling back to CPU.")
        return "cpu"

    if preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    logger.warning("Unknown device preference '%s'. Falling back to CPU.", preference)
    return "cpu"


def recommended_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32
