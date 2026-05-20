"""SPARTA ALIGNMENT – Self-Play Arena for Reinforcement Training and Alignment."""

from sparta.config import SpartaConfig


def __getattr__(name):
    """Lazy-load heavy imports (SpartaAlignment triggers ML deps)."""
    if name == "SpartaAlignment":
        from sparta.alignment import SpartaAlignment
        return SpartaAlignment
    raise AttributeError(f"module 'sparta' has no attribute {name}")


__all__ = ["SpartaConfig", "SpartaAlignment"]
