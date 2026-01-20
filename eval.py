"""Compatibility wrapper for pointer_lora.eval -> pointer_lora.eval_runs.eval."""

from __future__ import annotations

from .eval_runs.eval import main

__all__ = ["main"]


if __name__ == "__main__":
    main()

