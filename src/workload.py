"""Synthetic two-phase page-reference workload generation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Workload:
    """A complete reference trace and its exact workload-shift index."""

    trace: tuple[int, ...]
    shift_index: int


def _validate_configuration(
    page_count: int,
    frame_count: int,
    references_per_phase: int,
) -> None:
    if not isinstance(page_count, Integral) or page_count < 3:
        raise ValueError("page_count must be an integer of at least 3")
    if not isinstance(frame_count, Integral) or frame_count < 2:
        raise ValueError("frame_count must be an integer of at least 2")
    if frame_count >= page_count:
        raise ValueError("frame_count must be smaller than page_count")
    if not isinstance(references_per_phase, Integral) or references_per_phase < 25:
        raise ValueError("references_per_phase must be an integer of at least 25")


def generate_workload(
    page_count: int,
    frame_count: int,
    references_per_phase: int = 1000,
    seed: int = 42,
) -> Workload:
    """Generate a reproducible locality-heavy to random/bursty trace."""

    _validate_configuration(page_count, frame_count, references_per_phase)
    rng = np.random.default_rng(seed)

    working_set_size = min(page_count, max(2, min(frame_count, 4)))
    working_set = tuple(range(1, working_set_size + 1))
    cursor = 0
    before_shift: list[int] = []

    for _ in range(references_per_phase):
        if rng.random() < 0.85:
            page = working_set[cursor]
            cursor = (cursor + 1) % working_set_size
        else:
            page = int(rng.integers(1, page_count + 1))
        before_shift.append(page)

    after_shift: list[int] = []
    page_universe = np.arange(1, page_count + 1)
    hot_set_size = min(3, page_count)
    hot_set = rng.choice(page_universe, size=hot_set_size, replace=False)

    for index in range(references_per_phase):
        if index % 25 == 0:
            hot_set = rng.choice(page_universe, size=hot_set_size, replace=False)
        if rng.random() < 0.70:
            page = int(rng.integers(1, page_count + 1))
        else:
            page = int(rng.choice(hot_set))
        after_shift.append(page)

    trace = tuple(before_shift + after_shift)
    return Workload(trace=trace, shift_index=references_per_phase)
