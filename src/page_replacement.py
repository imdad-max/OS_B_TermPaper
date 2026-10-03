"""Classical page-replacement simulators used by the experiment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class SimulationResult:
    """Summary and per-reference events from one policy simulation."""

    algorithm: str
    hits: int
    faults: int
    events: tuple[bool, ...]
    evictions: tuple[int, ...]


def _validated_trace(trace: Sequence[int], frame_count: int) -> tuple[int, ...]:
    if not isinstance(frame_count, Integral) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if not trace:
        raise ValueError("trace must contain at least one page reference")
    if any(not isinstance(page, Integral) or page <= 0 for page in trace):
        raise ValueError("page identifiers must be positive integers")
    return tuple(int(page) for page in trace)


def _result(
    algorithm: str,
    events: list[bool],
    evictions: list[int],
) -> SimulationResult:
    hits = sum(events)
    return SimulationResult(
        algorithm=algorithm,
        hits=hits,
        faults=len(events) - hits,
        events=tuple(events),
        evictions=tuple(evictions),
    )


def simulate_fifo(trace: Sequence[int], frame_count: int) -> SimulationResult:
    """Evict the resident page that entered memory first."""

    references = _validated_trace(trace, frame_count)
    frames: set[int] = set()
    arrival_order: deque[int] = deque()
    events: list[bool] = []
    evictions: list[int] = []

    for page in references:
        if page in frames:
            events.append(True)
            continue

        events.append(False)
        if len(frames) == frame_count:
            victim = arrival_order.popleft()
            frames.remove(victim)
            evictions.append(victim)

        frames.add(page)
        arrival_order.append(page)

    return _result("FIFO", events, evictions)


def choose_optimal_victim(
    resident_pages: Sequence[int],
    future_trace: Sequence[int],
) -> int:
    """Choose Belady's victim from resident pages."""

    if not resident_pages:
        raise ValueError("resident_pages must not be empty")

    farthest_distance = -1.0
    victim: int | None = None
    for page in sorted(int(candidate) for candidate in resident_pages):
        try:
            distance = float(future_trace.index(page))
        except ValueError:
            distance = float("inf")
        if distance > farthest_distance:
            farthest_distance = distance
            victim = page

    if victim is None:  # Defensive guard for type checkers.
        raise RuntimeError("unable to choose an optimal victim")
    return victim


def simulate_lru(trace: Sequence[int], frame_count: int) -> SimulationResult:
    """Evict the resident page used least recently."""

    references = _validated_trace(trace, frame_count)
    frames: set[int] = set()
    last_access: dict[int, int] = {}
    events: list[bool] = []
    evictions: list[int] = []

    for index, page in enumerate(references):
        if page in frames:
            events.append(True)
            last_access[page] = index
            continue

        events.append(False)
        if len(frames) == frame_count:
            victim = min(frames, key=lambda candidate: (last_access[candidate], candidate))
            frames.remove(victim)
            last_access.pop(victim)
            evictions.append(victim)

        frames.add(page)
        last_access[page] = index

    return _result("LRU", events, evictions)


def simulate_optimal(trace: Sequence[int], frame_count: int) -> SimulationResult:
    """Evict the page whose next reference is farthest in the future."""

    references = _validated_trace(trace, frame_count)
    frames: set[int] = set()
    events: list[bool] = []
    evictions: list[int] = []

    for index, page in enumerate(references):
        if page in frames:
            events.append(True)
            continue

        events.append(False)
        if len(frames) == frame_count:
            victim = choose_optimal_victim(frames, references[index + 1 :])
            frames.remove(victim)
            evictions.append(victim)

        frames.add(page)

    return _result("Optimal", events, evictions)
