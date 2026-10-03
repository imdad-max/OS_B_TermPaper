"""Lightweight learned page-eviction policy and training-data generation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.page_replacement import SimulationResult, choose_optimal_victim


FEATURE_NAMES = (
    "recency",
    "total_frequency",
    "recent_frequency",
    "residency_time",
)


@dataclass(frozen=True)
class TrainingData:
    """Candidate rows, binary labels, and their eviction-decision groups."""

    features: np.ndarray
    labels: np.ndarray
    decision_ids: np.ndarray


def build_candidate_features(
    page: int,
    time_index: int,
    last_access: Mapping[int, int],
    access_counts: Mapping[int, int],
    recent_window: Sequence[int],
    loaded_at: Mapping[int, int],
) -> tuple[int, int, int, int]:
    """Describe one resident page using only current and historical state."""

    return (
        time_index - last_access[page],
        access_counts[page],
        sum(reference == page for reference in recent_window),
        time_index - loaded_at[page],
    )


def generate_training_data(
    traces: Sequence[Sequence[int]],
    frame_count: int,
) -> TrainingData:
    """Label resident-page candidates with deterministic Belady decisions."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2 for candidate classification")
    if not traces or any(not trace for trace in traces):
        raise ValueError("training traces must not be empty")

    feature_rows: list[tuple[int, int, int, int]] = []
    labels: list[int] = []
    decision_ids: list[int] = []
    decision_id = 0

    for trace in traces:
        references = tuple(int(page) for page in trace)
        if any(page <= 0 for page in references):
            raise ValueError("page identifiers must be positive integers")

        frames: set[int] = set()
        last_access: dict[int, int] = {}
        access_counts: dict[int, int] = {}
        loaded_at: dict[int, int] = {}
        recent_window: deque[int] = deque(maxlen=20)

        for time_index, page in enumerate(references):
            if page not in frames:
                if len(frames) == frame_count:
                    victim = choose_optimal_victim(
                        frames,
                        references[time_index + 1 :],
                    )
                    for candidate in sorted(frames):
                        feature_rows.append(
                            build_candidate_features(
                                candidate,
                                time_index,
                                last_access,
                                access_counts,
                                recent_window,
                                loaded_at,
                            )
                        )
                        labels.append(int(candidate == victim))
                        decision_ids.append(decision_id)
                    decision_id += 1
                    frames.remove(victim)
                    loaded_at.pop(victim)

                frames.add(page)
                loaded_at[page] = time_index

            access_counts[page] = access_counts.get(page, 0) + 1
            last_access[page] = time_index
            recent_window.append(page)

    features_array = np.asarray(feature_rows, dtype=float).reshape(-1, len(FEATURE_NAMES))
    return TrainingData(
        features=features_array,
        labels=np.asarray(labels, dtype=int),
        decision_ids=np.asarray(decision_ids, dtype=int),
    )


def train_eviction_model(
    training_data: TrainingData,
    random_state: int,
) -> DecisionTreeClassifier:
    """Train the deliberately small and interpretable eviction classifier."""

    if training_data.features.size == 0 or training_data.labels.size == 0:
        raise ValueError("training data must contain candidate samples")
    if set(training_data.labels.tolist()) != {0, 1}:
        raise ValueError("training labels must contain both classes 0 and 1")

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(training_data.features, training_data.labels)
    return model


def choose_learned_victim(
    candidates: Sequence[int],
    feature_rows: Sequence[Sequence[float]],
    eviction_probabilities: Sequence[float],
) -> int:
    """Choose the highest-scoring candidate with deterministic tie-breaking."""

    if not candidates or not (
        len(candidates) == len(feature_rows) == len(eviction_probabilities)
    ):
        raise ValueError("candidates, features, and probabilities must align")

    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            float(eviction_probabilities[index]),
            float(feature_rows[index][0]),
            -int(candidates[index]),
        ),
    )
    return int(candidates[best_index])


def simulate_learned(
    trace: Sequence[int],
    frame_count: int,
    model: DecisionTreeClassifier,
) -> SimulationResult:
    """Simulate online decisions using only past-derived candidate features."""

    if not isinstance(frame_count, Integral) or frame_count < 2:
        raise ValueError("frame_count must be an integer of at least 2")
    if not trace:
        raise ValueError("trace must contain at least one page reference")
    references = tuple(int(page) for page in trace)
    if any(page <= 0 for page in references):
        raise ValueError("page identifiers must be positive integers")
    try:
        positive_column = list(model.classes_).index(1)
    except (AttributeError, ValueError) as error:
        raise ValueError("model must expose an eviction class labelled 1") from error

    frames: set[int] = set()
    last_access: dict[int, int] = {}
    access_counts: dict[int, int] = {}
    loaded_at: dict[int, int] = {}
    recent_window: deque[int] = deque(maxlen=20)
    events: list[bool] = []
    evictions: list[int] = []

    for time_index, page in enumerate(references):
        if page in frames:
            events.append(True)
        else:
            events.append(False)
            if len(frames) == frame_count:
                candidates = sorted(frames)
                feature_rows = [
                    build_candidate_features(
                        candidate,
                        time_index,
                        last_access,
                        access_counts,
                        recent_window,
                        loaded_at,
                    )
                    for candidate in candidates
                ]
                probabilities = model.predict_proba(np.asarray(feature_rows, dtype=float))[
                    :, positive_column
                ]
                victim = choose_learned_victim(
                    candidates,
                    feature_rows,
                    probabilities,
                )
                frames.remove(victim)
                loaded_at.pop(victim)
                evictions.append(victim)

            frames.add(page)
            loaded_at[page] = time_index

        access_counts[page] = access_counts.get(page, 0) + 1
        last_access[page] = time_index
        recent_window.append(page)

    hits = sum(events)
    return SimulationResult(
        algorithm="Learned",
        hits=hits,
        faults=len(events) - hits,
        events=tuple(events),
        evictions=tuple(evictions),
    )
