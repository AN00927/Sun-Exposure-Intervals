import pytest
import numpy as np
from vitd_optimization import (
    schedule_uniform,
    schedule_frontloaded,
    schedule_backloaded,
    schedule_random,
)


def test_schedule_uniform_correct_count():
    sched = schedule_uniform(10, 84, 1.0)
    assert len(sched) == 10, f"Expected 10 sessions, got {len(sched)}"


def test_schedule_uniform_ordered():
    sched = schedule_uniform(10, 84, 1.0)
    days = [d for d, s in sched]
    assert all(days[i] <= days[i+1] for i in range(len(days)-1))


def test_schedule_uniform_within_bounds():
    sched = schedule_uniform(10, 84, 1.0)
    days = [d for d, s in sched]
    assert min(days) >= 0
    assert max(days) < 84


def test_schedule_frontloaded_first_interval_smaller():
    sched = schedule_frontloaded(10, 84, 1.0, skew=2.0)
    days = [d for d, s in sched]
    first_interval = days[1] - days[0]
    last_interval = days[-1] - days[-2]
    assert last_interval > first_interval, (
        f"Front-loaded: last interval {last_interval} should be > first {first_interval}"
    )


def test_schedule_backloaded_first_interval_larger():
    sched = schedule_backloaded(10, 84, 1.0, skew=2.0)
    days = [d for d, s in sched]
    first_interval = days[1] - days[0]
    last_interval = days[-1] - days[-2]
    assert first_interval > last_interval, (
        f"Back-loaded: first interval {first_interval} should be > last {last_interval}"
    )


def test_schedule_random_generates_correct_count():
    samples = list(schedule_random(10, 84, 1.0, n_samples=5))
    assert len(samples) == 5
    for sched in samples:
        assert len(sched) == 10


def test_schedule_uniform_all_same_sed():
    sched = schedule_uniform(10, 84, 1.0)
    seds = [s for d, s in sched]
    assert all(s == 1.0 for s in seds)
