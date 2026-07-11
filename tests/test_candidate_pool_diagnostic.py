from benchmarks.diagnose_candidate_pool import _p95


def test_p95_sorts_samples_before_selecting_percentile():
    values = list(range(1, 101))
    scrambled = values[50:] + values[:50]

    assert _p95(scrambled) == 95


def test_p95_handles_empty_and_singleton_samples():
    assert _p95([]) == 0
    assert _p95([7.5]) == 7.5
