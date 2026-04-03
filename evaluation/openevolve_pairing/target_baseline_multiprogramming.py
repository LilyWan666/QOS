"""Baseline multiprogramming target wrapper for evaluator-compatible scoring."""

from qos.multiprogrammer.multiprogrammer import Multiprogrammer as _BaselineMultiprogrammer


def get_matching_score(self, q1, q2, backend, weighted: bool = False, weights=[]):
    """Delegate to the baseline Multiprogrammer implementation without modification."""
    return _BaselineMultiprogrammer.get_matching_score(
        self,
        q1,
        q2,
        backend,
        weighted=weighted,
        weights=weights,
    )
