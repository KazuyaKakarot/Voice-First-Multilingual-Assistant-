"""Tiny statistics helpers (no numpy needed)."""


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile, p in 0-100. None for an empty list."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
