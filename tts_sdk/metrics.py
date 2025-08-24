from __future__ import annotations
import time

class CapLimiter:
    def __init__(self, init: int, max_cap: int):
        self._target = init
        self._max_cap = max_cap

    @property
    def target(self): return self._target

    def set_target(self, n: int):
        self._target = max(1, min(self._max_cap, int(n)))
        
class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self._v = None

    def add(self, x: float):
        self._v = x if self._v is None else (self.alpha * x + (1 - self.alpha) * self._v)

    @property
    def value(self): return self._v

class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_s=10):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        return time.time() >= self.open_until

    def record_success(self):
        self.failures = 0

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open_until = time.time() + self.cooldown_s
            self.failures = 0  # reset counter on opening

class AIMDConcurrency:
    """
    Additive Increase / Multiplicative Decrease.
    Increase slowly when healthy, cut fast on tail spikes or errors.
    """
    def __init__(self, init=8, max_cap=128, add=1, mult=0.7):
        self.current = init
        self.max_cap = max_cap
        self.add = add
        self.mult = mult

    def ok_tick(self):
        self.current = min(self.max_cap, self.current + self.add)

    def bad_tick(self):
        self.current = max(1, int(self.current * self.mult))
