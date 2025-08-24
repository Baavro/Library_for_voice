import time
from collections import defaultdict

class TokenBucket:
    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0
        self.tokens = defaultdict(lambda: per_minute)
        self.last = defaultdict(lambda: time.time())

    def allow(self, key: str, cost: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last[key]
        self.last[key] = now
        # refill
        self.tokens[key] = min(self.tokens[key] + elapsed * self.rate, self.rate * 60)
        if self.tokens[key] >= cost:
            self.tokens[key] -= cost
            return True
        return False
