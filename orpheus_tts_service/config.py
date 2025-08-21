import os

def _csv(name: str, default: str = ""):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]

class Config:
    # Downstream GPU workers (NDJSON endpoints)
    DOWNSTREAM_URLS = _csv("ORPHEUS_DOWNSTREAM_URLS", "http://localhost:8085")
    DOWNSTREAM_API_KEY = os.getenv("ORPHEUS_DOWNSTREAM_API_KEY")  # if workers require auth
    # Gateway auth for *clients* of the gateway
    VALID_API_KEYS = set(_csv("ORPHEUS_CLIENT_API_KEYS", "dev_key"))
    # Concurrency + QoS
    INBOUND_MAX_CONCURRENCY = int(os.getenv("ORPHEUS_INBOUND_MAX_CONCURRENCY", "512"))
    HEDGE_TTFB_MS = int(os.getenv("ORPHEUS_HEDGE_TTFB_MS", "400"))
    # Rate limiting
    RATE_LIMIT_RPM = int(os.getenv("ORPHEUS_RATE_LIMIT_RPM", "600"))  # per API key
    # Metrics heartbeat toward clients (ms)
    STREAM_METRICS_INTERVAL_MS = int(os.getenv("ORPHEUS_STREAM_METRICS_INTERVAL_MS", "250"))
