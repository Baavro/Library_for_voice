from fastapi import Header, HTTPException, status
from .config import Config
import hmac

def _contains(valid_set: set[str], key: str) -> bool:
    return any(hmac.compare_digest(k, key) for k in valid_set)

async def require_api_key(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    key = None
    if x_api_key:
        key = x_api_key
    elif authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:]
    if not key or not _contains(Config.VALID_API_KEYS, key):
        raise HTTPException(status_code=401, detail="invalid api key")
    return key
