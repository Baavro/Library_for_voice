# config_loader.py
from __future__ import annotations
import os, pathlib, json
from typing import Dict, Any, Callable

# Simple KEY=VALUE loader (no external deps).
# Supports comments (# ...), quoted strings, and type casting via schema.

def _str_to_bool(s: str) -> bool:
    return s.strip().lower() in ("1","true","yes","y","on")

_CASTERS: Dict[str, Callable[[str], Any]] = {
    "str": str,
    "int": lambda v: int(v.strip()),
    "float": lambda v: float(v.strip()),
    "bool": _str_to_bool,
}

def _cast(val: str, typ: str) -> Any:
    if val is None:
        return None
    if typ in _CASTERS:
        return _CASTERS[typ](val)
    return val  # default: string

def load_env_file(path: str) -> Dict[str, str]:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    out: Dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out

def merge_defaults(values: Dict[str, str], schema_defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(schema_defaults)
    for k, v in values.items():
        merged[k] = v
    return merged

def cast_by_schema(values: Dict[str, Any], schema_types: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in values.items():
        typ = schema_types.get(k, "str")
        if isinstance(v, str):
            out[k] = _cast(v, typ)
        else:
            out[k] = v
    return out

def load_config(path: str, schema_defaults: Dict[str, Any], schema_types: Dict[str, str]) -> Dict[str, Any]:
    # Prefer explicit file path; if not provided, allow ENV var CONFIG_PATH as last resort
    path = path or os.getenv("CONFIG_PATH", "")
    if not path:
        raise FileNotFoundError("CONFIG_PATH not provided and no explicit path was given.")
    vals = load_env_file(path)
    merged = merge_defaults(vals, schema_defaults)
    return cast_by_schema(merged, schema_types)
