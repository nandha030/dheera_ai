import importlib_metadata

try:
    version = importlib_metadata.version("dheera_ai")
except Exception:
    version = "unknown"
