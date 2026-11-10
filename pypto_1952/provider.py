"""Load operator test assets from a gym checkout, independently of the working directory."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys


def load_provider(path=None, gym_root=None):
    if path is None:
        root = Path(gym_root or os.environ.get("PYPTO_GYM_ROOT") or Path(__file__).resolve().parents[2] / "pypto-gym")
        path = root / "ops/pro/experimental/native_5102/tests/provider.py"
    path = Path(path).expanduser().resolve(strict=True)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = "_pypto_1952_provider_" + hashlib.sha256((str(path) + digest).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Provider changed while it was being loaded")
            module.__provider_sha256__ = digest
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return path, sys.modules[name]
