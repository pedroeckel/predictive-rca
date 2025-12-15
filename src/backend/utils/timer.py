from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer(name: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"[{name}] Tempo decorrido: {end - start:.3f}s")
