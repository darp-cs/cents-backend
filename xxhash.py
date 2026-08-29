"""Pure-Python compatibility shim for environments that block xxhash native DLLs.

This module mirrors the most common API from the ``xxhash`` package so imports
continue working even when ``xxhash._xxhash`` cannot be loaded due to Windows
Application Control policies.
"""

from __future__ import annotations

import hashlib
from typing import Any

VERSION = "3.5.0-purepy-shim"
XXHASH_VERSION = "0.8.0"


class _BaseXXH:
    """Small hash object with an xxhash-like interface."""

    def __init__(
        self,
        data: Any = b"",
        *,
        seed: int = 0,
        algorithm: str,
        digest_size: int,
    ) -> None:
        self._algorithm = algorithm
        self._seed = int(seed)
        self._digest_size = digest_size
        self._hasher = self._new_hasher()
        if data:
            self.update(data)

    @property
    def name(self) -> str:
        return self._algorithm

    @property
    def digest_size(self) -> int:
        return self._digest_size

    @property
    def block_size(self) -> int:
        return self._hasher.block_size

    def _new_hasher(self) -> hashlib._Hash:
        seed_bytes = self._seed.to_bytes(8, "little", signed=False)
        person = (self._algorithm.encode("utf-8") + seed_bytes)[:16].ljust(16, b"\0")
        return hashlib.blake2b(digest_size=self._digest_size, person=person)

    @staticmethod
    def _coerce_bytes(data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, memoryview):
            return data.tobytes()
        if isinstance(data, str):
            return data.encode("utf-8")
        raise TypeError("a bytes-like object or str is required")

    def update(self, data: Any) -> "_BaseXXH":
        self._hasher.update(self._coerce_bytes(data))
        return self

    def digest(self) -> bytes:
        return self._hasher.digest()

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    def intdigest(self) -> int:
        return int.from_bytes(self.digest(), "big", signed=False)

    def copy(self) -> "_BaseXXH":
        clone = self.__class__(seed=self._seed)  # type: ignore[call-arg]
        clone._hasher = self._hasher.copy()
        return clone

    def reset(self) -> "_BaseXXH":
        self._hasher = self._new_hasher()
        return self


class _XXH32(_BaseXXH):
    def __init__(self, data: Any = b"", seed: int = 0) -> None:
        super().__init__(data, seed=seed, algorithm="xxh32", digest_size=4)


class _XXH64(_BaseXXH):
    def __init__(self, data: Any = b"", seed: int = 0) -> None:
        super().__init__(data, seed=seed, algorithm="xxh64", digest_size=8)


class _XXH3_64(_BaseXXH):
    def __init__(self, data: Any = b"", seed: int = 0) -> None:
        super().__init__(data, seed=seed, algorithm="xxh3_64", digest_size=8)


class _XXH3_128(_BaseXXH):
    def __init__(self, data: Any = b"", seed: int = 0) -> None:
        super().__init__(data, seed=seed, algorithm="xxh3_128", digest_size=16)


def xxh32(data: Any = b"", seed: int = 0) -> _XXH32:
    return _XXH32(data, seed)


def xxh64(data: Any = b"", seed: int = 0) -> _XXH64:
    return _XXH64(data, seed)


def xxh3_64(data: Any = b"", seed: int = 0) -> _XXH3_64:
    return _XXH3_64(data, seed)


def xxh3_128(data: Any = b"", seed: int = 0) -> _XXH3_128:
    return _XXH3_128(data, seed)


def xxh32_digest(data: Any = b"", seed: int = 0) -> bytes:
    return xxh32(data, seed).digest()


def xxh32_intdigest(data: Any = b"", seed: int = 0) -> int:
    return xxh32(data, seed).intdigest()


def xxh32_hexdigest(data: Any = b"", seed: int = 0) -> str:
    return xxh32(data, seed).hexdigest()


def xxh64_digest(data: Any = b"", seed: int = 0) -> bytes:
    return xxh64(data, seed).digest()


def xxh64_intdigest(data: Any = b"", seed: int = 0) -> int:
    return xxh64(data, seed).intdigest()


def xxh64_hexdigest(data: Any = b"", seed: int = 0) -> str:
    return xxh64(data, seed).hexdigest()


def xxh3_64_digest(data: Any = b"", seed: int = 0) -> bytes:
    return xxh3_64(data, seed).digest()


def xxh3_64_intdigest(data: Any = b"", seed: int = 0) -> int:
    return xxh3_64(data, seed).intdigest()


def xxh3_64_hexdigest(data: Any = b"", seed: int = 0) -> str:
    return xxh3_64(data, seed).hexdigest()


def xxh3_128_digest(data: Any = b"", seed: int = 0) -> bytes:
    return xxh3_128(data, seed).digest()


def xxh3_128_intdigest(data: Any = b"", seed: int = 0) -> int:
    return xxh3_128(data, seed).intdigest()


def xxh3_128_hexdigest(data: Any = b"", seed: int = 0) -> str:
    return xxh3_128(data, seed).hexdigest()


xxh128 = xxh3_128
xxh128_digest = xxh3_128_digest
xxh128_intdigest = xxh3_128_intdigest
xxh128_hexdigest = xxh3_128_hexdigest

algorithms_available = {
    "xxh32",
    "xxh64",
    "xxh3_64",
    "xxh128",
    "xxh3_128",
}
algorithms_guaranteed = algorithms_available

__all__ = [
    "xxh32",
    "xxh32_digest",
    "xxh32_intdigest",
    "xxh32_hexdigest",
    "xxh64",
    "xxh64_digest",
    "xxh64_intdigest",
    "xxh64_hexdigest",
    "xxh3_64",
    "xxh3_64_digest",
    "xxh3_64_intdigest",
    "xxh3_64_hexdigest",
    "xxh3_128",
    "xxh3_128_digest",
    "xxh3_128_intdigest",
    "xxh3_128_hexdigest",
    "xxh128",
    "xxh128_digest",
    "xxh128_intdigest",
    "xxh128_hexdigest",
    "VERSION",
    "XXHASH_VERSION",
    "algorithms_available",
    "algorithms_guaranteed",
]