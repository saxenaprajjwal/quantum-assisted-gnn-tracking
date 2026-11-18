"""Read members of a remote zip file (e.g. TrackML train_1.zip on Zenodo) with HTTP range requests.

zipfile only needs seek() and read(); this file object fetches 1 MB blocks on demand, so a few events
(~10 MB each for hits + truth + particles) can be read without downloading the 16 GB archive.  Used for
development on the PC and as a fallback on lxplus; the production path is the MD5-verified full download.
"""
from __future__ import annotations

import io
import time
import urllib.request
from collections import OrderedDict
from typing import Optional

ZENODO_TRACKML = "https://zenodo.org/api/records/4730167/files/{name}/content"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, cache_blocks: int = 256, timeout: float = 120.0,
                 retries: int = 4) -> None:
        super().__init__()
        self.block = int(block)
        self.cache_blocks = int(cache_blocks)
        self.timeout = timeout
        self.retries = retries
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self._pos = 0
        self.n_requests = 0
        self.bytes_fetched = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "hsf2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.url = r.geturl()                  # follow the Zenodo redirect once
            cr = r.headers.get("Content-Range")
            if r.status != 206 or not cr or "/" not in cr:
                raise IOError(f"server does not support range requests for {url} (status {r.status})")
            self.size = int(cr.split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = int(offset)
        elif whence == io.SEEK_CUR:
            self._pos += int(offset)
        elif whence == io.SEEK_END:
            self._pos = self.size + int(offset)
        else:
            raise ValueError("bad whence")
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def _block(self, k: int) -> bytes:
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        a = k * self.block
        b = min(a + self.block, self.size) - 1
        err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "hsf2"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                if len(data) != b - a + 1:
                    raise IOError(f"short read {len(data)} of {b - a + 1} bytes")
                break
            except Exception as exc:
                err = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            raise IOError(f"range request {a}-{b} failed: {err}")
        self.n_requests += 1
        self.bytes_fetched += len(data)
        self._cache[k] = data
        if len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return data

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self._pos)
        if n <= 0:
            return 0
        out = memoryview(buf)
        done = 0
        while done < n:
            k, off = divmod(self._pos + done, self.block)
            data = self._block(k)
            take = min(len(data) - off, n - done)
            out[done:done + take] = data[off:off + take]
            done += take
        self._pos += n
        return n


def open_remote(name: str = "train_1.zip", url: Optional[str] = None) -> io.BufferedReader:
    """Buffered, seekable file object for a TrackML zip on Zenodo (pass it to zipfile.ZipFile)."""
    return io.BufferedReader(HTTPRangeFile(url or ZENODO_TRACKML.format(name=name)), buffer_size=1 << 16)
