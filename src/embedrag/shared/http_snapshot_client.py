"""HTTP-based snapshot client for downloading snapshots from a static file server or CDN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 600


class HttpSnapshotClient:
    """Downloads snapshots from an HTTP/HTTPS base URL.

    Expects the remote layout to mirror the object-store convention::

        {base_url}/latest.json
        {base_url}/{version}/manifest.json
        {base_url}/{version}/index/text/shard_0.faiss.zst
        ...

    Implements the same ``get_json`` / ``download_file`` interface used by
    ``ObjectStoreClient`` so the syncer can work with either backend.
    """

    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}/{path}"

    def get_json(self, remote_path: str) -> Optional[dict]:
        url = self._url(remote_path)
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                resp = client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            logger.warn("http_get_json_error", url=url)
            return None
        except Exception:
            logger.exception("http_get_json_failed", url=url)
            return None

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        url = self._url(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("http_download", url=url, local=str(local_path))
        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
