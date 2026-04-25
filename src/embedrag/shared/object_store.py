"""Abstraction over S3-compatible object storage (S3, TOS, MinIO)."""

from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from embedrag.config import ObjectStoreConfig
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


class ObjectStoreClient:
    """Thin wrapper around boto3 S3 client for snapshot upload/download."""

    def __init__(self, config: ObjectStoreConfig):
        self._bucket = config.bucket
        self._prefix = config.prefix.rstrip("/")
        kwargs: dict = {
            "aws_access_key_id": config.access_key,
            "aws_secret_access_key": config.secret_key,
            "config": BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=10,
            ),
        }
        if config.endpoint:
            kwargs["endpoint_url"] = config.endpoint
        if config.region:
            kwargs["region_name"] = config.region

        self._client = boto3.client("s3", **kwargs)

    def _key(self, path: str) -> str:
        return f"{self._prefix}/{path}" if self._prefix else path

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        key = self._key(remote_path)
        logger.info("object_store_upload", key=key, local=str(local_path))
        self._client.upload_file(str(local_path), self._bucket, key)

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        key = self._key(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("object_store_download", key=key, local=str(local_path))
        self._client.download_file(self._bucket, key, str(local_path))

    def put_json(self, remote_path: str, data: dict) -> None:
        key = self._key(remote_path)
        body = json.dumps(data, indent=2).encode()
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)

    def get_json(self, remote_path: str) -> dict | None:
        key = self._key(remote_path)
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read())
        except self._client.exceptions.NoSuchKey:
            return None

    def head_object(self, remote_path: str) -> dict | None:
        key = self._key(remote_path)
        try:
            return self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception:
            return None

    def list_prefix(self, prefix: str) -> list[str]:
        key = self._key(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=key):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
