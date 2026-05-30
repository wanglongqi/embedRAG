"""Abstraction over S3-compatible object storage (S3, TOS, MinIO).

This module provides a unified interface for interacting with various object
storage providers. It is primarily used for uploading snapshots from the
writer node and downloading them on the query node for synchronization.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from embedrag.config import ObjectStoreConfig
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


class ObjectStoreClient:
    """Thin wrapper around boto3 S3 client for snapshot upload/download.

    This client abstracts away provider-specific configurations (like custom
    endpoints for MinIO or ByteDance TOS) while providing a simplified API
    for common file and JSON operations.
    """

    def __init__(self, config: ObjectStoreConfig):
        """Initialize the ObjectStoreClient.

        Args:
            config (ObjectStoreConfig): The configuration object containing
                credentials, bucket name, and provider settings.
        """
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
        """Helper to prepend the configured prefix to a path."""
        return f"{self._prefix}/{path}" if self._prefix else path

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a local file to the object store.

        Args:
            local_path (str | Path): The path to the file on the local filesystem.
            remote_path (str): The destination path (key) in the bucket, relative
                to the configured prefix.
        """
        key = self._key(remote_path)
        logger.info("object_store_upload", key=key, local=str(local_path))
        self._client.upload_file(str(local_path), self._bucket, key)

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from the object store to the local filesystem.

        Args:
            remote_path (str): The source path (key) in the bucket, relative
                to the configured prefix.
            local_path (str | Path): The local destination path. Parent directories
                will be created if they don't exist.
        """
        key = self._key(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("object_store_download", key=key, local=str(local_path))
        self._client.download_file(self._bucket, key, str(local_path))

    def put_json(self, remote_path: str, data: dict) -> None:
        """Serialize a dictionary to JSON and upload it to the object store.

        Args:
            remote_path (str): The destination path (key) in the bucket.
            data (dict): The dictionary to serialize and upload.
        """
        key = self._key(remote_path)
        body = json.dumps(data, indent=2).encode()
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)

    def get_json(self, remote_path: str) -> dict | None:
        """Download a JSON file and deserialize it into a dictionary.

        Args:
            remote_path (str): The source path (key) in the bucket.

        Returns:
            dict | None: The deserialized dictionary, or None if the key
                does not exist.
        """
        key = self._key(remote_path)
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read())
        except self._client.exceptions.NoSuchKey:
            return None

    def head_object(self, remote_path: str) -> dict | None:
        """Retrieve metadata for an object without downloading its content.

        Args:
            remote_path (str): The path (key) in the bucket.

        Returns:
            dict | None: The object metadata, or None if an error occurs
                (e.g., object not found).
        """
        key = self._key(remote_path)
        try:
            return self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception:
            return None

    def list_prefix(self, prefix: str) -> list[str]:
        """List all object keys under a specific prefix.

        Args:
            prefix (str): The prefix to list objects from.

        Returns:
            list[str]: A list of full object keys (including the bucket prefix).
        """
        key = self._key(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=key):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
