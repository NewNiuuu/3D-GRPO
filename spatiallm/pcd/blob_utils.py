"""Azure Blob reader for SpatialLM point-cloud datasets.

Environment variables
---------------------
BLOB_CONTAINER_URL   [REQUIRED]
    The Azure Blob *container* URL (no SAS token, no trailing slash).
    Meaning : which storage account + container holds your point clouds.
    Format  : https://<account>.blob.core.windows.net/<container>
    Example : https://myaccount.blob.core.windows.net/pointclouds

BLOB_SAS_TOKEN       [REQUIRED]
    A Shared Access Signature granting at least READ + LIST on the container.
    Meaning : the credential used to download blobs.
    Format  : a query string; the leading "?" is optional (it is stripped).
    Example : sv=2022-11-02&ss=b&srt=co&sp=rl&se=2026-12-31T00:00:00Z&sig=XXXX

BLOB_BASE_PREFIX     [OPTIONAL, default ""]
    A prefix prepended to every logical path to form the blob name.
    Meaning : lets you store files under a sub-"folder" inside the container.
    Final blob name = BLOB_BASE_PREFIX + "/" + <local-path-without-leading-slash>
    Format  : a plain path fragment, no leading/trailing slash needed.
    Example : with prefix "" , local /root/nescene/pcd/x.ply
              -> blob name   root/nescene/pcd/x.ply
              with prefix "datasets", same local path
              -> blob name   datasets/root/nescene/pcd/x.ply

BLOB_MAX_CONCURRENCY [OPTIONAL, default "4"]
    Azure SDK parallel connections used to download ONE blob.
    Meaning : higher can speed up large files; tune to your bandwidth.
    Format  : a positive integer (as a string).
    Example : 8

Quick start
-----------
    export BLOB_CONTAINER_URL="https://myaccount.blob.core.windows.net/pointclouds"
    export BLOB_SAS_TOKEN="sv=2022-11-02&ss=b&srt=co&sp=rl&se=...&sig=..."
    # optional:
    # export BLOB_BASE_PREFIX="datasets"
    # export BLOB_MAX_CONCURRENCY="8"
    cd /root/lnj/SpatialLM && python train.py configs/spatiallm_vqa_train.yaml

Note: BLOB_CACHE_DIR from LlamaFactory is intentionally NOT used here -- point
clouds are parsed in memory (see pcd_loader._load_pcd_from_bytes), nothing is
cached to disk, which is the whole point of the blob migration (free up disk).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path


logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _join_blob_path(*parts: str | None) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def has_blob_config() -> bool:
    return bool(_env("BLOB_CONTAINER_URL") and _env("BLOB_SAS_TOKEN"))


@dataclass(frozen=True)
class BlobMediaConfig:
    container_url: str
    sas_token: str
    base_prefix: str = ""
    max_concurrency: int = 8

    @classmethod
    def from_env(cls) -> "BlobMediaConfig":
        container_url = _env("BLOB_CONTAINER_URL").rstrip("/")
        sas_token = _env("BLOB_SAS_TOKEN").lstrip("?")
        base_prefix = _env("BLOB_BASE_PREFIX").strip("/")
        max_concurrency = int(_env("BLOB_MAX_CONCURRENCY") or "8")

        if not container_url:
            raise RuntimeError("Missing BLOB_CONTAINER_URL.")
        if not sas_token:
            raise RuntimeError("Missing BLOB_SAS_TOKEN.")

        return cls(
            container_url=container_url,
            sas_token=sas_token,
            base_prefix=base_prefix,
            max_concurrency=max_concurrency,
        )


class BlobMediaReader:
    def __init__(self, config: BlobMediaConfig | None = None) -> None:
        self.config = config or BlobMediaConfig.from_env()
        self._container_client = None

    @property
    def container_client(self):
        if self._container_client is None:
            try:
                from azure.storage.blob import ContainerClient
            except ImportError as err:
                raise RuntimeError(
                    "Azure Blob support requires `azure-storage-blob`. "
                    "Install it in the SpatialLM environment first."
                ) from err

            self._container_client = ContainerClient.from_container_url(
                self.container_url_with_sas()
            )

        return self._container_client

    def container_url_with_sas(self) -> str:
        return f"{self.config.container_url}?{self.config.sas_token}"

    def resolve_path(self, path: str | os.PathLike[str]) -> str:
        path_text = os.fspath(path).strip()
        if path_text.startswith(("http://", "https://")):
            raise ValueError("BlobMediaReader expects a logical blob path, not a full URL.")

        return _join_blob_path(self.config.base_prefix, path_text.strip("/"))

    def exists(self, path: str | os.PathLike[str]) -> bool:
        return self.container_client.get_blob_client(self.resolve_path(path)).exists()

    def read_bytes(self, path: str | os.PathLike[str]) -> bytes:
        blob_name = self.resolve_path(path)
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.download_blob(max_concurrency=self.config.max_concurrency).readall()

    def open_binary(self, path: str | os.PathLike[str]) -> BytesIO:
        stream = BytesIO(self.read_bytes(path))
        stream.name = os.fspath(path)
        return stream


@lru_cache(maxsize=1)
def default_blob_media_reader() -> BlobMediaReader:
    return BlobMediaReader()


def should_use_blob(path: object) -> bool:
    if not isinstance(path, (str, os.PathLike)):
        return False

    path_text = os.fspath(path)
    return not path_text.startswith(("http://", "https://")) and not os.path.exists(path_text)


def read_blob_media_bytes(path: str | os.PathLike[str]) -> bytes:
    return default_blob_media_reader().read_bytes(path)


def open_blob_media_stream(path: str | os.PathLike[str]) -> BytesIO:
    return default_blob_media_reader().open_binary(path)
