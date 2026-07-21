"""Loads the studio's FAQ markdown and ingests it into the pgvector store for RAG.

Source is profile-gated: prod reads the live object from S3 (s3://<FAQ_S3_BUCKET>/<FAQ_S3_KEY>),
local reads a checked-in copy at data/faq.md (avoids requiring AWS creds for a laptop dev loop).
Chunked by markdown header (## sections — one per topic: General Membership, Yoga, Pilates, ...)
so retrieval returns whole, coherent FAQ sections rather than arbitrary character windows.

Idempotent on content hash: re-running with unchanged FAQ content is a no-op; if the FAQ text
changes, the old chunks for that source are deleted and re-ingested (delete-then-add — pgvector
has no concept of "this row is stale", so there's no cheaper diff to do without hand-tracking
per-chunk hashes, which isn't worth it for a single-document corpus)."""

import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app import config
from app.ai_config import vector_store
from app.db import pool

log = logging.getLogger(__name__)

SOURCE_ID = f"s3://{config.FAQ_S3_BUCKET}/{config.FAQ_S3_KEY}"

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2")]


def _load_markdown_text() -> str:
    if config.PROFILE == "prod":
        import boto3

        client = boto3.client("s3", region_name=config.AWS_REGION)
        obj = client.get_object(Bucket=config.FAQ_S3_BUCKET, Key=config.FAQ_S3_KEY)
        return obj["Body"].read().decode("utf-8")
    return config.FAQ_LOCAL_PATH.read_text(encoding="utf-8")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _already_ingested(content_hash: str) -> bool:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'content_hash' = %s",
            (content_hash,),
        ).fetchone()
    return row[0] > 0


def _delete_stale(source: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source' = %s",
            (source,),
        )
        conn.commit()


def load_faq_knowledge() -> None:
    text = _load_markdown_text()
    content_hash = _content_hash(text)

    if _already_ingested(content_hash):
        log.info("FAQ knowledge already ingested (unchanged content); skipping.")
        return

    _delete_stale(SOURCE_ID)

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks = splitter.split_text(text)

    documents = [
        Document(
            page_content=chunk.page_content,
            metadata={
                "source": SOURCE_ID,
                "content_hash": content_hash,
                "section": chunk.metadata.get("h2") or chunk.metadata.get("h1"),
            },
        )
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not documents:
        log.warning("FAQ document produced no chunks; nothing ingested.")
        return

    vector_store().add_documents(documents)
    log.info("Ingested %d FAQ chunks from %s.", len(documents), SOURCE_ID)
