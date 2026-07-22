"""Loads the studio's FAQ markdown and ingests it into the pgvector store for RAG.

Source is profile-gated: prod reads the live object from S3 (s3://<FAQ_S3_BUCKET>/<FAQ_S3_KEY>),
which is the single source of truth; local reads data/faq.md instead (avoids requiring AWS
creds for a laptop dev loop), gitignored rather than checked in so there's no second copy that
can drift from S3 -- pull/place your own copy there before running locally.
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

from app.ai_config import vector_store
from app.config import settings
from app.db import pool

log = logging.getLogger(__name__)

SOURCE_ID = f"s3://{settings.faq_s3_bucket}/{settings.faq_s3_key}"

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2")]


def _load_markdown_text() -> str:
    """Reads the raw FAQ markdown from the profile-appropriate source.

    Returns:
        The FAQ document's full text: fetched from S3 in prod, read from
        settings.faq_local_path in local.
    """
    if settings.app_profile == "prod":
        import boto3

        client = boto3.client("s3", region_name=settings.aws_region)
        obj = client.get_object(Bucket=settings.faq_s3_bucket, Key=settings.faq_s3_key)
        return obj["Body"].read().decode("utf-8")
    return settings.faq_local_path.read_text(encoding="utf-8")


def _content_hash(text: str) -> str:
    """Hashes FAQ text for the idempotency check.

    Args:
        text: The full FAQ document text.

    Returns:
        A hex SHA-256 digest of the UTF-8 encoded text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _already_ingested(content_hash: str) -> bool:
    """Checks whether chunks for this exact FAQ content hash already exist.

    Args:
        content_hash: The hash returned by _content_hash() for the current FAQ text.

    Returns:
        True if at least one existing embedding row carries this content_hash.
    """
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'content_hash' = %s",
            (content_hash,),
        ).fetchone()
    return row[0] > 0


def _delete_stale(source: str) -> None:
    """Deletes all embedding rows for a given source before re-ingesting changed content.

    Args:
        source: The SOURCE_ID identifying which document's chunks to delete.
    """
    with pool.connection() as conn:
        conn.execute(
            "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source' = %s",
            (source,),
        )
        conn.commit()


def load_faq_knowledge() -> None:
    """Loads, chunks, and (re-)ingests the FAQ document into the pgvector store.

    No-op if the current FAQ content's hash already matches what's stored. If the
    content changed, deletes the old chunks for SOURCE_ID and re-ingests fresh ones,
    split by markdown header so each chunk is one coherent FAQ section. Called once at
    app startup (app.main's lifespan).
    """
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
