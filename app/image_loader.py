"""Ingests gym-equipment photos into the same pgvector FAQ store as data/faq.md.

embedding_model() (app/ai_config.py) is text-only in both profiles -- MiniLM locally,
OpenAI text-embeddings in prod -- so there's no direct image-embedding path here. Instead
each image is captioned once by chat_model() (gpt-4o-mini, vision-capable) and the caption
text is what actually gets embedded. This means retrieval quality for an image question is
bounded by how well the caption describes it, not by any visual-similarity search -- a
"do you have a squat rack" question matches because the caption mentions "Rogue squat rack",
not because of pixel similarity to a stored reference image.

Chunked one document per image (no header splitting, unlike faq_loader.py's markdown
sections) since a caption is already a single coherent unit.

Idempotent on the image bytes' content hash, same delete-then-add pattern as
app.faq_loader -- re-running with an unchanged image is a no-op; a changed image at the
same path deletes and re-capions/re-embeds.
"""

import base64
import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.ai_config import chat_model, vector_store
from app.db import pool

log = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this gym equipment photo for a fitness studio's knowledge base. Name the "
    "specific equipment visible (racks, benches, dumbbells, rowers, plyo boxes, etc.), "
    "any visible brands, and the general layout. Write 2-4 plain sentences, no markdown."
)


def _source_id(image_path: Path) -> str:
    """Builds the metadata "source" value used for idempotency and deletion.

    Args:
        image_path: Path to the image file.

    Returns:
        A stable "image://<path>" identifier for this image, distinct from
        app.faq_loader's "google-drive://..." source so the two never collide or get
        deleted by each other's stale-chunk cleanup.
    """
    return f"image://{image_path}"


def _content_hash(image_bytes: bytes) -> str:
    """Hashes raw image bytes for the idempotency check.

    Args:
        image_bytes: The raw image file contents.

    Returns:
        A hex SHA-256 digest of the bytes.
    """
    return hashlib.sha256(image_bytes).hexdigest()


def _already_ingested(content_hash: str) -> bool:
    """Checks whether a chunk for this exact image content hash already exists.

    Args:
        content_hash: The hash returned by _content_hash() for the current image bytes.

    Returns:
        True if an existing embedding row already carries this content_hash.
    """
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'content_hash' = %s",
            (content_hash,),
        ).fetchone()
    return row[0] > 0


def _delete_stale(source: str) -> None:
    """Deletes the existing embedding row for this image's source before re-ingesting.

    Args:
        source: The source_id identifying which image's chunk to delete.
    """
    with pool.connection() as conn:
        conn.execute(
            "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source' = %s",
            (source,),
        )
        conn.commit()


def _caption(image_bytes: bytes, mime_type: str) -> str:
    """Generates a text caption for an image via the vision-capable chat model.

    Args:
        image_bytes: The raw image file contents.
        mime_type: The image's MIME type (e.g. "image/png"), used to build the data URI.

    Returns:
        The model's plain-text description of the image.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
        ]
    )
    response = chat_model().invoke([message])
    return response.content


def load_equipment_image(image_path: Path, *, section: str = "Gym Equipment") -> None:
    """Captions and (re-)ingests one equipment photo into the pgvector store.

    No-op if the image's content hash already matches what's stored. If the image at
    this path changed (or is new), deletes any prior chunk for this path and ingests a
    freshly captioned one.

    Args:
        image_path: Path to the image file to ingest.
        section: The "section" metadata value stored alongside the chunk -- lets this
            surface in FAQ search results grouped like any other FAQ section (see
            app.tools.faq.search_faq_raw's return shape).
    """
    image_bytes = image_path.read_bytes()
    content_hash = _content_hash(image_bytes)

    if _already_ingested(content_hash):
        log.info("Equipment image %s already ingested (unchanged content); skipping.", image_path)
        return

    source = _source_id(image_path)
    _delete_stale(source)

    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    caption = _caption(image_bytes, mime_type)

    document = Document(
        page_content=caption,
        metadata={"source": source, "content_hash": content_hash, "section": section},
    )
    vector_store().add_documents([document])
    log.info("Ingested equipment image %s as 1 chunk (caption: %r).", image_path, caption[:80])
