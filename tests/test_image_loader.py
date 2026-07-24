"""Mocked unit tests for app.image_loader -- mocks chat_model, vector_store, and the
psycopg pool directly (no live DB, no OpenAI call)."""

from unittest.mock import MagicMock, patch

from app.image_loader import load_equipment_image

FAKE_IMAGE_BYTES = b"not-a-real-png-but-thats-fine-for-this-test"


def _mock_pool(already_ingested: bool):
    """Builds a pool mock whose connection().execute(...).fetchone() reports the given
    already-ingested state for _already_ingested's count(*) query."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1 if already_ingested else 0,)
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    return pool, conn


# Full happy path: caption the image via chat_model, embed it, and delete any stale rows
# for this same source first (the content-hash idempotency pattern shared with
# app.faq_loader) -- checks the actual document content/metadata that gets stored, not
# just that add_documents was called.
def test_load_equipment_image_captions_and_embeds_new_image(tmp_path):
    image_path = tmp_path / "gym.png"
    image_path.write_bytes(FAKE_IMAGE_BYTES)

    pool, conn = _mock_pool(already_ingested=False)
    chat_model = MagicMock()
    chat_model.return_value.invoke.return_value.content = "A rack of dumbbells and a Rogue squat rack."
    vector_store = MagicMock()

    with (
        patch("app.image_loader.pool", pool),
        patch("app.image_loader.chat_model", chat_model),
        patch("app.image_loader.vector_store", vector_store),
    ):
        load_equipment_image(image_path)

    vector_store.return_value.add_documents.assert_called_once()
    [document] = vector_store.return_value.add_documents.call_args.args[0]
    assert document.page_content == "A rack of dumbbells and a Rogue squat rack."
    assert document.metadata["section"] == "Gym Equipment"
    assert document.metadata["source"] == f"image://{image_path}"
    conn.execute.assert_any_call(
        "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source' = %s",
        (f"image://{image_path}",),
    )


# Idempotency check: re-running against an unchanged image must be a no-op -- no wasted
# chat_model captioning call, no re-embedding.
def test_load_equipment_image_skips_when_content_hash_already_ingested(tmp_path):
    image_path = tmp_path / "gym.png"
    image_path.write_bytes(FAKE_IMAGE_BYTES)

    pool, _conn = _mock_pool(already_ingested=True)
    chat_model = MagicMock()
    vector_store = MagicMock()

    with (
        patch("app.image_loader.pool", pool),
        patch("app.image_loader.chat_model", chat_model),
        patch("app.image_loader.vector_store", vector_store),
    ):
        load_equipment_image(image_path)

    chat_model.return_value.invoke.assert_not_called()
    vector_store.return_value.add_documents.assert_not_called()


# section defaults to "Gym Equipment" (see the first test) but must be overridable per
# call, since a real ingest batch spans multiple equipment categories.
def test_load_equipment_image_uses_custom_section(tmp_path):
    image_path = tmp_path / "gym.jpg"
    image_path.write_bytes(FAKE_IMAGE_BYTES)

    pool, _conn = _mock_pool(already_ingested=False)
    chat_model = MagicMock()
    chat_model.return_value.invoke.return_value.content = "A rowing machine and a plyo box."
    vector_store = MagicMock()

    with (
        patch("app.image_loader.pool", pool),
        patch("app.image_loader.chat_model", chat_model),
        patch("app.image_loader.vector_store", vector_store),
    ):
        load_equipment_image(image_path, section="Cardio Equipment")

    [document] = vector_store.return_value.add_documents.call_args.args[0]
    assert document.metadata["section"] == "Cardio Equipment"


def test_load_equipment_image_batch_produces_distinct_non_colliding_sources(tmp_path):
    """Mirrors ingesting a batch of gym photos (treadmill, rower, dumbbells, ...): each
    image must get its own source id, and deleting stale rows for one must never touch
    another's chunk -- otherwise a re-run of image N would silently wipe image N-1."""
    filenames = ["treadmill.jpg", "rower.jpg", "dumbbells.jpg"]
    captions = [
        "A treadmill for cardio.",
        "A rowing machine.",
        "A rack of dumbbells.",
    ]

    pool, conn = _mock_pool(already_ingested=False)
    chat_model = MagicMock()
    vector_store = MagicMock()

    with (
        patch("app.image_loader.pool", pool),
        patch("app.image_loader.chat_model", chat_model),
        patch("app.image_loader.vector_store", vector_store),
    ):
        for filename, caption in zip(filenames, captions):
            image_path = tmp_path / filename
            image_path.write_bytes(FAKE_IMAGE_BYTES + filename.encode())
            chat_model.return_value.invoke.return_value.content = caption
            load_equipment_image(image_path)

    assert vector_store.return_value.add_documents.call_count == len(filenames)

    embedded_sources = [
        call.args[0][0].metadata["source"] for call in vector_store.return_value.add_documents.call_args_list
    ]
    assert len(set(embedded_sources)) == len(filenames), "each image must embed under its own distinct source"

    deleted_sources = [call.args[1][0] for call in conn.execute.call_args_list if "DELETE" in call.args[0]]
    assert deleted_sources == embedded_sources, "each ingest must only delete stale rows for its own image"
