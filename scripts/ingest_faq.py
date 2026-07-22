"""Offline FAQ ingestion entrypoint -- meant to run as a Kubernetes Job/CronJob (or
manually), not from the app's own startup. Re-embeds the FAQ source into pgvector if
its content has changed since the last run; no-op otherwise (see
app.faq_loader.load_faq_knowledge's content-hash idempotency).

Run:
    python scripts/ingest_faq.py
"""

import logging

from app.faq_loader import load_faq_knowledge

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    load_faq_knowledge()
