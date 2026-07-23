"""Offline entrypoint for captioning and embedding one gym-equipment photo into the FAQ
pgvector store (see app.image_loader for the mechanics and why captions, not pixels, are
what gets embedded).

Run:
    python scripts/ingest_equipment_image.py data/images/home-gym-equipment.png
"""

import logging
import sys
from pathlib import Path

from app.image_loader import load_equipment_image

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>", file=sys.stderr)
        sys.exit(1)
    load_equipment_image(Path(sys.argv[1]))
