from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import Card
from .visual import DEFAULT_MODEL, PREPROCESS_VERSION

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "data" / "vector_db"
DEFAULT_TABLE = "card_embeddings"


class CardVectorStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, table_name: str = DEFAULT_TABLE) -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name

    def exists(self) -> bool:
        if not self.db_path.exists():
            return False
        import lancedb

        database = lancedb.connect(str(self.db_path))
        return self.table_name in database.table_names()

    def create(self, rows: list[dict[str, Any]], mode: str = "overwrite") -> int:
        import lancedb

        self.db_path.mkdir(parents=True, exist_ok=True)
        database = lancedb.connect(str(self.db_path))
        database.create_table(self.table_name, data=rows, mode=mode)
        return len(rows)

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        import lancedb

        database = lancedb.connect(str(self.db_path))
        table = database.open_table(self.table_name)
        return table.search(vector.tolist()).metric("cosine").limit(top_k).to_list()

    @staticmethod
    def card_to_row(
        card: Card,
        vector: np.ndarray,
        embedding_model: str = DEFAULT_MODEL,
        preprocess_version: str = PREPROCESS_VERSION,
    ) -> dict[str, Any]:
        return {
            "card_id": card.card_id,
            "name": card.name,
            "set_code": card.set_code,
            "set_name": card.set_name,
            "printed_number": card.printed_number,
            "collector_number": card.collector_number or card.normalized_collector_number,
            "image_path": card.local_image_path or "",
            "source_url": card.image_url or "",
            "embedding_model": embedding_model,
            "embedding_dim": int(vector.shape[0]),
            "preprocess_version": preprocess_version,
            "vector": vector.astype("float32").tolist(),
        }

    @staticmethod
    def row_to_card(row: dict[str, Any]) -> Card:
        return Card(
            card_id=str(row.get("card_id") or ""),
            name=str(row.get("name") or ""),
            set_code=str(row.get("set_code") or ""),
            set_name=str(row.get("set_name") or ""),
            printed_number=str(row.get("printed_number") or ""),
            collector_number=str(row.get("collector_number") or ""),
            local_image_path=str(row.get("image_path") or "") or None,
            image_url=str(row.get("source_url") or "") or None,
        )
