from pathlib import Path
from typing import Optional, Type, Any
from pydantic import BaseModel, Field, Extra
from langchain_core.tools import BaseTool
import sqlite3
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class QueryToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Vollständige, gültige SQLite-SELECT-Query",
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------
class QueryTool(BaseTool):
    name: str = "modulhandbuch_abfragen"
    description: str = (
        "Gibt passende Informationen aus dem Modulhandbuch "
        "Technisches Informationsdesign für eine filterbasierte SQL-Abfrage zurück."
    )
    args_schema: Type[BaseModel] = QueryToolInput

    class Config:
        extra = Extra.allow

    def __init__(self):
        super().__init__()

    def _run(self, query: str, **kwargs) -> Any:
        cleaned_query = query.strip()
        logger.info("Executing SQL: %s", cleaned_query)

        try:
            # Use a context manager so the connection is reliably closed
            with sqlite3.connect("study_buddy.db") as conn:
                conn.row_factory = sqlite3.Row  # dict-like row access
                cursor = conn.cursor()
                cursor.execute(cleaned_query)
                rows = cursor.fetchall()

                if not rows:
                    return "Die Abfrage lieferte keine Ergebnisse."

                # Format as a readable Markdown table for the LLM
                headers = rows[0].keys()
                lines = [
                    " | ".join(headers),
                    " | ".join(["---"] * len(headers)),
                ]
                for row in rows:
                    lines.append(" | ".join(str(cell) for cell in row))

                return "\n".join(lines)

        except Exception as e:
            logger.exception("Datenbankabfrage fehlgeschlagen")
            return f"Fehler bei der Datenbankabfrage: {str(e)}"
