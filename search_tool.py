from typing import Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, Extra
from chromadb import Collection


# Anfrage, die Nutzende stellen
class SearchToolInput(BaseModel):
    query: str = Field(
        ...,
        description="nutzerfrage_studium_pruefung",
    )


# Werkzeug (Tool), um die Datenbank abzufragen
class SearchTool(BaseTool):
    name: str = "pruefungsordnung_abfragen"
    description: str = "Ermittle eine Antwort aus der Prüfungsordnung auf Basis der Nutzerfrage."
    args_schema: Type[BaseModel] = SearchToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, collection: Collection):
        """Initialisiere das Tool und lade die Vektordatenbank"""
        super().__init__()
        self.collection = collection

    def _run(self, query: str, **kwargs) -> Any:
        try:
            return self.collection.query(
                query_texts=[query],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )

        except Exception as e:
            print(e)
            return f"Error querying items: {str(e)}"
