import json
import time
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


PARA_ROOT = Path("pruefungsordnung/allgemein")
DB_PATH = "./chroma_po_db"
COLLECTION_NAME = "pruefungsordnung"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"


class JinaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("Loading embedding model...")
        start = time.time()

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            model_kwargs={"default_task": "retrieval"},
        )

        print(f"Model loaded in {time.time() - start:.2f}s")

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)

        print(f"Embedding {len(texts)} document(s)...")
        start = time.time()

        embeddings = self.model.encode(
            texts,
            task="retrieval",
            normalize_embeddings=True,
        ).tolist()

        print(f"Embedding done in {time.time() - start:.2f}s")
        return embeddings


def load_paragraph_files(root: Path):
    json_files = sorted(root.glob("*.json"))
    paragraphs = []

    for file_path in json_files:
        print(f"Loading {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            paragraphs.extend(data)
        else:
            paragraphs.append(data)

    return paragraphs


def make_document(paragraph: dict, subsection: dict) -> str:
    return (
        f"{subsection.get('citation', paragraph.get('citation', ''))} – "
        f"{paragraph.get('title', '')}\n"
        f"Paragraph summary: {paragraph.get('summary', '')}\n"
        f"Subsection summary: {subsection.get('summary', '')}\n"
        f"Original text: {subsection.get('text', '')}"
    )


def make_metadata(paragraph: dict, subsection: dict) -> dict:
    return {
        "paragraph_id": paragraph.get("id"),
        "subsection_id": subsection.get("id"),
        "section": paragraph.get("section"),
        "section_title": paragraph.get("section_title"),
        "paragraph": paragraph.get("paragraph"),
        "paragraph_citation": paragraph.get("citation"),
        "subsection_citation": subsection.get("citation"),
        "title": paragraph.get("title"),
        "paragraph_summary": paragraph.get("summary"),
        "subsection_summary": subsection.get("summary"),
        # wichtig für Quellenangaben / Filter
        "source": paragraph.get("source"),
        "page_from": paragraph.get("page_from"),
        "page_to": paragraph.get("page_to"),
        "source_type": "subsection",
    }


def clean_metadata(md: dict, doc_id: str) -> dict:
    clean = {}

    for key, value in md.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            print(
                f"Metadata conversion needed for {doc_id}: "
                f"{key}={type(value).__name__}"
            )
            clean[key] = json.dumps(value, ensure_ascii=False)

    return clean


def fill_database_debug():
    embedding_fn = JinaEmbeddingFunction()

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )

    paragraphs = load_paragraph_files(PARA_ROOT)

    inserted = 0

    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        paragraph_id = paragraph.get("id")
        paragraph_citation = paragraph.get("citation", "")
        subsections = paragraph.get("subsections", [])

        print(
            f"\nParagraph {paragraph_index}/{len(paragraphs)}: "
            f"{paragraph_id} {paragraph_citation} "
            f"with {len(subsections)} subsection(s)"
        )

        for subsection_index, subsection in enumerate(subsections, start=1):
            subsection_id = subsection.get("id")

            if not subsection_id:
                raise ValueError(
                    f"Missing subsection id in paragraph {paragraph_id}"
                )

            citation = subsection.get("citation", paragraph_citation)
            document = make_document(paragraph, subsection)
            metadata = clean_metadata(
                make_metadata(paragraph, subsection),
                subsection_id,
            )

            print("\n----------------------------------------")
            print(f"Trying subsection {subsection_index}/{len(subsections)}")
            print(f"ID: {subsection_id}")
            print(f"Citation: {citation}")
            print(f"Document length: {len(document)} characters")
            print(f"Metadata keys: {list(metadata.keys())}")

            start = time.time()

            try:
                collection.upsert(
                    ids=[subsection_id],
                    documents=[document],
                    metadatas=[metadata],
                )

                inserted += 1
                print(
                    f"OK: {subsection_id} "
                    f"in {time.time() - start:.2f}s"
                )

            except Exception as e:
                print("\nFAILED")
                print(f"ID: {subsection_id}")
                print(f"Citation: {citation}")
                print(f"Paragraph ID: {paragraph_id}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error: {repr(e)}")
                raise

    print(
        f"\nDone. Inserted/updated {inserted} subsection chunks "
        f"from {len(paragraphs)} paragraph files."
    )


if __name__ == "__main__":
    fill_database_debug()