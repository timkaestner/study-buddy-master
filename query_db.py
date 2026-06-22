import chromadb
from fill_db import JinaEmbeddingFunction

DB_PATH = "./chroma_po_db"
COLLECTION_NAME = "pruefungsordnung"
#EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"
embedding_fn = JinaEmbeddingFunction()
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

query = "Ist das Bachelorstudium berufsqualifizierend?"

results = collection.query(
    query_texts=[query],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

for i, doc in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print("=" * 80)
    print("Citation:", metadata["subsection_citation"])
    print("Title:", metadata["title"])
    print("Distance:", distance)
    print()
    print(doc)