"""ChromaDB accessor."""

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from src.config import settings


class ChromaClient:
    _client = None
    _embed_fn = None

    @classmethod
    def _embedding_function(cls):
        # Pinned explicitly rather than relying on Chroma's default, so the
        # ingestion script and this service cannot silently diverge.
        if cls._embed_fn is None:
            cls._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBED_MODEL
            )
        return cls._embed_fn

    @classmethod
    def get_collection(cls, collection_name: str) -> Collection:
        """Return an existing collection.

        Deliberately not get_or_create: if ingestion has not run, we want a
        loud failure at startup rather than an empty collection that answers
        every query with nothing.
        """
        if cls._client is None:
            cls._client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
        return cls._client.get_collection(
            name=collection_name,
            embedding_function=cls._embedding_function(),
        )
