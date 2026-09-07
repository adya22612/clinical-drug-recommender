"""MongoDB accessor.

The connection is created lazily and reused. Motor clients are safe to share
across coroutines; creating one per call opens a new pool on every tool
invocation.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from src.config import settings


class MongoClient:
    _client: AsyncIOMotorClient | None = None

    @classmethod
    def get_collection(cls, collection_name: str) -> AsyncIOMotorCollection:
        """Return a collection handle from the configured database."""
        if cls._client is None:
            cls._client = AsyncIOMotorClient(settings.MONGO_URI)
        return cls._client[settings.MONGO_DB_NAME][collection_name]

    @classmethod
    def close(cls) -> None:
        if cls._client is not None:
            cls._client.close()
            cls._client = None
