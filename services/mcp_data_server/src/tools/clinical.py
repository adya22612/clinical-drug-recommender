"""Drug literature retrieval over the openFDA label corpus in ChromaDB."""

import logging
from typing import Any

from src.config import settings
from src.database.chroma import ChromaClient

logger = logging.getLogger(__name__)

# SNOMED qualifiers Synthea appends to condition names. The corpus is FDA
# indication prose, so these parentheticals only hurt semantic similarity.
_QUALIFIERS = ("(disorder)", "(finding)", "(situation)", "(procedure)")


def _normalise(text: str) -> str:
    for qualifier in _QUALIFIERS:
        text = text.replace(qualifier, "")
    return " ".join(text.split())


async def search_drug_literature(
    disease: str, top_k: int | None = None
) -> dict[str, Any]:
    """Retrieve drugs whose FDA-approved indications match a condition.

    Returns a dict, not a list: MCP tool results are easier for a model to
    parse when errors and payload share one envelope.
    """
    try:
        collection = ChromaClient.get_collection(settings.CHROMA_COLLECTION)
        n_results = top_k or settings.DEFAULT_TOP_K
        n_results = max(1, min(n_results, settings.MAX_TOP_K))

        results = collection.query(
            query_texts=[_normalise(disease)],
            n_results=n_results,
        )

        # Guard every index - an empty collection returns empty lists, and
        # results['documents'][0] raises rather than returning nothing.
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        candidates = [
            {
                "generic_name": meta.get("generic_name", ""),
                "brand_name": meta.get("brand_name", ""),
                "rxcui": meta.get("rxcui", ""),
                "indications": meta.get("indications", ""),
                "contraindications": meta.get("contraindications", ""),
                "boxed_warning": meta.get("boxed_warning", ""),
                "similarity": round(1 - dist, 3),
            }
            for meta, dist in zip(metadatas, distances)
        ]

        return {"query": disease, "count": len(candidates), "candidates": candidates}
    except Exception as exc:
        logger.exception("search_drug_literature failed")
        return {"error": f"Drug search failed: {exc}"}
