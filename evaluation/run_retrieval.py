"""
Retrieval evaluation - recall@k against ChromaDB directly.

Measures the retrieval layer in isolation, with no LLM in the loop. If retrieval
never surfaces the right drug, no amount of prompt engineering downstream will
fix the recommendation - so measure this before blaming the model.

Ground truth is Synthea's own prescription for the condition. Because the corpus
is built from openFDA labels rather than Synthea, a hit is real agreement between
two independent sources.

Run:  uv run python run_retrieval.py
"""

import json
import logging
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CASES = Path(__file__).parent / "cases" / "generated.json"
K_VALUES = (1, 3, 5, 10)
QUALIFIERS = ("(disorder)", "(finding)", "(situation)", "(procedure)")


def normalise(text: str) -> str:
    for qualifier in QUALIFIERS:
        text = text.replace(qualifier, "")
    return " ".join(text.split())


def ingredient(label: str) -> str:
    """Strip dose and form from a Synthea label to get the ingredient."""
    noise = {
        "ml", "mg", "hr", "actuat", "oral", "tablet", "capsule", "injection",
        "solution", "suspension", "film", "coated", "extended", "release",
        "sodium", "hydrochloride", "sulfate", "prefilled", "syringe", "mcg",
    }
    for token in re.findall(r"[A-Za-z]+", label):
        lowered = token.lower()
        if len(lowered) > 3 and lowered not in noise:
            return lowered
    return ""


def main() -> None:
    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8")) if c["reference_drugs"]]
    logger.info("%d cases have a reference prescription.", len(cases))

    if not cases:
        logger.error("No cases with references - nothing to measure.")
        return

    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    collection = client.get_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBED_MODEL
        ),
    )

    max_k = max(K_VALUES)
    hits = {k: 0 for k in K_VALUES}
    ranks = []

    # One query per distinct condition - patients sharing a condition share results
    by_condition: dict[str, list[dict]] = {}
    for case in cases:
        by_condition.setdefault(case["condition"], []).append(case)

    for condition, group in by_condition.items():
        results = collection.query(
            query_texts=[normalise(condition)], n_results=max_k
        )
        retrieved = [
            (m.get("generic_name") or "").lower()
            for m in (results.get("metadatas") or [[]])[0]
        ]

        for case in group:
            wanted = {ingredient(d) for d in case["reference_drugs"]}
            wanted.discard("")

            rank = None
            for position, name in enumerate(retrieved, 1):
                if any(w and (w in name or name in w) for w in wanted):
                    rank = position
                    break

            if rank:
                ranks.append(rank)
                for k in K_VALUES:
                    if rank <= k:
                        hits[k] += 1

    total = len(cases)
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"  cases with reference   {total}")
    print(f"  distinct conditions    {len(by_condition)}")
    for k in K_VALUES:
        print(f"  recall@{k:<15} {hits[k] / total:.1%}  ({hits[k]}/{total})")
    if ranks:
        mrr = sum(1 / r for r in ranks) / total
        print(f"  MRR                    {mrr:.3f}")
    print("=" * 60)
    print(
        "\nA low recall@5 means the corpus lacks the drug or the embedding does not\n"
        "connect the condition wording to FDA indication prose. Fix retrieval before\n"
        "tuning prompts."
    )


if __name__ == "__main__":
    main()
