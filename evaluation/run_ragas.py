"""
RAGAS evaluation - faithfulness, answer relevancy, context precision/recall.

Read the caveat in README.md before reporting these numbers. RAGAS uses an LLM as
judge, and the only judge available locally is a 3B model. A weak judge produces
noisy scores, so treat these as directional, not authoritative - and always report
which judge produced them.

run_safety.py is the more trustworthy signal for this project: it is deterministic
and needs no judge at all.

Run:  uv run python run_ragas.py --limit 20
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CASES = Path(__file__).parent / "cases" / "generated.json"
RESULTS_DIR = Path(__file__).parent / "results"


async def collect(limit: int) -> list[dict]:
    """Run the pipeline and assemble RAGAS's expected fields."""
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    cases.sort(key=lambda c: (not c["reference_drugs"], c["case_id"]))
    cases = cases[:limit]

    rows = []
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        for index, case in enumerate(cases, 1):
            logger.info("[%d/%d] %s", index, len(cases), case["case_id"])
            try:
                response = await client.post(
                    f"{settings.ORCHESTRATOR_URL}/api/v1/recommend",
                    json={
                        "patient_id": case["patient_id"],
                        "disease_context": case["condition"],
                    },
                )
                if response.status_code != 200:
                    logger.warning("  HTTP %d", response.status_code)
                    continue

                payload = response.json()

                # Contexts = the retrieved drug documents. RAGAS scores the answer
                # against these, so they must be what the model actually saw.
                contexts = []
                for step in payload.get("audit_trail", []):
                    if step["tool"] != "query_drug_database":
                        continue
                    try:
                        for candidate in json.loads(step["result"]).get("candidates", []):
                            contexts.append(
                                f"{candidate.get('generic_name', '')}. "
                                f"Indications: {candidate.get('indications', '')} "
                                f"Contraindications: {candidate.get('contraindications', '')}"
                            )
                    except (json.JSONDecodeError, AttributeError):
                        pass

                if not contexts:
                    logger.warning("  no contexts retrieved - skipping")
                    continue

                rows.append({
                    "user_input": (
                        f"What should be prescribed for {case['condition']} "
                        f"in this patient?"
                    ),
                    "retrieved_contexts": contexts,
                    "response": payload.get("recommendation", ""),
                    "reference": (
                        ", ".join(case["reference_drugs"])
                        if case["reference_drugs"]
                        else payload.get("recommendation", "")
                    ),
                })
            except Exception as exc:
                logger.warning("  %s", exc)

    return rows


def score(rows: list[dict]) -> None:
    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge = ChatOllama(
        model=settings.JUDGE_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        num_ctx=8192,
    )
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", base_url=settings.OLLAMA_BASE_URL
    )

    logger.info("Scoring %d rows with judge=%s ...", len(rows), settings.JUDGE_MODEL)
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"ragas_{stamp}.json"
    scores = {k: (round(v, 3) if isinstance(v, float) else v)
              for k, v in dict(result).items()}
    path.write_text(
        json.dumps(
            {"judge_model": settings.JUDGE_MODEL, "rows": len(rows), "scores": scores},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)
    print(f"  judge   {settings.JUDGE_MODEL}")
    print(f"  rows    {len(rows)}")
    for metric, value in scores.items():
        print(f"  {metric:<22} {value}")
    print(f"\n  written to {path}")
    print("=" * 60)
    print("\nReport the judge model alongside these numbers. A 3B judge is weak;")
    print("the scores are directional, not authoritative.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    collected = asyncio.run(collect(args.limit))
    if not collected:
        logger.error("No rows collected - is the orchestrator running?")
    else:
        score(collected)
