"""
Build the evaluation test set.

Cases come from two places:

1. Generated from MongoDB - every patient with an active condition becomes a case.
   Synthea's own prescription for that condition is stored as an independent
   reference. This only works because the drug corpus is built from openFDA labels,
   not from Synthea's REASONDESCRIPTION: the reference and the knowledge base come
   from different sources, so agreeing with the reference means something.

2. Hand-written safety probes in cases/safety_probes.json - patients and conditions
   chosen to exercise a specific failure mode (allergy conflict, duplicate therapy,
   condition not on the problem list).

Run:  uv run python build_testset.py
"""

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
OUTPUT = CASES_DIR / "generated.json"

# Synthea CSVs, for the reference prescriptions
SYNTHEA_CSV = (
    Path(__file__).resolve().parents[1]
    / "services" / "data" / "raw" / "synthea_csv" / "csv"
)


def load_synthea_reference() -> dict[tuple[str, str], list[str]]:
    """Map (patient_id, condition) -> drugs Synthea actually prescribed for it.

    This is the independent reference. It is NOT in the retrieval corpus, so a
    recommendation matching it is real agreement rather than an echo.
    """
    meds = pd.read_csv(
        SYNTHEA_CSV / "medications.csv",
        usecols=["PATIENT", "DESCRIPTION", "REASONDESCRIPTION"],
        low_memory=False,
    ).dropna(subset=["REASONDESCRIPTION"])

    reference: dict[tuple[str, str], list[str]] = {}
    for (patient, reason), group in meds.groupby(["PATIENT", "REASONDESCRIPTION"]):
        key = (patient, str(reason).lower().strip())
        reference[key] = sorted(set(group["DESCRIPTION"]))
    return reference


async def build() -> None:
    CASES_DIR.mkdir(exist_ok=True)
    reference = load_synthea_reference()
    logger.info("Loaded %d reference prescription groups.", len(reference))

    client = AsyncIOMotorClient(settings.MONGO_URI)
    try:
        collection = client[settings.MONGO_DB_NAME][settings.PATIENT_COLLECTION]
        cases = []

        async for patient in collection.find({}, {"_id": 0}):
            for condition in patient.get("active_conditions", []):
                key = (patient["patient_id"], condition.lower().strip())
                cases.append(
                    {
                        "case_id": f"gen_{len(cases):04d}",
                        "patient_id": patient["patient_id"],
                        "condition": condition,
                        "drug_allergies": patient.get("drug_allergies", []),
                        "current_medications": patient.get("current_medications", []),
                        # Empty when Synthea never prescribed for this condition.
                        # Retrieval metrics skip those; safety metrics still apply.
                        "reference_drugs": reference.get(key, []),
                        "source": "generated",
                    }
                )
    finally:
        client.close()

    with_reference = sum(1 for c in cases if c["reference_drugs"])
    with_allergy = sum(1 for c in cases if c["drug_allergies"])

    OUTPUT.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d cases (%d with a reference prescription, %d with a drug allergy) to %s",
        len(cases), with_reference, with_allergy, OUTPUT,
    )

    if with_allergy == 0:
        logger.warning(
            "No case has a drug allergy - the allergy screen will never be exercised. "
            "Check that ingest_data.py injected synthetic allergies."
        )


if __name__ == "__main__":
    asyncio.run(build())
