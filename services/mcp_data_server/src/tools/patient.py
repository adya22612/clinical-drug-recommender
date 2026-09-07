"""Patient-facing tool implementations.

Plain async functions with no MCP dependency, so they can be unit tested
without a running server. src/server.py wraps them as MCP tools.
"""

import logging
from typing import Any

from src.config import settings
from src.database.mongo import MongoClient

logger = logging.getLogger(__name__)


async def _find_patient(patient_id: str) -> dict[str, Any] | None:
    collection = MongoClient.get_collection(settings.PATIENT_COLLECTION)
    return await collection.find_one({"patient_id": patient_id}, {"_id": 0})


async def fetch_patient_data(patient_id: str) -> dict[str, Any]:
    """Return a patient's full record, or an error dict if not found."""
    try:
        document = await _find_patient(patient_id)
        if not document:
            return {"error": f"No patient found with id '{patient_id}'"}
        return document
    except Exception as exc:
        logger.exception("fetch_patient_data failed")
        return {"error": f"Patient lookup failed: {exc}"}


def _allergy_parts(allergy: Any) -> tuple[str, str]:
    """Allergy records may be plain strings or {rxcui, description} objects."""
    if isinstance(allergy, dict):
        return (
            str(allergy.get("rxcui", "")),
            str(allergy.get("description", "")).lower(),
        )
    return "", str(allergy).lower()


async def verify_allergies(
    patient_id: str, proposed_drugs: list[str]
) -> dict[str, Any]:
    """Screen candidate drugs against a patient's recorded drug allergies.

    Takes a list, not a single drug: screening only the preferred candidate
    leaves every alternative unchecked, which is the wrong default for a
    clinical system.
    """
    try:
        patient = await _find_patient(patient_id)
        if not patient:
            return {"error": f"No patient found with id '{patient_id}'"}

        # Field name must match what the ingestion writes.
        allergies = patient.get("drug_allergies", [])
        safe: list[str] = []
        conflicts: list[dict[str, Any]] = []

        for drug in proposed_drugs:
            drug_text = str(drug).lower().strip()
            matched = None

            for allergy in allergies:
                rxcui, description = _allergy_parts(allergy)

                if rxcui and rxcui == drug_text:
                    matched = {"allergy": description or rxcui, "matched_on": "rxcui"}
                    break

                stem = description.replace("allergy to", "").strip()
                if stem and (stem in drug_text or drug_text in stem):
                    matched = {"allergy": description, "matched_on": "name"}
                    break

            if matched:
                conflicts.append({"drug": drug, **matched})
            else:
                safe.append(drug)

        return {
            "patient_id": patient_id,
            "recorded_drug_allergies": len(allergies),
            "safe": safe,
            "conflicts": conflicts,
        }
    except Exception as exc:
        logger.exception("verify_allergies failed")
        return {"error": f"Allergy check failed: {exc}"}


async def verify_interactions(
    patient_id: str, proposed_drugs: list[str]
) -> dict[str, Any]:
    """Flag candidates the patient already takes and return current meds.

    A duplicate-therapy screen plus context, not a validated interaction
    database. Named honestly on purpose.
    """
    try:
        patient = await _find_patient(patient_id)
        if not patient:
            return {"error": f"No patient found with id '{patient_id}'"}

        current = patient.get("current_medications", [])
        current_lower = [str(m).lower() for m in current]

        duplicates = [
            {"drug": drug, "already_taking": existing}
            for drug in proposed_drugs
            for existing in current_lower
            if str(drug).lower() in existing
        ]

        return {
            "patient_id": patient_id,
            "current_medications": current,
            "duplicate_therapy": duplicates,
            "note": (
                "Duplicate-therapy screen only. Not a validated interaction "
                "database - assess interaction risk using each candidate's "
                "contraindications and boxed warning."
            ),
        }
    except Exception as exc:
        logger.exception("verify_interactions failed")
        return {"error": f"Interaction check failed: {exc}"}
