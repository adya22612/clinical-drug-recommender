"""
ETL pipeline for the Personalized Drug Recommender (v2).

Two independent data sources:
  - Synthea  -> synthetic patient records (demographics, conditions, allergies, meds)
  - openFDA  -> drug knowledge base (approved indications, pharmacology, warnings)

They are joined on RxNorm codes (Synthea medications.CODE <-> openFDA openfda.rxcui).

Keeping the knowledge base independent of Synthea's prescribing rules matters:
Synthea's own prescriptions can then be used as an independent reference to
sanity-check recommendations, rather than being the source of them.

Run (from services/mcp_data_server):  uv run python scripts/ingest_data.py
"""

import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import asyncio
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import requests
from motor.motor_asyncio import AsyncIOMotorClient

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths - anchored to the repo root, not the working directory
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = DATA_DIR / "synthea.zip"
EXTRACT_PATH = DATA_DIR / "synthea_csv"
FDA_CACHE_PATH = DATA_DIR / "openfda_cache.json"

SYNTHEA_URL = (
    "https://synthetichealth.github.io/synthea-sample-data/downloads/"
    "synthea_sample_data_csv_apr2020.zip"
)

FDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
FDA_DAILY_BUDGET = 800  # openFDA allows 1000/day without a key; leave headroom

# Allergy descriptions matching these are environmental or food, not drug allergies.
# Used only as a fallback when the CSV has no CATEGORY column (older Synthea exports).
NON_DRUG_ALLERGY_TERMS = {
    "pollen", "mold", "mould", "dust mite", "dander", "grass", "tree",
    "latex", "bee", "wasp", "peanut", "nut", "shellfish", "fish", "egg",
    "milk", "dairy", "wheat", "soy", "gluten", "sesame", "strawberr",
    "animal", "cat", "dog", "house dust", "organism", "food",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def safe_first(payload: dict, key: str, default: str) -> str:
    """openFDA fields are lists that may be present-but-empty. Never index blindly."""
    value = payload.get(key)
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value.strip():
        return value
    return default


def clean_text(text: str, limit: int) -> str:
    """Collapse whitespace and truncate for the LLM context window."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def read_csv_subset(path: Path, wanted: list[str]) -> pd.DataFrame:
    """
    Read only the requested columns that actually exist.

    Synthea's schema has changed over releases - the Apr 2020 export has no
    CATEGORY/TYPE columns on allergies.csv, so requesting them outright raises.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    present = [c for c in wanted if c in header]
    missing = set(wanted) - set(present)
    if missing:
        logger.warning("%s: columns not in this export: %s", path.name, sorted(missing))
    return pd.read_csv(path, usecols=present, low_memory=False)


def sanitize_metadata(record: dict) -> dict[str, Any]:
    """ChromaDB accepts only str/int/float/bool. NaN and None will throw."""
    clean = {}
    for key, value in record.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def find_csv_dir() -> Path:
    """The zip layout is not guaranteed - locate the folder holding patients.csv."""
    for candidate in EXTRACT_PATH.rglob("patients.csv"):
        return candidate.parent
    raise FileNotFoundError(f"patients.csv not found anywhere under {EXTRACT_PATH}")


def classify_allergies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split allergies into drug vs. non-drug.

    Synthea's allergy table skews heavily environmental (pollen, mold, dander).
    Without this split, allergy_check compares drug names against 'Mold (organism)
    allergy' forever and never fires.
    """
    if "CATEGORY" in df.columns:
        df["is_drug_allergy"] = df["CATEGORY"].str.lower().eq("medication")
        return df

    if "SYSTEM" in df.columns:
        # Synthea codes medication allergies with RxNorm and environmental /
        # food ones with SNOMED-CT. More reliable than keyword matching.
        logger.info("No CATEGORY column - classifying by coding SYSTEM.")
        df["is_drug_allergy"] = df["SYSTEM"].str.upper().eq("RXNORM")
        return df

    logger.info("No CATEGORY column - falling back to keyword classification.")
    lowered = df["DESCRIPTION"].str.lower()
    is_environmental = lowered.apply(
        lambda text: any(term in text for term in NON_DRUG_ALLERGY_TERMS)
    )
    df["is_drug_allergy"] = ~is_environmental
    return df


def guess_generic_name(label: str) -> str:
    """
    Fallback when the RxNorm lookup misses.

    Synthea labels routinely start with a number ('1 ML Enoxaparin sodium...',
    '24 HR Metformin hydrochloride...'), so taking the first token is wrong.
    Take the first alphabetic token that isn't a unit or dose form.
    """
    noise = {
        "ml", "mg", "hr", "actuat", "oral", "tablet", "capsule", "injection",
        "solution", "suspension", "film", "coated", "extended", "release",
        "sodium", "hydrochloride", "sulfate", "prefilled", "syringe", "auto",
        "injector", "topical", "cream", "ointment", "patch", "inhaler", "unt",
        "mcg", "meq", "hour", "day", "transdermal", "system", "dose", "pack",
    }
    for token in re.findall(r"[A-Za-z]+", label):
        lowered = token.lower()
        if len(lowered) > 3 and lowered not in noise:
            return lowered
    return ""


def query_openfda(rxcui: str, fallback_name: str) -> dict | None:
    """Try the exact RxNorm join first, then fall back to a generic-name search."""
    attempts = [f'openfda.rxcui:"{rxcui}"']
    if fallback_name:
        attempts.append(f'openfda.generic_name:"{fallback_name}"')

    for query in attempts:
        try:
            response = requests.get(
                FDA_ENDPOINT,
                params={"search": query, "limit": 1},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.debug("openFDA request failed (%s): %s", query, exc)
            continue

        if response.status_code == 404:
            continue  # no match for this query - try the next one
        if response.status_code == 429:
            logger.warning("Rate limited by openFDA. Backing off 60s.")
            time.sleep(60)
            continue
        if response.status_code != 200:
            continue

        results = response.json().get("results") or []
        if results:
            return results[0]

    return None


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------

def download_and_extract():
    """Downloads the open-source Synthea dataset and extracts the CSVs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_PATH.exists():
        logger.info("Dataset already exists locally. Skipping download.")
        return

    logger.info("Downloading Synthea sample dataset...")
    response = requests.get(SYNTHEA_URL, stream=True, timeout=120)
    response.raise_for_status()  # an HTML error page would become a BadZipFile later

    with open(ZIP_PATH, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)

    logger.info("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(EXTRACT_PATH)

    ZIP_PATH.unlink(missing_ok=True)
    logger.info("Download and extraction complete.")


# --------------------------------------------------------------------------
# 2. Transform
# --------------------------------------------------------------------------

def engineer_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges and cleans the Synthea CSVs into a nested document structure.

    Returns (patients, drug_catalog). The drug catalog is one row per distinct
    RxNorm code - clinical content is fetched separately in
    enrich_drug_knowledge(), so the knowledge base does not simply replay
    Synthea's own prescribing rules.
    """
    csv_dir = find_csv_dir()
    logger.info("Loading raw CSVs into Pandas from %s ...", csv_dir)

    df_patients = read_csv_subset(
        csv_dir / "patients.csv",
        ["Id", "BIRTHDATE", "DEATHDATE", "FIRST", "LAST", "GENDER"],
    )
    df_conditions = read_csv_subset(
        csv_dir / "conditions.csv", ["PATIENT", "DESCRIPTION", "STOP"]
    )
    df_allergies = read_csv_subset(
        csv_dir / "allergies.csv",
        ["PATIENT", "DESCRIPTION", "CODE", "SYSTEM", "CATEGORY", "STOP"],
    )
    df_medications = read_csv_subset(
        csv_dir / "medications.csv", ["PATIENT", "DESCRIPTION", "CODE", "STOP"]
    )

    logger.info("Transforming and aggregating data...")

    # Living patients only - recommending drugs to deceased patients is a bad demo
    if "DEATHDATE" in df_patients.columns:
        before = len(df_patients)
        df_patients = df_patients[df_patients["DEATHDATE"].isna()].copy()
        logger.info("Kept %d living patients (dropped %d).",
                    len(df_patients), before - len(df_patients))
        df_patients = df_patients.drop(columns=["DEATHDATE"])

    # --- Active conditions only -------------------------------------------
    # A row with a STOP date is resolved. Without this filter a 2014 case of
    # pneumonia shows up as an active condition today.
    active_conditions = df_conditions[df_conditions["STOP"].isna()]
    conditions_grouped = (
        active_conditions.groupby("PATIENT")["DESCRIPTION"]
        .apply(lambda s: sorted(set(s)))  # Synthea repeats rows across encounters
        .reset_index(name="active_conditions")
    )

    # --- Allergies, split by category -------------------------------------
    df_allergies = classify_allergies(df_allergies)
    if "STOP" in df_allergies.columns:
        df_allergies = df_allergies[df_allergies["STOP"].isna()]

    drug_frame = df_allergies[df_allergies["is_drug_allergy"]].copy()
    if "CODE" in drug_frame.columns:
        drug_frame["allergy_record"] = drug_frame.apply(
            lambda r: {
                "rxcui": "" if pd.isna(r["CODE"]) else str(int(r["CODE"])),
                "description": r["DESCRIPTION"],
            },
            axis=1,
        )
    else:
        drug_frame["allergy_record"] = drug_frame["DESCRIPTION"].apply(
            lambda d: {"rxcui": "", "description": d}
        )

    drug_allergies_grouped = (
        drug_frame.groupby("PATIENT")["allergy_record"]
        .apply(lambda s: list({r["description"]: r for r in s}.values()))
        .reset_index(name="drug_allergies")
    )
    other_allergies_grouped = (
        df_allergies[~df_allergies["is_drug_allergy"]]
        .groupby("PATIENT")["DESCRIPTION"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="other_allergies")
    )

    # --- Current medications (no STOP date) --------------------------------
    current_meds = df_medications[df_medications["STOP"].isna()]
    meds_grouped = (
        current_meds.groupby("PATIENT")["DESCRIPTION"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="current_medications")
    )

    # --- Merge everything into the primary patients dataframe --------------
    df_merged = df_patients.rename(
        columns={
            "Id": "patient_id",
            "BIRTHDATE": "birth_date",
            "FIRST": "first_name",
            "LAST": "last_name",
            "GENDER": "gender",
        }
    )

    for frame in (conditions_grouped, drug_allergies_grouped,
                  other_allergies_grouped, meds_grouped):
        df_merged = df_merged.merge(
            frame, left_on="patient_id", right_on="PATIENT", how="left"
        )
        df_merged = df_merged.drop(columns=["PATIENT"])

    # Handle NaN for patients with no conditions, allergies or medications
    for column in ["active_conditions", "drug_allergies",
                   "other_allergies", "current_medications"]:
        df_merged[column] = df_merged[column].apply(
            lambda v: v if isinstance(v, list) else []
        )

    df_merged["age"] = (
        pd.Timestamp.now() - pd.to_datetime(df_merged["birth_date"])
    ).dt.days // 365
    df_merged["birth_date"] = df_merged["birth_date"].astype(str)

    # Patients with no active conditions are useless for a recommender demo
    df_merged = df_merged[df_merged["active_conditions"].str.len() > 0]
    df_merged = df_merged.reset_index(drop=True)

    logger.info(
        "Built %d patient records (%d with at least one drug allergy).",
        len(df_merged),
        int((df_merged["drug_allergies"].str.len() > 0).sum()),
    )

    # --- Drug catalog, deduplicated on RxNorm code -------------------------
    # Deduplicating on CODE rather than the description string keeps the API
    # call count manageable - 'Amoxicillin 250 MG tablet' and 'Amoxicillin
    # 500 MG capsule' are different descriptions but often the same ingredient.
    logger.info("Extracting RxNorm drug catalog...")
    df_drugs = df_medications[["CODE", "DESCRIPTION"]].dropna(subset=["CODE"])
    df_drugs = df_drugs.drop_duplicates(subset=["CODE"])
    df_drugs["CODE"] = df_drugs["CODE"].astype(int).astype(str)
    df_drugs = df_drugs.rename(
        columns={"CODE": "rxcui", "DESCRIPTION": "synthea_label"}
    ).reset_index(drop=True)

        # Synthea's Apr 2020 sample contains no medication allergies: all 597
    # records are environmental or food, SNOMED-coded. Inject a small set so
    # the allergy-screening path is exercisable and evaluable. Flagged as
    # synthetic so it can never be mistaken for source data.
    if int((df_merged["drug_allergies"].str.len() > 0).sum()) == 0:
        logger.warning("No drug allergies in source data - injecting synthetic set.")
        seeded = [
            {"rxcui": "7980",  "description": "Allergy to penicillin V"},
            {"rxcui": "10180", "description": "Allergy to sulfamethoxazole"},
            {"rxcui": "1191",  "description": "Allergy to aspirin"},
            {"rxcui": "5640",  "description": "Allergy to ibuprofen"},
        ]
        for position, index in enumerate(df_merged.index[::3]):
            df_merged.at[index, "drug_allergies"] = [seeded[position % len(seeded)]]
    df_merged["synthetic_allergy"] = df_merged["drug_allergies"].str.len() > 0
    
    logger.info("Found %d distinct RxNorm codes.", len(df_drugs))
    return df_merged, df_drugs


# --------------------------------------------------------------------------
# 3. Enrich
# --------------------------------------------------------------------------

def enrich_drug_knowledge(df_drugs: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the RxNorm catalog with clinical pharmacology from OpenFDA.

    Matches on openfda.rxcui first (an exact join), falling back to a generic
    name search. Responses are cached to disk so re-runs cost zero API calls -
    important given the 1000/day limit without an API key.
    """
    logger.info("Enriching drug data via OpenFDA API. This may take 10-20 minutes...")

    cache: dict[str, Any] = {}
    if FDA_CACHE_PATH.exists():
        cache = json.loads(FDA_CACHE_PATH.read_text())
        logger.info("Loaded %d cached OpenFDA responses.", len(cache))

    calls_made = 0
    enriched_docs = []

    for _, row in df_drugs.iterrows():
        rxcui = row["rxcui"]
        label_text = row["synthea_label"]

        if rxcui in cache:
            payload = cache[rxcui]
        elif calls_made >= FDA_DAILY_BUDGET:
            logger.warning("Hit the API budget of %d calls. Remaining drugs skipped.",
                           FDA_DAILY_BUDGET)
            payload = None
        else:
            payload = query_openfda(rxcui, guess_generic_name(label_text))
            calls_made += 1
            cache[rxcui] = payload
            time.sleep(0.3)  # polite rate-limiting

            if calls_made % 25 == 0:
                logger.info("  %d/%d looked up...", calls_made, len(df_drugs))
                FDA_CACHE_PATH.write_text(json.dumps(cache))

        if not payload:
            continue  # no FDA label - leave it out rather than embedding a placeholder

        openfda = payload.get("openfda", {})
        enriched_docs.append(
            {
                "rxcui": rxcui,
                "synthea_label": label_text,
                "generic_name": safe_first(openfda, "generic_name", label_text),
                "brand_name": safe_first(openfda, "brand_name", ""),
                "indications": clean_text(
                    safe_first(payload, "indications_and_usage", ""), 900
                ),
                "pharmacology": clean_text(
                    safe_first(payload, "clinical_pharmacology", ""), 600
                ),
                "contraindications": clean_text(
                    safe_first(payload, "contraindications", ""), 500
                ),
                "boxed_warning": clean_text(
                    safe_first(payload, "boxed_warning", ""), 400
                ),
            }
        )

    FDA_CACHE_PATH.write_text(json.dumps(cache))

    df_enriched = pd.DataFrame(enriched_docs)
    if df_enriched.empty:
        raise RuntimeError(
            "No drugs could be enriched - check network access to api.fda.gov"
        )

    # Drugs with no stated indication contribute nothing to retrieval
    df_enriched = df_enriched[df_enriched["indications"].str.len() > 0]
    df_enriched = df_enriched.reset_index(drop=True)

    logger.info(
        "Built knowledge base for %d drugs (%d API calls this run).",
        len(df_enriched), calls_made,
    )
    return df_enriched


# --------------------------------------------------------------------------
# 4. Load
# --------------------------------------------------------------------------

async def load_into_mongodb(df_patients: pd.DataFrame):
    """Loads cleaned Pandas DataFrame into MongoDB."""
    logger.info("Loading patient records into MongoDB...")
    client = AsyncIOMotorClient(settings.MONGO_URI)
    try:
        collection = client[settings.MONGO_DB_NAME][settings.PATIENT_COLLECTION]
        records = df_patients.to_dict(orient="records")
        await collection.delete_many({})
        await collection.insert_many(records)
        await collection.create_index("patient_id", unique=True)
        logger.info("Successfully inserted %d patient records.", len(records))
    finally:
        client.close()


def get_chroma_client():
    """HTTP client by default - the MCP server reads from the chromadb
    container, so a local PersistentClient would write data the server can
    never see. Set CHROMA_MODE=persistent only for offline experiments."""
    if os.getenv("CHROMA_MODE", "http") == "persistent":
        store = PROJECT_ROOT / "data" / "chroma"
        store.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(store))
    return chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)


def get_embedding_function():
    """Must match src/database/chroma.py, or query and document vectors land
    in different spaces and retrieval silently returns wrong drugs."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBED_MODEL
    )


def load_into_chromadb(df_drugs: pd.DataFrame):
    """Embeds and loads drug literature into ChromaDB."""
    logger.info("Loading drug literature into ChromaDB...")
    client = get_chroma_client()

    try:
        client.delete_collection(name=settings.CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    documents = df_drugs.apply(
        lambda row: (
            f"Drug: {row['generic_name']}"
            f"{f' (brand: {row.brand_name})' if row['brand_name'] else ''}. "
            f"Approved indications: {row['indications']} "
            f"Contraindications: {row['contraindications'] or 'None listed.'} "
            f"Boxed warning: {row['boxed_warning'] or 'None.'} "
            f"Pharmacology: {row['pharmacology'] or 'Not detailed in label.'}"
        ),
        axis=1,
    ).tolist()

    metadatas = [
        sanitize_metadata(record) for record in df_drugs.to_dict(orient="records")
    ]
    ids = [f"rx_{code}" for code in df_drugs["rxcui"]]

    # Batch to stay well under Chroma's per-call limit
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        logger.info("  embedded %d/%d", min(end, len(documents)), len(documents))

    logger.info("Successfully embedded %d drug documents.", collection.count())


# --------------------------------------------------------------------------

if __name__ == "__main__":
    download_and_extract()
    df_patients, df_drugs = engineer_data()

    # Fetch clinical content from OpenFDA, keyed on RxNorm
    df_drugs_enriched = enrich_drug_knowledge(df_drugs)

    asyncio.run(load_into_mongodb(df_patients))
    load_into_chromadb(df_drugs_enriched)
    logger.info("ETL pipeline completed successfully.")
