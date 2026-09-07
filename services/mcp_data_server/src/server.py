"""MCP server exposing clinical data as standardised tools.

The docstrings below are not documentation - FastMCP publishes them as each
tool's `description` over the protocol, and they are the only thing an LLM
reads when deciding whether to call a tool. Clients must not hardcode their
own descriptions; the server owns the contract, and that is what makes this
server usable by Claude Desktop, Inspector, or any other MCP client.

Run:     uv run python -m src.server
Inspect: npx @modelcontextprotocol/inspector
         then connect to http://localhost:8000/sse
"""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
import sys
from src.config import settings
from src.tools.clinical import search_drug_literature
from src.tools.patient import (
    fetch_patient_data,
    verify_allergies,
    verify_interactions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Settings come from src.config - the duplicate Settings class that used to
# live here defined a third set of defaults that drifted from the other two.
mcp = FastMCP(
    "Clinical Data Server",
    host=settings.MCP_HOST,
    port=settings.MCP_PORT,
)


@mcp.tool()
async def get_patient_history(patient_id: str) -> dict[str, Any]:
    """Retrieve a patient's full clinical record by their patient ID.

    Returns demographics (age, gender), active conditions, known drug
    allergies, non-drug allergies, and current medications. Call this first,
    before recommending anything, so later checks have the patient's
    allergies and current medications available.

    Returns a dict with an "error" key if no patient matches the ID.
    """
    return await fetch_patient_data(patient_id)


@mcp.tool()
async def query_drug_database(disease: str, top_k: int = 5) -> dict[str, Any]:
    """Find drugs whose FDA-approved indications match a condition.

    Searches FDA drug label text (approved indications, contraindications,
    boxed warnings) by semantic similarity. Pass the condition name alone,
    for example "type 2 diabetes mellitus" - do not include the patient ID
    or other context in this argument.

    Returns a "candidates" list, each with generic name, RxNorm code,
    approved indications, contraindications and boxed warning. These are
    candidates only: they have NOT been screened against the patient's
    allergies or current medications. Always pass every candidate to
    check_allergies and check_interactions before recommending any of them.
    """
    return await search_drug_literature(disease, top_k)


@mcp.tool()
async def check_allergies(
    patient_id: str, proposed_drugs: list[str]
) -> dict[str, Any]:
    """Screen a list of candidate drugs against a patient's known allergies.

    Pass ALL candidate drugs returned by query_drug_database, not only the
    preferred one - screening a single drug leaves the alternatives
    unchecked. Accepts generic names or RxNorm codes.

    Returns "safe" (no allergy conflict found) and "conflicts" (each with the
    drug, the matching allergy, and how it matched). A drug in "safe" means
    no recorded allergy matched; it is not a guarantee of overall safety -
    check_interactions still applies.
    """
    return await verify_allergies(patient_id, proposed_drugs)


@mcp.tool()
async def check_interactions(
    patient_id: str, proposed_drugs: list[str]
) -> dict[str, Any]:
    """Compare candidate drugs against the patient's current medications.

    Flags duplicate therapy (a candidate the patient already takes) and
    returns the current medication list so interaction risk can be reasoned
    about alongside each candidate's contraindications and boxed warning.

    This is a duplicate-therapy and context check, not a validated
    interaction database. Treat the output as information for the clinician,
    never as clearance to prescribe.
    """
    return await verify_interactions(patient_id, proposed_drugs)


if __name__ == "__main__":
    if settings.MCP_TRANSPORT == "stdio":
        # stdout is the protocol channel - logs must not go there
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)

    # Chroma connects lazily, so the embedding model would otherwise load
    # inside the first tool call. Pay that cost at boot instead, and fail
    # loudly here if Chroma is unreachable.
    from src.database.chroma import ChromaClient

    logger.info("Warming up Chroma and the embedding model...")
    collection = ChromaClient.get_collection(settings.CHROMA_COLLECTION)
    logger.info("Ready - %d drug documents indexed.", collection.count())

    if settings.MCP_TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    else:
        logger.info("MCP server on %s:%s", settings.MCP_HOST, settings.MCP_PORT)
        mcp.run(transport="sse")
