"""FastAPI orchestrator: a LlamaIndex ReAct agent driving the MCP server's tools.

Two design points:

1. Tool descriptions are discovered from the server (mcp_client.build_tools),
   never hardcoded here.
2. Every tool call is recorded and returned. The audit trail - which tools ran,
   with what arguments, and what came back - is what makes this decision
   support rather than a black box.

Run: uvicorn src.agent:app --host 0.0.0.0 --port 8000
"""

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.mcp_client import build_tools, extract_text, mcp_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Port 8000 to match what the data server actually binds. The previous
    # 8001 default silently worked only because compose overrode it.
    MCP_SERVER_URL: str = "http://mcp_data_server:8000/sse"
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # Orchestration needs reliable ReAct formatting across a four-step loop.
    # BioMistral is a domain fine-tune with no tool-calling training and drops
    # out of format mid-loop; a small general model drives the loop and the
    # medical model writes the final synthesis.
    ORCHESTRATOR_MODEL: str = "llama3.2:3b"
    CLINICAL_MODEL: str = "biomistral"

    REQUEST_TIMEOUT: float = 300.0


settings = Settings()
app = FastAPI(title="Clinical Orchestrator API")

orchestrator_llm = Ollama(
    model=settings.ORCHESTRATOR_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    request_timeout=settings.REQUEST_TIMEOUT,
    context_window=8192,
    additional_kwargs={"num_ctx": 8192},
)
clinical_llm = Ollama(
    model=settings.CLINICAL_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    request_timeout=settings.REQUEST_TIMEOUT,
    context_window=4096,
    additional_kwargs={"num_ctx": 4096},
)
class RecommendationRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    disease_context: str = Field(..., description="The condition requiring treatment")


SYSTEM_PROMPT = """You are a clinical decision support tool for licensed \
healthcare professionals. You do not prescribe; you assemble evidence.

Follow this sequence exactly:
1. get_patient_history for the patient.
2. query_drug_database for the condition.
3. check_allergies with ALL candidate drug names from step 2.
4. check_interactions with the same list.

Rules:
- Never recommend a drug that appears in the allergy conflicts list.
- Never recommend a drug the patient is already taking.
- If a tool returns an error, report it. Do not guess or invent data.
- Base every claim on tool output only. If the tools do not support a
  statement, do not make it.
"""


async def synthesise(patient: dict, disease: str, findings: str) -> str:
    """Final clinical write-up, using the medical model."""
    allergies = ", ".join(str(a) for a in patient.get("drug_allergies", []))
    prompt = f"""Write a concise clinical summary for a physician.

Patient: {patient.get('age')}y {patient.get('gender')}
Active conditions: {', '.join(patient.get('active_conditions', [])) or 'none recorded'}
Known drug allergies: {allergies or 'none recorded'}
Current medications: {', '.join(patient.get('current_medications', [])) or 'none recorded'}
Condition to treat: {disease}

Screened findings:
{findings}

CLINICAL SUMMARY

Recommended: """

    response = await clinical_llm.acomplete(prompt)
    return str(response)


@app.post("/api/v1/recommend")
async def generate_recommendation(request: RecommendationRequest) -> dict[str, Any]:
    """Run the clinical reasoning pipeline over MCP-provided tools."""
    logger.info("Recommendation request: patient=%s", request.patient_id)
    audit: list[dict] = []

    try:
        async with mcp_connection(settings.MCP_SERVER_URL) as session:
            tools = await build_tools(session, audit)

            # Validate before reasoning: a caller can otherwise request
            # treatment for a condition the patient does not have.
            history = await session.call_tool(
                "get_patient_history", {"patient_id": request.patient_id}
            )
            patient = json.loads(extract_text(history))
            if "error" in patient:
                raise HTTPException(status_code=404, detail=patient["error"])

            active = [c.lower() for c in patient.get("active_conditions", [])]
            requested = request.disease_context.lower()
            condition_match = any(requested in c or c in requested for c in active)

            # ReActAgent moved to llama_index.core.agent.workflow. On older
            # llama-index-core use:
            #   from llama_index.core.agent import ReActAgent
            #   agent = ReActAgent.from_tools(tools, llm=..., verbose=True)
            #   reasoning = await agent.achat(user_msg)
            from llama_index.core.agent.workflow import ReActAgent

            agent = ReActAgent(
                tools=tools,
                llm=orchestrator_llm,
                system_prompt=SYSTEM_PROMPT,
            )
            reasoning = await agent.run(
                user_msg=(
                    f"Patient ID: {request.patient_id}. "
                    f"Condition to treat: {request.disease_context}. "
                    f"Work through the sequence and report the screened findings."
                )
            )

            findings = json.dumps(audit, indent=2)[:6000]
            summary = await synthesise(patient, request.disease_context, findings)

        return {
            "status": "success",
            "patient_id": request.patient_id,
            "condition": request.disease_context,
            "condition_in_active_problem_list": condition_match,
            "recommendation": summary,
            "agent_reasoning": str(reasoning),
            "audit_trail": audit,
            "disclaimer": (
                "Decision support only. Requires review by a licensed "
                "clinician before any prescribing decision."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Reasoning pipeline failed",
                "error": str(exc),
                "completed_tool_calls": audit,
            },
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report whether the MCP server is reachable and its tools are described."""
    try:
        async with mcp_connection(settings.MCP_SERVER_URL) as session:
            listing = await session.list_tools()
            return {
                "status": "ok",
                "mcp_server": settings.MCP_SERVER_URL,
                "tools": [
                    {"name": t.name, "has_description": bool(t.description)}
                    for t in listing.tools
                ],
            }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
