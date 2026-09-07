import logging
import streamlit as st
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

# Structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Strict configuration management
class Settings(BaseSettings):
    # Pydantic v2 style - `class Config` is deprecated
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ORCHESTRATOR_URL: str = "http://orchestrator:8000"

settings = Settings()

# Configure the UI
st.set_page_config(
    page_title="Clinical Decision Support",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Personalized Drug Recommender")
st.markdown(
    "Enter patient details and the disease context to generate a privacy-preserving, "
    "AI-driven recommendation powered by BioMistral."
)

# Input Form
with st.form("recommendation_form"):
    patient_id = st.text_input("Patient ID", placeholder="e.g., PT-10492")
    disease_context = st.text_input("Disease Context", placeholder="e.g., Type 2 Diabetes")
    
    submitted = st.form_submit_button("Generate Recommendation")

if submitted:
    if not patient_id or not disease_context:
        st.warning("Please provide both a Patient ID and a Disease Context.")
    else:
        # Visual feedback during the multi-step reasoning process
        with st.spinner("Agent initializing: Fetching history, checking allergies, and querying guidelines..."):
            try:
                # Synchronous HTTP call to the orchestrator container
                response = httpx.post(
                    f"{settings.ORCHESTRATOR_URL}/api/v1/recommend",
                    json={"patient_id": patient_id, "disease_context": disease_context},
                    timeout=600.0  # A four-tool ReAct loop on CPU inference is slow
                )
                response.raise_for_status()
                data = response.json()
                
                st.success("Recommendation Pipeline Completed")
                
                st.markdown("### Clinical Recommendation")
                st.info(data.get("recommendation", "No response generated."))

                if not data.get("condition_in_active_problem_list", True):
                    st.warning(
                        "The requested condition is not on this patient's "
                        "active problem list. Verify before acting."
                    )

                # The audit trail is the point of the architecture - showing
                # which tools ran with what arguments is what separates this
                # from an opaque model call.
                with st.expander("Audit trail - tools called"):
                    for step in data.get("audit_trail", []):
                        st.markdown(f"**{step['tool']}**")
                        st.json(step["arguments"])
                        st.code(step["result"][:1500], language="json")

                with st.expander("Agent reasoning"):
                    st.text(data.get("agent_reasoning", ""))

                st.caption(data.get("disclaimer", ""))
                
            except httpx.HTTPStatusError as e:
                # Surface the server's actual error instead of a generic
                # "containers are down" message that is usually wrong.
                logger.error("Orchestrator returned %s", e.response.status_code)
                try:
                    st.error(e.response.json().get("detail", e.response.text))
                except Exception:
                    st.error(f"Orchestrator error {e.response.status_code}")
            except httpx.HTTPError as e:
                logger.error("Orchestrator request failed: %s", e)
                st.error(
                    "Could not reach the orchestrator. Check that all "
                    "containers are running and the models are pulled."
                )