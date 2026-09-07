# Frontend

Streamlit UI for the clinical recommender. Stateless — it posts to the orchestrator and
renders the response.

## Running

```bash
uv sync
ORCHESTRATOR_URL=http://localhost:8000 uv run streamlit run app.py
```

In Docker it reaches the orchestrator at `http://orchestrator:8000` and is the only
service published to the host, on port 8501.

## What it shows

- The clinical recommendation
- A warning when the requested condition is not on the patient's active problem list
- **Audit trail** — every tool call with its arguments and result
- Agent reasoning (the raw ReAct trace)

The audit trail expander is the part worth looking at. A recommendation you cannot trace
back to the tools that produced it is not decision support.

## Timeouts

The request timeout is deliberately long. A four-tool ReAct loop plus synthesis on CPU
inference takes minutes, not seconds.
