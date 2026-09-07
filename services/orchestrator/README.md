# Orchestrator

FastAPI service running a LlamaIndex ReAct agent against the MCP data server's tools.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/recommend` | Run the pipeline for a patient and condition |
| `GET` | `/health` | Verify MCP connectivity and that every tool advertises a description |

`/health` reports `has_description` per tool — a fast check that the server's docstrings
are reaching the protocol.

## Response shape

```json
{
  "status": "success",
  "recommendation": "...",
  "agent_reasoning": "...",
  "audit_trail": [{"tool": "...", "arguments": {}, "result": "..."}],
  "condition_in_active_problem_list": true,
  "disclaimer": "..."
}
```

The audit trail is returned on failure too, under `completed_tool_calls` — so a partial
run still shows which tools ran before the error.

## Design

**Tools are discovered, not declared.** `mcp_client.build_tools()` calls `list_tools()` and
constructs LlamaIndex tools from what the server advertises, including a Pydantic schema
built from each tool's `inputSchema`. Without that schema the model guesses argument names,
burning a failed call per tool before it lands on the right ones.

**Two models.** `llama3.2:3b` drives the ReAct loop; BioMistral writes the clinical
synthesis. BioMistral has no tool-calling training and drops out of ReAct format across a
four-step loop; the small general model handles the format reliably. Each does the job it
was trained for.

**The request is validated before reasoning.** `get_patient_history` is called directly
first, and the requested condition is checked against the patient's active problem list.
A caller can otherwise request treatment for a condition the patient does not have.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `MCP_SERVER_URL` | `http://mcp_data_server:8000/sse` | Compose service name, not localhost |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama runs on the host |
| `ORCHESTRATOR_MODEL` | `llama3.2:3b` | Drives the ReAct loop |
| `CLINICAL_MODEL` | `cniongolo/biomistral` | Writes the synthesis |

Context windows are capped (`num_ctx`) to keep the KV cache within an 8GB budget. Ollama
otherwise defaults to llama3.2's full 128k window, which needs ~14GB for the cache alone.
