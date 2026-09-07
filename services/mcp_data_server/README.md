# MCP Data Server

Exposes clinical data as MCP tools over SSE. Backed by MongoDB (patients) and ChromaDB
(FDA drug labels).

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `get_patient_history` | `patient_id` | Demographics, active conditions, drug allergies, current medications |
| `query_drug_database` | `disease`, `top_k` | Candidate drugs with indications, contraindications, boxed warnings |
| `check_allergies` | `patient_id`, `proposed_drugs[]` | `safe` and `conflicts` lists |
| `check_interactions` | `patient_id`, `proposed_drugs[]` | Duplicate therapy against current medications |

**Tool descriptions come from the docstrings in `src/server.py`.** FastMCP publishes them
over the protocol, and they are the only thing an LLM reads when deciding whether to call
a tool. Clients must not hardcode their own — doing so makes the server unusable by any
other MCP client. Treat those docstrings as an API contract, not comments.

`check_allergies` and `check_interactions` take a **list**. Screening only the preferred
candidate leaves the alternatives unchecked, which is the wrong default for a clinical
system.

## Layout

```
scripts/ingest_data.py   ETL: Synthea + openFDA joined on RxNorm
src/
├── config.py            settings (all have defaults; compose overrides hosts)
├── database/            connection management, created once at startup
├── tools/               implementations - plain async functions, no MCP dependency
└── server.py            MCP tool definitions
```

Tools have no MCP dependency so they can be unit tested directly, without a running
server. `server.py` is a thin wrapper that owns the protocol contract.

## Running

```bash
uv sync

# Local, against published container ports
MONGO_URI=mongodb://localhost:27017 CHROMA_HOST=localhost \
  MCP_TRANSPORT=stdio uv run python -m src.server
```

`MCP_TRANSPORT=stdio` is for local testing with MCP Inspector or a Python client. In
Docker the transport is SSE, because the orchestrator runs in a separate container and
cannot spawn the server as a subprocess.

Under stdio, logs go to **stderr** — stdout is the protocol channel and anything written
there corrupts the stream.

## Ingestion

```bash
MONGO_URI=mongodb://localhost:27017 CHROMA_HOST=localhost \
  uv run python scripts/ingest_data.py
```

Downloads the Synthea sample, fetches openFDA labels keyed on RxNorm, and loads both
stores. openFDA responses are cached to `data/raw/openfda_cache.json`, so re-runs cost no
API calls (the limit is 1000/day without a key).

## Gotcha: embedding function

The embedding model must be **identical** in `scripts/ingest_data.py` and
`src/database/chroma.py`. If they differ, query and document vectors land in different
spaces and retrieval returns plausible-looking nonsense with no error raised. Both read
`EMBED_MODEL` from config for this reason.
