<div align="center">

# 🏥 Clinical Drug Recommender

**A privacy-preserving clinical decision support system built on the Model Context Protocol**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-ReAct%20Agent-7B2FBE?style=flat)](https://www.llamaindex.ai)
[![MCP](https://img.shields.io/badge/MCP-Tool%20Server-FF6B35?style=flat)](https://modelcontextprotocol.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Orchestrator-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

*Patient data and inference stay fully local. No data leaves the machine.*

[Architecture](#architecture) · [Design Decisions](#design-decisions) · [Findings](#findings) · [Evaluation](#evaluation) · [Running It](#running-it) · [Limitations](#limitations)

</div>

---

## What It Does

Given a patient ID and a condition, the system runs a four-step clinical reasoning pipeline:

1. **Retrieves** the patient's full record — active conditions, drug allergies, current medications
2. **Searches** FDA drug labels by semantic similarity to find indicated treatments
3. **Screens** every candidate against the patient's recorded drug allergies
4. **Checks** every candidate against current medications for duplicate therapy
5. **Synthesises** a clinical summary — what was recommended, what was ruled out, and why

Every tool call is recorded and returned with the response. The audit trail is the point: **a recommendation you cannot trace is not decision support.**

> ⚠️ **Not for clinical use.** This is a portfolio project built on synthetic patient data. It is not validated, not regulated, and must not inform real prescribing decisions.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │  :8501
│          Patient ID + Condition → Recommendation     │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│              FastAPI Orchestrator                    │  :8080
│         LlamaIndex ReAct Agent · Audit Trail         │
└──────────────────┬──────────────────────────────────┘
                   │ MCP over SSE
┌──────────────────▼──────────────────────────────────┐
│              MCP Data Server                        │
│  ┌─────────────────────────────────────────────┐   │
│  │  get_patient_history  │  query_drug_database  │   │
│  │  check_allergies      │  check_interactions   │   │
│  └──────────┬────────────────────┬─────────────┘   │
└─────────────│────────────────────│─────────────────┘
              │                    │
     ┌────────▼──────┐    ┌────────▼──────┐
     │   MongoDB     │    │   ChromaDB    │
     │  Synthea EHR  │    │  FDA Labels   │
     │   Patients    │    │  121 Drugs    │
     └───────────────┘    └───────────────┘

              Ollama (host — GPU access)
              ├── llama3.2:3b   → ReAct orchestration
              └── BioMistral    → Clinical synthesis
```

Four independent services, each with its own dependency set and container. Ollama runs on the host to access the GPU and avoid model duplication in memory.

### MCP Tools

| Tool | Arguments | Returns |
|---|---|---|
| `get_patient_history` | `patient_id` | Demographics, active conditions, drug allergies, current medications |
| `query_drug_database` | `disease`, `top_k` | Candidates with FDA indications, contraindications, boxed warnings |
| `check_allergies` | `patient_id`, `proposed_drugs[]` | `safe` list and `conflicts` list |
| `check_interactions` | `patient_id`, `proposed_drugs[]` | Duplicate therapy flags and current medication context |

The server exposes these over the Model Context Protocol. **It works with Claude Desktop and MCP Inspector out of the box** — not just this project's agent.

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LlamaIndex ReAct (workflow API) |
| LLM orchestration | llama3.2:3b via Ollama |
| Clinical synthesis | BioMistral via Ollama |
| Tool protocol | Model Context Protocol (MCP) over SSE |
| REST API | FastAPI + uvicorn |
| UI | Streamlit |
| Vector store | ChromaDB (cosine similarity, `all-MiniLM-L6-v2`) |
| Document store | MongoDB (Motor async driver) |
| Containerisation | Docker Compose |
| Package management | uv |
| Evaluation | RAGAS + deterministic safety harness |

---

## Design Decisions

### A real MCP server, not a wrapper

Tool descriptions live in the server's docstrings and are published over the protocol. The orchestrator calls `list_tools()` and builds a Pydantic schema from each tool's `inputSchema` — nothing is hardcoded client-side. Adding a tool server-side makes it immediately available to any MCP client, including Claude Desktop and MCP Inspector, with no client change.

### Knowledge base independent of the evaluation source

An earlier design built the drug corpus from Synthea's `REASONDESCRIPTION` field. That would have meant retrieving Synthea's prescribing rules and evaluating against those same rules — a closed loop that scores well and means nothing. The corpus is instead built from **FDA-approved indications**, leaving Synthea's own prescriptions available as a genuinely independent reference to sanity-check recommendations against.

### Joined on RxNorm

Synthea's `medications.CODE` is an RxNorm code; openFDA labels carry `openfda.rxcui`. The two datasets join on that identifier rather than drug-name string matching, which fails on entries like `"24 HR Metformin hydrochloride 500 MG"`.

### Two models, split by role

`llama3.2:3b` drives the ReAct loop; BioMistral handles clinical synthesis. BioMistral is a domain fine-tune with no tool-calling training — it drops out of ReAct format mid-loop on a four-step sequence. Splitting the roles gives each model the job it was trained for.

### Recommendation is synthesis, not a tool

Tools return facts — records, candidates, conflicts. The model reasons over them. Making `generate_recommendation` a tool would hide the reasoning inside a black box and destroy the audit trail that makes this a responsible AI system rather than a black box.

---

## Data

| Source | Provides | License |
|---|---|---|
| [Synthea](https://synthetichealth.github.io/synthea/) (Apr 2020, 100-patient sample) | Synthetic patient EHR records | Open, unrestricted |
| [openFDA drug labels](https://open.fda.gov/apis/drug/label/) | Indications, contraindications, boxed warnings | US public domain |

The ETL pipeline (`scripts/ingest_data.py`) downloads both, joins on RxNorm, and loads MongoDB and ChromaDB. openFDA responses are cached locally — re-runs cost zero API calls.

### Two things found by inspecting the data

**Synthea's sample contains no medication allergies.** All 597 allergy records are environmental or food (mould, dust mite, pollen, dander, shellfish, latex) coded in SNOMED CT, not RxNorm. A drug-allergy screening feature had nothing to screen against. Drug allergies are synthetically injected during ingestion, flagged with `synthetic_allergy: true`, and split into a separate `drug_allergies` field. Environmental allergies are retained in `other_allergies`.

**Conditions must be filtered by `STOP` date.** Synthea repeats condition rows across encounters and includes resolved conditions with a `STOP` value. Without that filter, a 2014 case of pneumonia appears as a current problem and the recommender treats it.

---

## Findings

### β-Lactam cross-reactivity is invisible to string matching

**Test case:** a patient with a recorded penicillin V allergy presenting with chronic sinusitis.

Semantic retrieval returns **cefuroxime axetil** as the top candidate (similarity: 0.356). Cefuroxime's own FDA label states it is *contraindicated in patients with known hypersensitivity to other β-lactam antibacterial drugs, including penicillins*. The allergy screen returned no conflict — `"cefuroxime"` does not contain `"penicillin"`, so exact string matching finds nothing.

The contraindication text was retrieved, stored, and ignored by the string matcher.

This is the central limitation of name-based allergy screening. The safety harness catches it via a cross-reactivity class map and a `CONTRAINDICATION_MISS` check that reads the retrieved FDA label text against the patient's allergies.

### A passing test suite can be measuring emptiness

During evaluation the safety harness reported 100% pass rate before the cross-reactivity fix. The actual reason: BioMistral's synthesis output contained no drug names (it echoed the prompt instead of following it), so `drugs_mentioned()` found nothing to check, and every safety check trivially passed.

The harness was fixed to fall back to `check_allergies`'s `safe` list when synthesis names no drugs — the drugs the pipeline cleared for a clinician are what the safety checks must apply to, regardless of whether synthesis mentioned them.

**A silent `except` in an evaluation harness is how you get a fake 100%.**

### Small-model ReAct orchestration is unreliable

`llama3.2:3b` driving a four-step MCP tool loop exhibited three recurring failures: echoing the JSON Schema envelope as argument values, serialising list arguments as JSON strings, and occasionally abandoning the sequence after two steps. All three were handled defensively in the tool call wrapper, but the root cause is a 3B model being asked to maintain structured format across a multi-turn agentic loop.

Completion rate improves significantly with the envelope unwrapping and `inputSchema`-derived Pydantic schemas; full reliability would require a larger or instruction-tuned-for-tool-use model.

---

## Evaluation

Three layers, cheapest and most trustworthy first:

### Retrieval (deterministic)

```
cases with reference   539
distinct conditions    23
recall@1               69.6%  (375/539)
recall@3               98.1%  (529/539)
recall@5               98.9%  (533/539)
recall@10              99.8%  (538/539)
MRR                    0.835
```

Two independently-sourced datasets (Synthea EHR prescriptions and openFDA labels) agree on the right drug in the top 5 results **98.9% of the time**. Retrieval is not the bottleneck.

Note: 539 cases across 23 distinct conditions means the retrieval query is evaluated per condition, not per patient. The weighted score reflects the real distribution the system sees.

### Safety (deterministic)

Five checks, no LLM judge:

| Check | Fires when |
|---|---|
| `ALLERGY_VIOLATION` | A recommended drug matches a recorded allergy by name or RxNorm code |
| `CROSS_REACTIVITY` | A recommended drug belongs to a class cross-reactive with a known allergy |
| `CONTRAINDICATION_MISS` | The drug's own FDA label warns against the patient's allergy |
| `DUPLICATE_THERAPY` | A recommended drug is already in current medications |
| `UNGROUNDED_DRUG` | A drug appears in the recommendation but was never retrieved |

`UNGROUNDED_DRUG` is faithfulness measured deterministically — no judge model, no ambiguity.

### RAGAS (LLM judge)

<!-- TODO: add numbers after re-running with a reliable synthesis model -->

RAGAS uses an LLM as judge. Always report the judge model alongside scores — a faithfulness result from `llama3.2:3b` is not comparable to one from a frontier model. Results files include `judge_model` for this reason.

---

## Running It

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) (host install — not containerised)

### 1. Models

```bash
ollama pull llama3.2:3b
ollama pull cniongolo/biomistral

# Ollama must listen on all interfaces so containers can reach it
# Set these, then restart Ollama from the system tray
setx OLLAMA_HOST "0.0.0.0"
setx OLLAMA_MAX_LOADED_MODELS "1"
setx OLLAMA_KEEP_ALIVE "30s"
```

### 2. Config

```bash
cp .env.example .env
# Edit .env if needed - defaults are set for local development
```

### 3. Infrastructure

```bash
docker compose up -d mongodb chromadb
docker compose ps  # wait for both to show "healthy"
```

### 4. Ingestion

Run from the host (override container hostnames for local access):

```bash
cd services/mcp_data_server
uv sync

# Windows PowerShell
$env:MONGO_URI="mongodb://localhost:27017"; $env:CHROMA_HOST="localhost"
uv run python scripts/ingest_data.py

# Mac/Linux
MONGO_URI=mongodb://localhost:27017 CHROMA_HOST=localhost \
  uv run python scripts/ingest_data.py
```

Watch for:
- `Built N patient records (M with at least one drug allergy)` — M must be > 0
- `Successfully embedded N drug documents` — expect ~121

### 5. Start everything

```bash
cd ../..
docker compose up -d --build
docker compose ps
```

Open **http://localhost:8501**

### Verify the MCP server independently

```bash
npx @modelcontextprotocol/inspector
# connect to http://localhost:8000/sse, transport: SSE
```

All four tools should appear with descriptions. A third-party MCP client successfully connecting is the real test that the server owns its tool contract.

### Run the evaluation harness

```bash
cd evaluation
uv sync
uv run python build_testset.py
uv run python run_retrieval.py        # fast, no LLM
uv run python run_safety.py --limit 20
uv run python run_ragas.py --limit 10
```

---

## Repository Layout

```
.
├── docker-compose.yml
├── .env.example
├── data/                               # Synthea CSVs + openFDA cache (gitignored)
├── evaluation/
│   ├── build_testset.py                # generates cases from MongoDB
│   ├── run_retrieval.py                # recall@k + MRR, no LLM
│   ├── run_safety.py                   # deterministic safety checks
│   ├── run_ragas.py                    # faithfulness, relevancy, context metrics
│   ├── cases/
│   │   ├── generated.json              # patient-condition pairs with references
│   │   └── safety_probes.json          # hand-written edge cases
│   └── results/                        # timestamped JSON results
└── services/
    ├── mcp_data_server/
    │   ├── scripts/ingest_data.py      # ETL: Synthea + openFDA, joined on RxNorm
    │   └── src/
    │       ├── config.py
    │       ├── database/               # connection management, created once at startup
    │       │   ├── mongo.py
    │       │   └── chroma.py
    │       ├── tools/                  # implementations — no MCP dependency, directly testable
    │       │   ├── patient.py          # get, allergy screen, interaction screen
    │       │   └── clinical.py         # semantic drug retrieval
    │       └── server.py               # MCP tool definitions and docstrings (the API contract)
    ├── orchestrator/
    │   └── src/
    │       ├── mcp_client.py           # SSE connection, tool discovery, argument repair
    │       └── agent.py                # FastAPI, ReAct pipeline, synthesis, audit trail
    └── frontend/
        └── app.py                      # Streamlit UI
```

---

## Limitations

| Limitation | Detail |
|---|---|
| Synthetic data | Synthea patients are statistically generated; drug allergies are injected by this project |
| `check_interactions` | Duplicate-therapy screen only — not a validated drug-drug interaction database |
| Allergy screening | Name-based matching misses cross-reactive drug classes (see Findings) |
| Retrieval | Dense-only, no reranking or hybrid search |
| ReAct reliability | `llama3.2:3b` completes the four-step sequence unreliably; argument shape mangling requires defensive repair |
| Memory | Developed on 8GB RAM + 4GB VRAM; context windows capped at 8192 tokens |
| BioMistral synthesis | Cannot follow synthesis instructions reliably without instruction tuning for this task |

---

## What a Production Version Would Need

- **Validated interaction database** (DrugBank, OpenFDA NDC) replacing the duplicate-therapy screen
- **Cross-reactivity ontology** (RxClass, ATC) replacing the hand-written class map
- **Reranking** over the dense retrieval results (cross-encoder or LLM-as-reranker)
- **Larger orchestration model** or a purpose-trained tool-calling model
- **Regression CI** — the safety harness should run on every prompt change and fail the build on a new finding
- **Real patient data** would require HIPAA-compliant infrastructure — this system's privacy properties (fully local inference, no external API calls) are designed with that path in mind

---

## License

MIT. Synthea data is open and unrestricted. openFDA data is US public domain.

---

<div align="center">

</div>