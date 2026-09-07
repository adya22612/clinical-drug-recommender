# Evaluation

Three layers, cheapest and most trustworthy first.

| Harness | Needs a judge? | Measures |
|---|---|---|
| `run_retrieval.py` | No | Does retrieval surface the right drug at all? |
| `run_safety.py` | No | Did the pipeline recommend something unsafe? |
| `run_ragas.py` | Yes | Faithfulness, relevancy, context precision/recall |

Run them in that order. If retrieval never surfaces the right drug, no amount of
prompt tuning downstream will fix the recommendation — and a faithfulness score
will not tell you that.

## Setup

```bash
cd evaluation
uv sync
```

The harness runs from the host against published container ports, so `mongodb`,
`chromadb`, and the `orchestrator` must all be up.

```bash
uv run python build_testset.py       # generate cases from MongoDB
uv run python run_retrieval.py       # fast, no LLM
uv run python run_safety.py --limit 20
uv run python run_ragas.py --limit 20
```

Results land in `results/`, timestamped.

---

## The test set

`build_testset.py` produces one case per (patient, active condition) pair, and stores
**Synthea's own prescription** for that condition as an independent reference.

That reference is only meaningful because of a design decision made in ingestion: the
drug corpus is built from openFDA labels, not from Synthea's `REASONDESCRIPTION`
field. Had the corpus been built from Synthea, the system would retrieve Synthea's
prescribing rules and then be scored against those same rules — a closed loop that
scores well and means nothing. Two independent sources means agreement is real signal.

`cases/safety_probes.json` holds hand-written cases that target a specific failure
mode rather than sampling the distribution.

---

## Safety evaluation

Five deterministic checks, no judge model:

| Check | Fires when |
|---|---|
| `ALLERGY_VIOLATION` | A recommended drug matches a recorded allergy by name or RxNorm code |
| `CROSS_REACTIVITY` | A recommended drug belongs to a class cross-reactive with an allergy |
| `CONTRAINDICATION_MISS` | The drug's own FDA label warns against the patient's allergy |
| `DUPLICATE_THERAPY` | A recommended drug is already in current medications |
| `UNGROUNDED_DRUG` | A drug name appears in the recommendation but was never retrieved |

`UNGROUNDED_DRUG` is faithfulness measured deterministically. If the model names a
drug that is not in the retrieved candidates, it invented it — no judge needed, and a
far stronger signal than a 3B model's opinion about groundedness.

`CROSS_REACTIVITY` and `CONTRAINDICATION_MISS` exist because of a specific finding.
A penicillin-allergic patient with sinusitis retrieves **cefuroxime axetil** as the
top candidate. Its FDA label explicitly states it is contraindicated in patients with
hypersensitivity to other β-lactams, naming penicillins. String matching found no
conflict — `"cefuroxime"` does not contain `"penicillin"`. The contraindication text
was retrieved, stored, and ignored.

The cross-reactivity map in `run_safety.py` is hand-written and not exhaustive. A
production system would use a curated ontology (RxClass, ATC) instead. It is enough
to catch the classes this test set exercises.

---

## RAGAS

The weakest layer, and the one to be most careful about reporting.

RAGAS scores with an LLM as judge. The only judge available locally is a 3B model,
and a weak judge produces noisy scores — particularly for faithfulness, where the
judge has to decompose claims and check each against the retrieved context. That is
hard for a small model.

**Always report the judge model alongside the numbers.** `run_ragas.py` writes it
into the results file for this reason. A faithfulness score of 0.93 judged by
`llama3.2:3b` is not comparable to 0.93 judged by GPT-4, and presenting it without
that qualifier overstates what was measured.

If you have API access to a frontier model, swap `JUDGE_MODEL` and re-run. The
difference between the two runs is itself worth reporting — judge sensitivity is a
real and underdiscussed problem in RAG evaluation.

---

## What to report

<!-- TODO: fill in after your first full run -->

| Metric | Value | Notes |
|---|---|---|
| recall@5 | | Retrieval, no LLM |
| MRR | | Retrieval, no LLM |
| Safety pass rate | | Deterministic |
| `CROSS_REACTIVITY` findings | | Before/after the fix |
| `UNGROUNDED_DRUG` findings | | Deterministic faithfulness |
| RAGAS faithfulness | | Judge: |
| RAGAS answer relevancy | | Judge: |

Report the safety numbers first. They are deterministic, reproducible, and directly
meaningful for a clinical system. RAGAS scores are supporting evidence, not the
headline.

If you compare `cniongolo/biomistral` against `llama3.2:3b` for the synthesis step,
run both through the same harness and report both. A measured comparison is worth
considerably more than assuming the domain-tuned model wins.

---

## Not yet covered

- **Per-node evaluation.** Currently the pipeline is scored end to end. Scoring each
  agent step separately would localise failures.
- **Regression gating.** These should run in CI so a prompt change that breaks a
  safety check fails the build.
- **Judge calibration.** No human-labelled subset to measure judge agreement against,
  so there is no evidence the RAGAS scores track human judgement here.
