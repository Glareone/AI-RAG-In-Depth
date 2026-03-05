# LangGraph ReAct AML Agent — Implementation Plan

## Context

Build a LangGraph ReAct agent for Anti-Money Laundering (AML) fact extraction.
The agent analyzes incoming AML cases (flagged by upstream ML), fetches related transactions
and customer data, and generates structured facts.

**Stack:** AWS Bedrock (Claude Sonnet 4.6), OpenTelemetry + Arize Phoenix, Pydantic-settings.
Foundation for future FastAPI/uvicorn transformation.

---

## Project Structure

```
LangGraph_ReAct/
├── pyproject.toml              # UV-managed, Python 3.12+
├── .env                        # Local config (gitignored)
├── .env.example                # Template
├── .python-version             # 3.12
├── data/
│   ├── transactions.json       # Sample AML transaction data
│   ├── customers.json          # Sample customer profiles
│   └── cases.json              # ML-flagged suspicious cases
├── results/                    # JSON output per run (gitignored)
├── ab_testing/
│   ├── PLAN.md                 # A/B testing design and workflow
│   ├── README.md               # Quick-start: run order and Phoenix navigation
│   ├── evaluators.py           # Shared Phoenix evaluator functions (token F1, G-eval, label match)
│   ├── compare_runs.py         # Fetch spans → upload dataset → run experiment
│   ├── evaluate_run.py         # Re-evaluate any existing dataset on demand
│   └── ground_truth/
│       └── CASE-2024-001.json  # Reference facts, risk level, recommendation
└── src/
    ├── config/
    │   └── settings.py         # Pydantic BaseSettings (Bedrock, Phoenix, app config)
    ├── infrastructure/
    │   └── bedrock.py          # boto3 session factory (single responsibility)
    ├── agent/
    │   ├── state.py            # AgentState (Pydantic BaseModel)
    │   ├── graph.py            # AMLAgentGraph class + build_graph factory
    │   ├── output.py           # AMLReport + RiskAssessment Pydantic models
    │   └── prompts.yaml        # System prompts with XML tags (analyst + formatter)
    ├── tools/
    │   ├── calculator.py       # Risk score, transaction velocity
    │   ├── analyzer.py         # Pattern detection, fact extraction
    │   └── data_loader.py      # Load transactions/customers/cases from JSON
    ├── telemetry/
    │   └── setup.py            # Phoenix OTel registration + LangChain instrumentor
    └── main.py                 # Entry point: telemetry → graph → save AMLReport
```

---

## Implementation Steps

### ✅ Step 1: Project setup
- `pyproject.toml` with UV, Python 3.12+, all deps
- `.env.example` with AWS Bedrock and Phoenix config
- `.python-version`

### ✅ Step 2: Pydantic configuration (`src/config/settings.py`)
- `Settings` with `aws_profile`, `aws_region`, `bedrock_model_id`, Phoenix, app config
- Loads from `.env`

### ✅ Step 3: Sample AML data files (`data/`)
- `transactions.json` — case-scoped transactions (no flags, case is root entity)
- `customers.json` — KYC profiles with risk ratings, PEP status
- `cases.json` — ML-flagged cases with confidence scores and alert types

### ✅ Step 4: Tools (`src/tools/`)
- `data_loader.py` — `get_case_details`, `get_customer_profile`, `search_transactions`, `search_transactions_by_customer`, `get_transaction_details`
- `calculator.py` — `calculate_risk_score`, `calculate_transaction_velocity`
- `analyzer.py` — `analyze_transaction_patterns`, `extract_facts`

### ✅ Step 5: Agent state and graph (`src/agent/`)
- `state.py` — `AgentState` with `messages`, `case_id`, `report`
- `graph.py` — ReAct graph with two LLMs:
  - `react_llm` (tool-bound) drives the 7-step investigation loop
  - `formatter_llm` (`with_structured_output(AMLReport)`) runs once at the end
- Flow: `START → agent → [tools → agent]* → formatter → END`
- `MemorySaver` checkpointer with `thread_id = case_id`

### ✅ Step 6: Structured output (`src/agent/output.py`)
- `AMLReport` Pydantic model: `risk_level`, `recommendation`, `facts`, `analyst_summary`
- Enforced via Claude's native tool-use mechanism (no JSON parsing)

### ✅ Step 7: Prompts (`src/agent/prompts.yaml`)
- XML-tagged sections: `<role>`, `<plan>`, `<considerations>`
- Separate `aml_formatter` prompt for the structured output node
- Tool call sequence enforced explicitly (each tool exactly once)

### ✅ Step 8: Telemetry (`src/telemetry/setup.py`)
- Phoenix OTel `register()` + `LangChainInstrumentor`
- Graceful degradation if Phoenix is not running

### ✅ Step 9: Entry point (`src/main.py`)
- Reads `result["report"]` directly (no parsing)
- Saves `AMLReport` to `results/<case_id>_<timestamp>.json`

### ✅ Step 10: Verification
- `uv sync` — all deps installed
- Phoenix traces visible with LangGraph spans and tool call breakdown
- Agent runs full 7-step loop and produces structured `AMLReport`

---

## Done

### ✅ Step 11a: A/B run comparison (`ab_testing/compare_runs.py`)
- Fetches the most recent root `"LangGraph"` span for a case from Phoenix using
  `SpanQuery().where("name == 'LangGraph'").select(...)` with `root_spans_only=True`
- Re-runs the agent with the updated `prompts.yaml` prompt
- Fetches the new (candidate) span filtered by `after_timestamp`
- Uploads a 2-row (baseline / candidate) Phoenix Dataset
- Immediately runs a Phoenix Experiment on that dataset via `run_experiment`
- Dependency added: `arize-phoenix-client==1.29.0`, `pandas>=2.0.0`

### ✅ Step 11b: Shared evaluators (`ab_testing/evaluators.py`)
- All evaluators follow the Phoenix convention: plain Python functions with
  parameters named `output`, `input` — matched by Phoenix automatically
- **`token_metrics(output, input)`** — bag-of-words precision / recall / F1
  (deterministic, no LLM call)
- **`make_geval_recall(settings)`** — factory returning a closure; Bedrock Claude
  judges what fraction of reference facts are semantically covered
- **`make_geval_precision(settings)`** — factory returning a closure; Claude judges
  what fraction of generated facts are grounded (hallucination detection)
- **`label_match(output, input)`** — exact-match on `risk_level` and `recommendation`
- Ground truth loaded from `ab_testing/ground_truth/<case_id>.json` via `input["case_id"]`
- `passthrough_task` and `build_evaluators(settings)` exported for reuse

### ✅ Step 11c: Standalone evaluation (`ab_testing/evaluate_run.py`)
- Auto-selects the most recent `aml-ab-<case_id>-*` dataset from Phoenix
- Runs `run_experiment` on it using the shared evaluators from `evaluators.py`
- All experiment results (scores + explanations per row) visible in
  Phoenix → Experiments board, side-by-side for baseline and candidate

### ✅ Step 11d: Ground truth (`ab_testing/ground_truth/CASE-2024-001.json`)
- 7 canonical reference facts for `CASE-2024-001` derived from the data files
- `expected_risk_level: CRITICAL`, `expected_recommendation: FILE_SAR`

---

## Pending

---

### 🔲 Step 12: FastAPI / uvicorn wrapper
- `POST /cases/{case_id}/analyse` endpoint
- Returns `AMLReport` as JSON response
- Background task or streaming support
- Reuse `run(case_id)` from `main.py` as the core handler

### 🔲 Step 13: Persistent storage for results
- Replace `results/*.json` file output with database writes
- Candidates: PostgreSQL (via SQLAlchemy), or a document store

### 🔲 Step 14: Third-party enrichment tools
- New tools to fetch flags/risk signals from external sources
- Replace the removed `flags` field on transactions with live lookups

### 🔲 Step 15: Tests
- Unit tests for each tool (`data_loader`, `calculator`, `analyzer`)
- Integration test: full graph run against sample cases
