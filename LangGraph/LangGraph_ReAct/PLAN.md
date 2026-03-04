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
└── src/
    ├── config/
    │   └── settings.py         # Pydantic BaseSettings (Bedrock, Phoenix, app config)
    ├── agent/
    │   ├── state.py            # AgentState (TypedDict with messages + report)
    │   ├── graph.py            # StateGraph: react_llm, formatter_llm, tool_node
    │   ├── output.py           # AMLReport Pydantic model (structured output schema)
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

## Pending

### 🔲 Step 11: A/B Prompt Evaluation with Arize Phoenix
Goal: measure the impact of prompt changes on agent output quality using Phoenix
Experiments and Datasets boards.

**Sub-steps:**

1. **Build a ground-truth dataset**
   - Manually label expected outputs for all 3 sample cases:
     `expected_recommendation`, `expected_risk_level`, `expected_facts` (key facts that must appear)
   - Upload as a Phoenix Dataset via the Phoenix SDK (`px.Client().upload_dataset(...)`)
   - Dataset becomes the stable evaluation baseline across all experiments

2. **Fetch span results from a previous run (baseline)**
   - Query Phoenix for spans from the target project using the Phoenix Client or REST API
   - Extract per-case: `recommendation`, `risk_level`, `facts` from the formatter node output
   - Record as the baseline experiment in Phoenix (`px.Client().log_evaluations(...)`)

3. **Trigger a new run with the updated prompt (variant)**
   - Modify `prompts.yaml` (e.g. updated `<considerations>`, reordered steps, new XML tags)
   - Re-run the agent against all dataset cases
   - Log results as a new experiment in Phoenix linked to the same dataset

4. **Compare using precision and recall**
   - **Precision** — of the facts the agent reported, how many are in the ground truth?
   - **Recall** — of the ground-truth facts, how many did the agent find?
   - **Recommendation accuracy** — exact match of `recommendation` and `risk_level` vs ground truth
   - Log scores per case and aggregate; view side-by-side in the Phoenix Experiments board

5. **Phoenix boards to set up**
   - `Datasets` board: AML ground-truth dataset with input/expected output per case
   - `Experiments` board: one entry per prompt variant with evaluation scores attached

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
