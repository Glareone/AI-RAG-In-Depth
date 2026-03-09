# LangGraph Human-in-the-Loop Financial Document Analyst — Implementation Plan

## Context

Build a LangGraph agent that analyzes financial documents (annual reports, earnings calls,
balance sheets, SEC filings) and produces structured investment-grade insights.

The agent uses **Human-in-the-loop** interrupts at key decision points — the human can
steer the analysis, approve risk conclusions, and correct the agent before it finalizes
its output.

**MCP tools** (Model Context Protocol) provide pluggable, server-based data access:
file-system document reading, web search for market context, and structured data lookups.

**Stack:** AWS Bedrock (Claude Sonnet 4.6), LangGraph `interrupt` / `Command`, MCP adapters,
OpenTelemetry + Arize Phoenix, Pydantic-settings.

---

## Project Structure

```
LangGraph_Human_in_the_loop/
├── pyproject.toml              # UV-managed, Python 3.12+
├── .env                        # Local config (gitignored)
├── .env.example                # Template
├── .python-version             # 3.12
├── PLAN.md                     # This file
├── data/
│   ├── documents/              # Sample financial documents (PDFs / text)
│   └── metadata.json           # Document registry (ticker, type, period)
├── results/                    # JSON output per run (gitignored)
└── src/
    ├── config/
    │   └── settings.py         # Pydantic BaseSettings (Bedrock, Phoenix, app config)
    ├── infrastructure/
    │   └── bedrock.py          # boto3 session factory
    ├── agent/
    │   ├── state.py            # AnalystState (Pydantic BaseModel with interrupt fields)
    │   ├── graph.py            # FinancialAnalystGraph — nodes + interrupt points
    │   ├── output.py           # FinancialReport Pydantic output model
    │   └── prompts.yaml        # System prompts (analyst + formatter)
    ├── tools/
    │   ├── document_parser.py  # Extract text / tables from PDFs and text files
    │   ├── calculator.py       # Financial ratio calculations (P/E, D/E, ROE, etc.)
    │   └── data_loader.py      # Load document registry and financial data files
    ├── mcp_tools/
    │   ├── server.py           # MCP server definition (tools exposed via MCP protocol)
    │   └── client.py           # MCP client adapter for LangGraph (langchain-mcp-adapters)
    └── telemetry/
        └── setup.py            # Phoenix OTel registration + LangChain instrumentor
```

---

## Graph Architecture — Human-in-the-Loop Flow

```
START
  │
  ▼
[plan_analysis]          ← Agent proposes analysis plan
  │
  ▼
[INTERRUPT #1]           ← Human reviews / adjusts plan
  │  human: approve / redirect / add focus areas
  ▼
[extract_data]           ← Agent reads documents via MCP tools, extracts key numbers
  │
  ▼
[compute_ratios]         ← Calculator tools: liquidity, profitability, leverage ratios
  │
  ▼
[identify_risks]         ← Agent identifies material risks and flags
  │
  ▼
[INTERRUPT #2]           ← Human reviews risk flags before finalizing
  │  human: approve / override / add context
  ▼
[format_report]          ← Structured output via with_structured_output(FinancialReport)
  │
  ▼
END
```

### Interrupt Mechanics

- LangGraph `interrupt(value)` pauses the graph and surfaces a value to the caller
- Human input is injected back via `Command(resume=<human_input>)`
- Both interrupt points are named and tracked in `AnalystState`
- The graph is compiled with `MemorySaver` checkpointer (thread-safe persistence)

---

## Implementation Steps

### Step 1: Project setup
- `pyproject.toml` with UV, Python 3.12+, all deps
- `.env.example` with AWS Bedrock and Phoenix config
- `.python-version`, folder structure, `__init__.py` files

### Step 2: Pydantic configuration (`src/config/settings.py`)
- `Settings` with `aws_profile`, `aws_region`, `bedrock_model_id`, Phoenix, app config

### Step 3: Sample financial data (`data/`)
- 2–3 synthetic company financial documents: balance sheet, income statement, earnings call excerpt
- `metadata.json` — document registry (ticker, doc type, fiscal period, file path)

### Step 4: Local tools (`src/tools/`)
- `data_loader.py` — load document registry, fetch document text by ticker/type
- `document_parser.py` — extract text from PDF and `.txt` files
- `calculator.py` — `calculate_liquidity_ratios`, `calculate_profitability_ratios`, `calculate_leverage_ratios`

### Step 5: MCP tools (`src/mcp_tools/`)
- `server.py` — MCP server that exposes document read and financial lookup tools
- `client.py` — `MultiServerMCPClient` from `langchain-mcp-adapters` to integrate MCP tools into LangGraph

### Step 6: Agent state (`src/agent/state.py`)
- `AnalystState` fields:
  - `messages` — conversation history
  - `document_id` — current document under analysis
  - `analysis_plan` — agent-proposed plan (set at node 1)
  - `human_plan_feedback` — human input from interrupt #1
  - `risk_flags` — identified risk list (set at node 3)
  - `human_risk_feedback` — human input from interrupt #2
  - `report` — final `FinancialReport`

### Step 7: Graph with interrupts (`src/agent/graph.py`)
- Two `interrupt()` calls:
  1. After `plan_analysis` — expose `analysis_plan` to human
  2. After `identify_risks` — expose `risk_flags` to human
- Human input injected via `Command(resume=...)` from the caller
- `MemorySaver` checkpointer, `thread_id = document_id`

### Step 8: Structured output (`src/agent/output.py`)
- `FinancialReport` Pydantic model:
  - `ticker`, `period`, `doc_type`
  - `key_metrics` — dict of computed ratios
  - `risk_summary` — list of identified risks (after human review)
  - `investment_thesis` — one-paragraph narrative
  - `recommendation` — enum: BUY / HOLD / SELL / INSUFFICIENT_DATA

### Step 9: Prompts (`src/agent/prompts.yaml`)
- `financial_analyst` — instructs the agent on the investigation steps
- `financial_formatter` — instructs the formatter node to produce `FinancialReport`

### Step 10: Telemetry (`src/telemetry/setup.py`)
- Phoenix OTel `register()` + `LangChainInstrumentor`
- Graceful degradation if Phoenix is not running

### Step 11: Entry point (`src/main.py`)
- Interactive CLI runner demonstrating the interrupt/resume loop
- Prints interrupt payloads, reads human input from stdin, resumes via `Command`
- Saves `FinancialReport` to `results/<ticker>_<timestamp>.json`

---

## Key Concepts Demonstrated

| Concept | Where |
|---|---|
| `interrupt(value)` | `src/agent/graph.py` — plan and risk review nodes |
| `Command(resume=...)` | `src/main.py` — CLI loop injects human input |
| `MemorySaver` checkpointer | `src/agent/graph.py` — persistent thread state |
| MCP tool integration | `src/mcp_tools/` — server + client adapter |
| `with_structured_output` | formatter node in `graph.py` |
| Arize Phoenix tracing | `src/telemetry/setup.py` — full graph spans |

---

## Done

### ✅ Step 1: Project setup
- `.gitignore`, `.env.example`, `.python-version`, `pyproject.toml`, `PLAN.md`
- Folder structure: `src/{config,infrastructure,agent,tools,mcp_tools,telemetry}`, `data/`, `results/`

### ✅ Step 2: Pydantic configuration (`src/config/settings.py`)
- `Settings` with `aws_profile`, `aws_region`, `bedrock_model_id`, Phoenix, app config
- Added `results_dir` field (new vs ReAct)
- Loads from `.env` via `pydantic-settings`

### ✅ Step 3: Sample financial data (`data/`)
- `ACM_annual_report_2024.txt` — ACME Industrial Corp: stressed balance sheet, going concern note, debt covenant discussion, restructuring charges
- `TFI_annual_report_2024.txt` — TechFlow Inc: healthy SaaS company, strong ARR/NRR metrics, low leverage
- `ACM_earnings_call_Q4_2024.txt` — ACME Q4 2024 earnings call transcript with liquidity commentary
- `metadata.json` — document registry by ticker, doc type, and file path

### ✅ Step 4: Local tools (`src/tools/`)
- `data_loader.py` — `load_registry`, `list_documents`, `get_document_metadata`, `get_document_text`, `find_document`
- `document_parser.py` — `extract_text` (.txt, .pdf), `extract_text_preview`
- `calculator.py` — `calculate_liquidity_ratios`, `calculate_profitability_ratios`, `calculate_leverage_ratios`

### ✅ Step 5: MCP tools (`src/mcp_tools/`)
- `server.py` — FastMCP server exposing 5 tools: `list_available_documents`, `get_document`, `get_document_info`, `find_document_by_ticker_and_type`, `preview_document`; launched via stdio transport

---

## Pending

### ✅ Step 2: Pydantic configuration
### ✅ Step 3: Sample financial data
### ✅ Step 4: Local tools
### ✅ Step 5: MCP tools
### Step 6: Agent state
### Step 7: Graph with interrupts
### Step 8: Structured output
### Step 9: Prompts
### Step 10: Telemetry
### Step 11: Entry point + CLI interrupt loop

