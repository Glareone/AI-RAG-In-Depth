# LangGraph ReAct AML Agent

A LangGraph ReAct agent for Anti-Money Laundering (AML) fact extraction.
Analyses ML-flagged cases, fetches transactions and customer data, and produces
structured `AMLReport` outputs. Traces are captured in Arize Phoenix.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| [uv](https://docs.astral.sh/uv/) | latest |
| [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | v2+ |
| Docker (optional) | for Arize Phoenix |

---

## 1. AWS Authentication

The agent calls AWS Bedrock. Authenticate before running.

**Configure once** (first time only):
```bash
aws configure --profile aleksei-dev-macbook
# Enter: Access Key ID, Secret Access Key, region (eu-central-1), output (json)
```

**Verify your session is active:**
```bash
aws sts get-caller-identity --profile aleksei-dev-macbook
```

Expected output:
```json
{
    "UserId": "...",
    "Account": "0000000000",
    "Arn": "arn:aws:iam::xxxxxxx:user/xxxxxxxxx"
}
```

If the call fails, re-run `aws configure` or refresh your credentials.

---

## 2. Project Setup

**Clone and install dependencies:**
```bash
cd LangGraph_ReAct
uv sync
```

**Configure environment:**
```bash
cp .env.example .env
```

`.env` contents (defaults already set, adjust if needed):
```env
AWS_PROFILE=aleksei-dev-macbook
AWS_REGION=eu-central-1
BEDROCK_MODEL_ID=arn:aws:bedrock:eu-central-1:xxxxxxxx:inference-profile/eu.anthropic.claude-sonnet-4-6

PHOENIX_ENDPOINT=http://localhost:6006/
PHOENIX_PROJECT_NAME=aml-react-agent

LOG_LEVEL=INFO
DATA_DIR=data/
```

---

## 3. Start Arize Phoenix (optional, for tracing)
we do have docker-compose file in this repository
```bash
docker compose up -d
```

Phoenix UI: [http://localhost:6006](http://localhost:6006)

If Phoenix is not running, the agent continues without tracing (warning is printed).

---

## 4. Run the Agent

**Analyse the default case (`CASE-2024-001`):**
```bash
uv run python -m src.main
```

**Analyse a specific case:**
```bash
AML_CASE_ID=CASE-2024-002 uv run python -m src.main
```

**Sample output:**
```
============================================================
AML ReAct Agent — analysing case: CASE-2024-001
============================================================

============================================================
CASE:           CASE-2024-001
RISK LEVEL:     HIGH
RECOMMENDATION: FILE_SAR
RATIONALE:      Three wire transfers deliberately kept below the $10,000 CTR threshold...
FACTS:
  [1] STRUCTURING: ...
  [2] SAME-DAY MULTIPLE TRANSACTIONS: ...
  ...
SUMMARY:
  CUST-101 (John Smith) executed three outbound wire transfers...

[saved] results/CASE-2024-001_20240304T120000Z.json
============================================================
```

Results are saved to `results/<case_id>_<timestamp>.json`.

---

## 5. Available Cases

| Case ID | Alert Type | Priority |
|---------|-----------|----------|
| `CASE-2024-001` | Structuring | High |
| `CASE-2024-002` | Layering / Round-trip | Critical |
| `CASE-2024-003` | Smurfing | Medium |

---

## Project Structure

```
LangGraph_ReAct/
├── data/                   # Sample AML cases, customers, transactions
├── results/                # Agent output (gitignored)
├── src/
│   ├── agent/
│   │   ├── graph.py        # LangGraph ReAct graph
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── output.py       # AMLReport Pydantic model
│   │   └── prompts.yaml    # System prompts (XML-tagged)
│   ├── config/
│   │   └── settings.py     # Pydantic BaseSettings
│   ├── tools/              # LangChain @tool functions
│   ├── telemetry/          # Arize Phoenix OTel setup
│   └── main.py             # Entry point
├── .env.example
├── pyproject.toml
└── PLAN.md
```
