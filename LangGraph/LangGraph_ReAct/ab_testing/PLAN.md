# A/B Testing AML Prompts with Arize Phoenix

## Goal

Compare the consistency and quality of generated AML facts across two prompt versions
by capturing LangGraph traces from Arize Phoenix, building a labelled dataset, and
visualising the diff side-by-side in the Phoenix UI.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| `arize-phoenix-client` | `==1.29.0` | Lightweight Phoenix client (no server bundled) |
| Arize Phoenix server | `>=12.x` | Running locally via `docker compose up` |
| A completed agent run | — | At least one trace for the target `case_id` already in Phoenix |

Install:
```bash
uv add "arize-phoenix-client==1.29.0"
```

---

## Workflow

```
 [Baseline run already in Phoenix]
          │
  Step 1  │  fetch_baseline_span(case_id)
          │  SpanQuery().where("name == 'LangGraph'") + filter by case_id + limit=1
          ▼
 baseline_span  ←── input.value, output.value, attribute.tags
          │
  Step 2  │  extract_span_data(span)
          │  pull out: system_prompt, generated_facts, tools_called
          ▼
 baseline_row  (dict ready for dataset)
          │
  Step 3  │  edit prompts.yaml  ← YOU DO THIS MANUALLY before running the script
          │
  Step 4  │  run_agent(case_id)  ← re-runs src/main.py with the updated prompt
          ▼
 new AMLReport written to results/
          │
  Step 5  │  fetch_latest_span(case_id)
          │  same SpanQuery, sorted by start_time DESC, limit=1
          ▼
 candidate_span
          │
  Step 6  │  extract_span_data(span)  →  candidate_row
          │
  Step 7  │  build_dataset(baseline_row, candidate_row)
          │  phoenix_client.datasets.create_dataset(
          │      name=f"aml-ab-{case_id}-{timestamp}",
          │      dataframe=combined_df,
          │      input_keys=["case_id", "prompt_version"],
          │      output_keys=["facts", "risk_level", "recommendation"],
          │  )
          ▼
 Dataset visible in Phoenix UI → Datasets tab
```

---

## Step-by-Step Details

### Step 0 — Install arize-phoenix-client

```bash
# from the LangGraph_ReAct project root
uv add "arize-phoenix-client==1.29.0"
```

The lightweight client exposes `phoenix.client.Client` with `.spans` and `.datasets`
namespaces. It does **not** bundle the Phoenix server — keep `arize-phoenix-otel` for
the OTel tracing side.

---

### Step 1 — Fetch the baseline span

```python
from phoenix.client import Client
from phoenix.trace.dsl import SpanQuery

phoenix_client = Client(base_url="http://localhost:6006")

query = (
    SpanQuery()
    .where("name == 'LangGraph'")   # root LangGraph chain span
    .select(
        input="input.value",
        output="output.value",
        start_time="start_time",
    )
)
df = phoenix_client.spans.get_spans_dataframe(
    query=query,
    project_name="aml-react-agent",
)

# Filter rows whose input contains the target case_id, take the most recent
baseline_df = (
    df[df["input"].str.contains(case_id, na=False)]
    .sort_values("start_time", ascending=False)
    .head(1)
)
```

> **Why the root "LangGraph" span?**
> OpenInference instruments LangGraph with a root CHAIN span named `"LangGraph"`.
> Its `input.value` contains the initial invoke payload (including `case_id` and the
> first HumanMessage) and its `output.value` contains the final state — including the
> serialised `AMLReport` with `facts`, `risk_level`, and `recommendation`.

---

### Step 2 — Extract prompt, facts, and tools

```python
import json

def extract_span_data(span_row: pd.Series) -> dict:
    raw_output = span_row["output"]          # JSON string of final AgentState
    raw_input  = span_row["input"]           # JSON string of initial invoke payload

    output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    report = output.get("report", {})

    return {
        "case_id":        report.get("case_id", ""),
        "risk_level":     report.get("risk_level", ""),
        "recommendation": report.get("recommendation", ""),
        "facts":          "\n".join(report.get("facts", [])),
        "analyst_summary": report.get("analyst_summary", ""),
        "raw_input":      raw_input,
    }
```

---

### Step 3 — Edit the prompt (manual step)

Open `src/agent/prompts.yaml` and update `aml_analyst_system` or `aml_formatter`.
The script does **not** auto-edit prompts — prompt authoring is your responsibility.

---

### Step 4 — Re-run the agent

```python
import os, sys
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("AML_CASE_ID", case_id)

from src.main import run
run(case_id)
```

---

### Step 5 — Fetch the latest span (post-update run)

Same query as Step 1, but take the most recent row after the agent run finishes.
Add a small delay (`time.sleep(3)`) to allow Phoenix to ingest the trace.

---

### Step 6 — Build and upload the dataset

```python
import pandas as pd
from datetime import datetime

rows = [
    {"prompt_version": "baseline", **baseline_data},
    {"prompt_version": "candidate", **candidate_data},
]
combined_df = pd.DataFrame(rows)

timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
dataset = phoenix_client.datasets.create_dataset(
    name=f"aml-ab-{case_id}-{timestamp}",
    dataframe=combined_df,
    input_keys=["case_id", "prompt_version", "raw_input"],
    output_keys=["facts", "risk_level", "recommendation", "analyst_summary"],
)
print(f"Dataset '{dataset.name}' created — open Phoenix → Datasets tab")
```

---

## Dataset Schema

| Column | Source | Description |
|---|---|---|
| `case_id` | `report.case_id` | AML case identifier |
| `prompt_version` | script label | `"baseline"` or `"candidate"` |
| `risk_level` | `report.risk_level` | `CRITICAL / HIGH / MEDIUM / LOW` |
| `recommendation` | `report.recommendation` | `FILE_SAR / ESCALATE_FOR_REVIEW / CLOSE_NO_ACTION` |
| `facts` | `report.facts` (joined) | Newline-separated discrete AML facts |
| `analyst_summary` | `report.analyst_summary` | Narrative summary |
| `raw_input` | `input.value` | Full invoke payload for traceability |

---

## Interpreting Results in Phoenix

1. Navigate to **Datasets** → open the `aml-ab-<case_id>-<timestamp>` dataset.
2. Compare `facts` column row-by-row: baseline vs. candidate.
3. Check whether `risk_level` and `recommendation` are stable across versions.
4. If facts diverge significantly for the same case, the prompt change is too aggressive.
5. If facts converge and `recommendation` is consistent, the candidate prompt is better.

---

## Future Extensions (Step 11 in PLAN.md)

- **Automated scoring**: add an LLM-as-judge evaluator that scores `facts` for
  completeness and accuracy against a ground-truth JSON.
- **Multi-case sweep**: loop over all three cases (`CASE-2024-001/002/003`) and build
  a single combined dataset for statistical comparison.
- **Phoenix Experiments**: use `run_experiment` + `evaluate_experiment` to get
  per-example pass/fail scores directly in the Phoenix UI.
