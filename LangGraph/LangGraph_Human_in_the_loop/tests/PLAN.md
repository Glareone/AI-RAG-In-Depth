# Test Plan — Phoenix Behavioral Experiments
## PREPARED BY Claude Code -> QWEN CLI

**Philosophy:** Test what unit tests cannot — the actual LLM-driven behavior of the LangGraph agent
with real MCP tools, real interrupts, and real structured outputs.

**Scope:** No unit tests. No e2e tests. No mocked Bedrock. No OTel span assertions.
Four standalone Phoenix experiment scripts, each with a dataset → task → evaluators → `run_experiment`.

---

## Directory Structure

```
tests/
└── experiments/
    ├── 01_tool_calling_behavior.py      # MCP tool selection & ordering
    ├── 02_interrupt_resume_flow.py      # Interrupt/pause/resume mechanics
    ├── 03_financial_report_structure.py # Output schema validity across document types
    └── 04_ratio_calculation_accuracy.py # Numerical accuracy of computed financial ratios
```

---

## How to Run

```bash
# 1. Start Phoenix (keep running in a separate terminal)
uv run phoenix serve

# 2. Run any experiment script
uv run python tests/experiments/01_tool_calling_behavior.py

# 3. View results
# Open http://localhost:6006 — filter by experiment prefix, sort by evaluator scores
```

---

## Script 1: `tests/experiments/01_tool_calling_behavior.py`

**Purpose:** Verify the graph calls the correct MCP tools (document_parser, data_loader) when
given different document types.

### Dataset

```python
import pandas as pd
from phoenix.client import Client

client = Client()

df = pd.DataFrame([
    {"document_id": "ACM_annual_2024",    "document_type": "annual_report",  "expected_tools": ["read_document", "parse_financials"]},
    {"document_id": "TFI_annual_2024",    "document_type": "annual_report",  "expected_tools": ["read_document", "parse_financials"]},
    {"document_id": "ACM_earnings_Q4",    "document_type": "earnings_call",  "expected_tools": ["read_document"]},
])

dataset = client.datasets.create_dataset(
    name="tool-calling-inputs",
    dataframe=df,
    input_keys=["document_id", "document_type", "expected_tools"],
    output_keys=[],
)
```

### Task

```python
def task(example):
    thread_config = {"configurable": {"thread_id": example["document_id"]}}
    result = graph.invoke({"document_id": example["document_id"]}, thread_config)
    return {
        "called_tools": extract_tool_names_from_trace(result),
        "expected_tools": example["expected_tools"],
    }
```

### Evaluators

```python
def has_required_tools(output) -> bool:
    return all(t in output["called_tools"] for t in output["expected_tools"])

def no_extraneous_tools(output) -> bool:
    return all(t in output["expected_tools"] for t in output["called_tools"])

def tool_order_valid(output) -> bool:
    tools = output["called_tools"]
    if "parse_financials" in tools and "calculate_ratios" in tools:
        return tools.index("parse_financials") < tools.index("calculate_ratios")
    return True
```

### Run

```python
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[has_required_tools, no_extraneous_tools, tool_order_valid],
    experiment_prefix="tool-calling-behavior",
)
```

---

## Script 2: `tests/experiments/02_interrupt_resume_flow.py`

**Purpose:** Verify the graph pauses at the correct interrupt nodes and that human feedback
injected via `Command(resume=...)` is reflected in the final report.

### Dataset

```python
df = pd.DataFrame([
    {
        "document_id": "ACM_annual_2024",
        "human_plan_feedback": {"approved": True, "modifications": "Focus on liquidity and debt covenants"},
        "human_risk_feedback": {"approved": True, "additional_risks": []},
    },
    {
        "document_id": "TFI_annual_2024",
        "human_plan_feedback": {"approved": False, "redirect": "Prioritize ARR growth analysis"},
        "human_risk_feedback": {"approved": True, "additional_risks": ["Regulatory risk: SEC"]},
    },
])

dataset = client.datasets.create_dataset(
    name="interrupt-resume-scenarios",
    dataframe=df,
    input_keys=["document_id", "human_plan_feedback", "human_risk_feedback"],
    output_keys=[],
)
```

### Task

```python
from langgraph.types import Command

def task(example):
    thread_config = {"configurable": {"thread_id": example["document_id"]}}

    # Run to interrupt #1
    result = graph.invoke({"document_id": example["document_id"]}, thread_config)
    interrupted_at_1 = result.get("__interrupt__") is not None

    # Resume interrupt #1
    result = graph.invoke(Command(resume=example["human_plan_feedback"]), thread_config)
    interrupted_at_2 = result.get("__interrupt__") is not None

    # Resume interrupt #2
    result = graph.invoke(Command(resume=example["human_risk_feedback"]), thread_config)

    return {
        "interrupted_at_plan": interrupted_at_1,
        "interrupted_at_risk": interrupted_at_2,
        "final_report": result.get("report"),
        "human_plan_feedback": example["human_plan_feedback"],
        "human_risk_feedback": example["human_risk_feedback"],
    }
```

### Evaluators

```python
def both_interrupts_fired(output) -> bool:
    return output["interrupted_at_plan"] and output["interrupted_at_risk"]

def graph_resumes_cleanly(output) -> bool:
    return output["final_report"] is not None

def human_feedback_reflected(output) -> bool:
    report = output["final_report"]
    if not report:
        return False
    # Check that additional_risks injected at interrupt #2 appear in the report
    extra_risks = output["human_risk_feedback"].get("additional_risks", [])
    if extra_risks:
        return any(r in str(report.risk_summary) for r in extra_risks)
    return True
```

### Run

```python
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[both_interrupts_fired, graph_resumes_cleanly, human_feedback_reflected],
    experiment_prefix="interrupt-resume-flow",
)
```

---

## Script 3: `tests/experiments/03_financial_report_structure.py`

**Purpose:** Verify the graph produces a structurally valid `FinancialReport` across documents
of different quality (complete, partial, noisy data).

### Dataset

```python
df = pd.DataFrame([
    {"document_id": "ACM_annual_2024",  "document_quality": "complete", "expected_recommendation": "SELL"},
    {"document_id": "TFI_annual_2024",  "document_quality": "complete", "expected_recommendation": "BUY"},
    {"document_id": "ACM_earnings_Q4",  "document_quality": "partial",  "expected_recommendation": "INSUFFICIENT_DATA"},
])

dataset = client.datasets.create_dataset(
    name="report-structure-inputs",
    dataframe=df,
    input_keys=["document_id", "document_quality", "expected_recommendation"],
    output_keys=[],
)
```

### Task

```python
def task(example):
    thread_config = {"configurable": {"thread_id": example["document_id"]}}

    result = graph.invoke({"document_id": example["document_id"]}, thread_config)

    # Auto-approve both interrupts
    for _ in range(2):
        if result.get("__interrupt__"):
            result = graph.invoke(Command(resume={"approved": True}), thread_config)

    report = result.get("report")
    return {
        "report": report,
        "document_quality": example["document_quality"],
        "expected_recommendation": example["expected_recommendation"],
    }
```

### Evaluators

```python
def report_has_required_fields(output) -> bool:
    report = output["report"]
    if not report:
        return False
    return all(hasattr(report, f) for f in [
        "ticker", "period", "doc_type", "key_metrics",
        "risk_summary", "investment_thesis", "recommendation"
    ])

def recommendation_is_valid_enum(output) -> bool:
    report = output["report"]
    if not report:
        return False
    return report.recommendation in {"BUY", "HOLD", "SELL", "INSUFFICIENT_DATA"}

def metrics_populated_for_complete_docs(output) -> bool:
    if output["document_quality"] != "complete":
        return True  # skip check for partial docs
    report = output["report"]
    return bool(report and report.key_metrics)

def partial_data_yields_insufficient(output) -> bool:
    if output["document_quality"] != "partial":
        return True  # skip
    report = output["report"]
    return report and report.recommendation == "INSUFFICIENT_DATA"
```

### Run

```python
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[
        report_has_required_fields,
        recommendation_is_valid_enum,
        metrics_populated_for_complete_docs,
        partial_data_yields_insufficient,
    ],
    experiment_prefix="financial-report-structure",
)
```

---

## Script 4: `tests/experiments/04_ratio_calculation_accuracy.py`

**Purpose:** Verify that the `compute_ratios` node uses the calculator MCP tools correctly
and produces numerically sensible financial ratios against known ground-truth values.

### Dataset

```python
df = pd.DataFrame([
    {
        "document_id": "ACM_annual_2024",
        "known_metrics": {"current_ratio": 0.87, "debt_to_equity": 3.12, "roe": -0.05},
        "tolerance": 0.1,
    },
    {
        "document_id": "TFI_annual_2024",
        "known_metrics": {"current_ratio": 2.45, "debt_to_equity": 0.31, "roe": 0.22},
        "tolerance": 0.1,
    },
])

dataset = client.datasets.create_dataset(
    name="ratio-accuracy-inputs",
    dataframe=df,
    input_keys=["document_id", "known_metrics", "tolerance"],
    output_keys=[],
)
```

### Task

```python
def task(example):
    thread_config = {"configurable": {"thread_id": example["document_id"]}}

    result = graph.invoke({"document_id": example["document_id"]}, thread_config)
    for _ in range(2):
        if result.get("__interrupt__"):
            result = graph.invoke(Command(resume={"approved": True}), thread_config)

    report = result.get("report")
    return {
        "computed_ratios": report.key_metrics if report else {},
        "expected_ratios": example["known_metrics"],
        "tolerance": example["tolerance"],
    }
```

### Evaluators

```python
import math

def ratios_within_tolerance(output) -> bool:
    computed = output["computed_ratios"]
    expected = output["expected_ratios"]
    tol = output["tolerance"]
    for key, expected_val in expected.items():
        computed_val = computed.get(key)
        if computed_val is None:
            return False
        if abs(computed_val - expected_val) > tol:
            return False
    return True

def ratios_are_finite(output) -> bool:
    return all(
        math.isfinite(v) for v in output["computed_ratios"].values()
    )

def all_expected_ratios_present(output) -> bool:
    return all(k in output["computed_ratios"] for k in output["expected_ratios"])
```

### Run

```python
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=task,
    evaluators=[ratios_within_tolerance, ratios_are_finite, all_expected_ratios_present],
    experiment_prefix="ratio-calculation-accuracy",
)
```

---

## Summary

| Script | What It Tests | Dataset Rows | Evaluators |
|---|---|---|---|
| `01_tool_calling_behavior.py` | MCP tool selection & call order | 3–5 documents | 3 |
| `02_interrupt_resume_flow.py` | Interrupt pause + human feedback injection | 2–4 scenarios | 3 |
| `03_financial_report_structure.py` | Output schema validity across document quality | 3–6 documents | 4 |
| `04_ratio_calculation_accuracy.py` | Numerical accuracy of computed ratios | 2–3 documents with known metrics | 3 |

**Total:** 4 scripts · 13 evaluators · real graph · real LLM · real MCP tools.

Results visible in Phoenix UI at `http://localhost:6006` — filter by evaluator score,
click into any row to see the full LangGraph trace with tool calls and interrupt events.
