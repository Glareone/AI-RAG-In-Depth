# AML Agent — A/B Testing with Arize Phoenix

Compare the impact of prompt changes on generated AML facts using
Phoenix Datasets and Experiments.

---

## Prerequisites

| Requirement | How to check |
|---|---|
| Phoenix running | `docker compose up` in `Arize-Phoenix/` → open `http://localhost:6006` |
| AWS credentials | `aws sts get-caller-identity --profile aleksei-dev-macbook` |
| At least one agent run | `uv run python -m src.main` (produces baseline trace) |

---

## Workflow at a Glance

```
 [baseline run already in Phoenix]
          │
  Step A  │  edit src/agent/prompts.yaml   ← your prompt change
          │
  Step B  │  uv run python ab_testing/compare_runs.py --case-id CASE-2024-001
          │   • fetches baseline span
          │   • re-runs agent with new prompt
          │   • uploads 2-row dataset  →  Phoenix → Datasets
          │   • runs experiment         →  Phoenix → Experiments
          │
  Step C  │  (optional) re-evaluate any older dataset:
          │  uv run python ab_testing/evaluate_run.py --case-id CASE-2024-001
          ▼
 Phoenix UI: Datasets + Experiments boards with per-row metric scores
```

---

## Step-by-Step

### 1. Produce a baseline run

If you have not run the agent yet, do so now:

```bash
uv run python -m src.main
# or for a different case:
AML_CASE_ID=CASE-2024-002 uv run python -m src.main
```

This sends a trace to Phoenix and writes a JSON result to `results/`.

---

### 2. Edit the prompt

Open `src/agent/prompts.yaml` and change `aml_analyst_system` or `aml_formatter`.
This is the **candidate** prompt that will be compared against the baseline.

---

### 3. Run the A/B comparison

```bash
uv run python ab_testing/compare_runs.py --case-id CASE-2024-001
```

What happens internally:
1. Fetches the most recent `"LangGraph"` root span for the case from Phoenix.
2. Re-runs the agent with your updated prompt (candidate run).
3. Waits 5 s for Phoenix to ingest the new trace, then fetches the candidate span.
4. Uploads a **2-row Phoenix Dataset** (`baseline` + `candidate`).
5. Runs a **Phoenix Experiment** on the dataset with four evaluators:

| Evaluator | Method | Score |
|---|---|---|
| `token_precision` | Bag-of-words overlap | 0–1 |
| `token_recall` | Bag-of-words overlap | 0–1 |
| `token_f1` | Harmonic mean of above | 0–1 |
| `geval_recall` | Bedrock Claude: % of reference facts covered | 0–1 |
| `geval_precision` | Bedrock Claude: % of generated facts grounded | 0–1 |
| `risk_level_match` | Exact string match vs ground truth | 0 or 1 |
| `recommendation_match` | Exact string match vs ground truth | 0 or 1 |

At the end, the script prints a direct URL to the experiment.

---

### 4. View results in Phoenix

1. Open `http://localhost:6006`.
2. Go to **Datasets** — find `aml-ab-CASE-2024-001-<timestamp>`.
   - Two rows: `baseline` and `candidate`.
   - `input` columns: `case_id`, `prompt_version`, `raw_input`.
   - `output` columns: `facts`, `risk_level`, `recommendation`, `analyst_summary`.
3. Go to **Experiments** — find `ab-eval-CASE-2024-001-<timestamp>`.
   - Each row shows the task output and all evaluator scores side-by-side.
   - Compare `geval_recall` between baseline and candidate to see if the new
     prompt captures more reference facts.
   - Check `geval_precision` to ensure no hallucinations were introduced.
   - `risk_level_match` and `recommendation_match` must both be `PASS` (1.0).

---

### 5. Re-evaluate an older dataset (optional)

```bash
# Auto-select the most recent aml-ab-CASE-2024-001-* dataset:
uv run python ab_testing/evaluate_run.py --case-id CASE-2024-001

# Or target a specific dataset by name:
uv run python ab_testing/evaluate_run.py \
  --dataset-name aml-ab-CASE-2024-001-20260305T120714
```

Use this when you want to apply updated evaluators to a previously created
dataset without re-running the agent.

---

## Ground Truth

Reference facts live in `ab_testing/ground_truth/<case_id>.json`.
Each file contains:

```json
{
  "case_id": "CASE-2024-001",
  "expected_risk_level": "CRITICAL",
  "expected_recommendation": "FILE_SAR",
  "expected_facts": [
    "Three wire transfers structured below the $10,000 CTR threshold ...",
    "..."
  ]
}
```

To add a new case, create `CASE-2024-002.json` following the same schema.
The evaluators pick up the file automatically via `input["case_id"]`.

---

## File Overview

| File | Purpose |
|---|---|
| `compare_runs.py` | Full A/B pipeline: spans → dataset → experiment |
| `evaluate_run.py` | Re-evaluate any existing dataset on demand |
| `evaluators.py` | Shared evaluator functions and `build_evaluators()` factory |
| `ground_truth/` | Reference facts per case (JSON) |
| `PLAN.md` | Detailed design notes and future extensions |
