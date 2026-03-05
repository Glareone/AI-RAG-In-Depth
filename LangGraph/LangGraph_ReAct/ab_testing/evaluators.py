"""Shared evaluators for AML fact quality experiments.

All evaluators follow the Phoenix evaluator function convention:
  - Parameter names are matched by Phoenix: `output` (task result),
    `input` (example input dict), `expected` (example output dict).
  - Return type: float, bool, or ExperimentEvaluation (TypedDict with
    score, label, explanation, and optional name keys).

The evaluators here use `input["case_id"]` to look up ground truth from
ab_testing/ground_truth/<case_id>.json, so no ground truth columns need to
live in the dataset itself.

Usage:
    from ab_testing.evaluators import build_evaluators
    evaluators = build_evaluators(settings)
    client.experiments.run_experiment(dataset=ds, task=task, evaluators=evaluators)
"""

import json
import re
from pathlib import Path
from typing import Any

from phoenix.client.resources.experiments.types import ExperimentEvaluation

from src.config import Settings

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"

# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

_gt_cache: dict[str, dict] = {}


def _load_ground_truth(case_id: str) -> dict:
    if case_id not in _gt_cache:
        path = GROUND_TRUTH_DIR / f"{case_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No ground truth file for {case_id}. Expected: {path}"
            )
        _gt_cache[case_id] = json.loads(path.read_text())
    return _gt_cache[case_id]


# ---------------------------------------------------------------------------
# Task helper — used by both compare_runs.py and evaluate_run.py
# ---------------------------------------------------------------------------

def passthrough_task(example: dict[str, Any]) -> dict[str, Any]:
    """Return generated content from the dataset example as the task output.

    Phoenix's run_experiment calls this for each example.  The dataset stores
    facts as a newline-joined string (as written by compare_runs.py); we split
    them back into a list so evaluators can work with individual facts.
    """
    out = example.get("output", {})
    raw_facts = out.get("facts", "")
    return {
        "facts": [f.strip() for f in raw_facts.splitlines() if f.strip()],
        "risk_level": out.get("risk_level", ""),
        "recommendation": out.get("recommendation", ""),
    }


# ---------------------------------------------------------------------------
# Evaluator 1: Token-level precision / recall / F1  (deterministic)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase and split into word tokens, stripping punctuation."""
    return set(re.sub(r"[^\w\s]", " ", text.lower()).split())


def token_metrics(output: dict[str, Any], input: dict[str, Any]) -> list[ExperimentEvaluation]:
    """Bag-of-words overlap between generated and reference facts."""
    case_id = input.get("case_id", "")
    exp_facts: list[str] = _load_ground_truth(case_id).get("expected_facts", [])
    gen_facts: list[str] = output.get("facts", [])

    gen_tokens = _tokenise(" ".join(gen_facts))
    exp_tokens = _tokenise(" ".join(exp_facts))

    if not gen_tokens or not exp_tokens:
        p, r, f1 = 0.0, 0.0, 0.0
    else:
        common = gen_tokens & exp_tokens
        p  = round(len(common) / len(gen_tokens), 4)
        r  = round(len(common) / len(exp_tokens), 4)
        f1 = round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    return [
        ExperimentEvaluation(
            name="token_precision",
            score=p,
            label="HIGH" if p >= 0.7 else "LOW",
            explanation=f"{p:.0%} of generated tokens found in reference facts.",
        ),
        ExperimentEvaluation(
            name="token_recall",
            score=r,
            label="HIGH" if r >= 0.7 else "LOW",
            explanation=f"{r:.0%} of reference tokens found in generated facts.",
        ),
        ExperimentEvaluation(
            name="token_f1",
            score=f1,
            label="HIGH" if f1 >= 0.7 else "LOW",
            explanation=f"F1={f1:.4f} (precision={p:.0%}, recall={r:.0%}).",
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluator 2 & 3: G-eval via Bedrock Claude  (semantic)
# ---------------------------------------------------------------------------

_GEVAL_RECALL_PROMPT = """\
You are an AML evaluation assistant.

<reference_facts>
{reference_facts}
</reference_facts>

<generated_facts>
{generated_facts}
</generated_facts>

<task>
For each REFERENCE fact (in order), decide whether it is semantically covered
by any generated fact. A fact is "covered" if the generated text conveys the
same information, even if worded differently.

Return ONLY a JSON object — no other text:
{{
  "covered": [0 or 1 per reference fact, in order],
  "explanation": "one sentence summarising coverage gaps, or 'All facts covered.'"
}}
</task>"""

_GEVAL_PRECISION_PROMPT = """\
You are an AML evaluation assistant.

<reference_facts>
{reference_facts}
</reference_facts>

<generated_facts>
{generated_facts}
</generated_facts>

<task>
For each GENERATED fact (in order), decide whether it is supported by (i.e.,
consistent with or derivable from) the reference facts. A fact is
"hallucinated" if it introduces information not present in the references.

Return ONLY a JSON object — no other text:
{{
  "supported": [0 or 1 per generated fact, in order],
  "explanation": "one sentence summarising unsupported facts, or 'All facts grounded.'"
}}
</task>"""


def _call_llm(prompt: str, settings: Settings) -> str:
    from langchain_aws import ChatBedrock
    from langchain_core.messages import HumanMessage
    from src.infrastructure import make_bedrock_client

    llm = ChatBedrock(
        model=settings.bedrock_model_id,
        client=make_bedrock_client(settings),
        provider="anthropic",
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


def _parse_geval_json(text: str) -> dict:
    cleaned = re.sub(r"^```[a-z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def make_geval_recall(settings: Settings):
    """Factory: returns a geval_recall evaluator bound to *settings*."""

    def geval_recall(output: dict[str, Any], input: dict[str, Any]) -> ExperimentEvaluation:
        case_id = input.get("case_id", "")
        exp_facts = _load_ground_truth(case_id).get("expected_facts", [])
        gen_facts = output.get("facts", [])

        raw = _call_llm(
            _GEVAL_RECALL_PROMPT.format(
                reference_facts="\n".join(f"- {f}" for f in exp_facts),
                generated_facts="\n".join(f"- {f}" for f in gen_facts),
            ),
            settings,
        )
        result = _parse_geval_json(raw)
        covered = result.get("covered", [])
        score = round(sum(covered) / len(covered), 4) if covered else 0.0

        return ExperimentEvaluation(
            name="geval_recall",
            score=score,
            label="HIGH" if score >= 0.8 else ("MEDIUM" if score >= 0.5 else "LOW"),
            explanation=result.get("explanation", ""),
        )

    return geval_recall


def make_geval_precision(settings: Settings):
    """Factory: returns a geval_precision evaluator bound to *settings*."""

    def geval_precision(output: dict[str, Any], input: dict[str, Any]) -> ExperimentEvaluation:
        case_id = input.get("case_id", "")
        exp_facts = _load_ground_truth(case_id).get("expected_facts", [])
        gen_facts = output.get("facts", [])

        raw = _call_llm(
            _GEVAL_PRECISION_PROMPT.format(
                reference_facts="\n".join(f"- {f}" for f in exp_facts),
                generated_facts="\n".join(f"- {f}" for f in gen_facts),
            ),
            settings,
        )
        result = _parse_geval_json(raw)
        supported = result.get("supported", [])
        score = round(sum(supported) / len(supported), 4) if supported else 0.0

        return ExperimentEvaluation(
            name="geval_precision",
            score=score,
            label="HIGH" if score >= 0.8 else ("MEDIUM" if score >= 0.5 else "LOW"),
            explanation=result.get("explanation", ""),
        )

    return geval_precision


# ---------------------------------------------------------------------------
# Evaluator 4: Label exact-match  (deterministic)
# ---------------------------------------------------------------------------

def label_match(output: dict[str, Any], input: dict[str, Any]) -> list[ExperimentEvaluation]:
    """Exact-match check for risk_level and recommendation against ground truth."""
    case_id = input.get("case_id", "")
    gt = _load_ground_truth(case_id)

    risk_ok = output.get("risk_level") == gt.get("expected_risk_level")
    rec_ok  = output.get("recommendation") == gt.get("expected_recommendation")

    return [
        ExperimentEvaluation(
            name="risk_level_match",
            score=1.0 if risk_ok else 0.0,
            label="PASS" if risk_ok else "FAIL",
            explanation=(
                f"'{output.get('risk_level')}' matches expected '{gt.get('expected_risk_level')}'."
                if risk_ok else
                f"'{output.get('risk_level')}' ≠ expected '{gt.get('expected_risk_level')}'."
            ),
        ),
        ExperimentEvaluation(
            name="recommendation_match",
            score=1.0 if rec_ok else 0.0,
            label="PASS" if rec_ok else "FAIL",
            explanation=(
                f"'{output.get('recommendation')}' matches expected '{gt.get('expected_recommendation')}'."
                if rec_ok else
                f"'{output.get('recommendation')}' ≠ expected '{gt.get('expected_recommendation')}'."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_evaluators(settings: Settings) -> list:
    """Return all evaluators ready to pass to run_experiment."""
    return [
        token_metrics,
        make_geval_recall(settings),
        make_geval_precision(settings),
        label_match,
    ]
