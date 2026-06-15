# DeepAgent — LOD (Level of Detail) Architecture

> Applying AI LOD principles from game AI to LangGraph DeepAgent implementation.
> A pattern for making agents cheaper, more coherent, and more reliable on top of a hierarchical todo list.

---

## 1. What LOD Means

LOD (Level of Detail) is borrowed from game AI, where it solves a specific scaling problem.

In an RTS with hundreds of units, full per-tick AI updates for every unit are wasteful. Most units are not doing anything interesting most of the time, so they don't need full attention every tick.

```
unit far from any action       → update every 20 ticks
unit in active zone            → update every 2 ticks
unit currently in combat       → update every 1 tick
```

Units are distributed across K update buckets. Each tick processes only one bucket. Total per-tick work = N/K instead of N. Players never notice — a 100ms delay in a distant unit reacting is imperceptible.

The principle: match cognitive cost to the rate of meaningful change.

This is the foundation of what follows. The same principle applies to LLM agents — but the resource being conserved is reasoning effort instead of CPU time.

---

## 2. Why DeepAgent + Todo List Needs LOD

DeepAgent with a todo list combines two control models that conflict:

```
ReAct:      fully dynamic — each step decided in the moment
            no commitment to a plan

Todo list:  fully static — committed plan, execute it
            no adaptation to new information
```

Naive implementations run the LLM at full reasoning effort on every iteration of the ReAct loop. Three failure modes follow:

```
1. Flip-flopping:    reasons about replanning on every tool result
                     never commits long enough to finish anything

2. Token waste:      full reasoning on routine "what tool to call next" decisions
                     where the answer is deterministic given the current task

3. Cost explosion:   10-step workflow = 10 reasoning-heavy LLM calls
                     scales poorly with workflow length
```

LOD solves all three by making reasoning effort contextual rather than uniform.

---

## 3. The Two-Trigger System

LOD alone is insufficient. Pure timer-based updates miss discrete state changes. Pure event-driven updates miss gradual drift. The complete pattern combines both.

```
Timer-based update (LOD bucket assignment):
  catches gradual state drift
  cost: predictable, constant
  misses: sudden changes between ticks

Event-based invalidation:
  catches discrete state changes immediately
  cost: unpredictable, spiky
  misses: gradual drift between events

Combined:
  timer catches drift, events catch shocks
```

### Event Taxonomy

Not all tool results deserve full re-evaluation:

```
FULL REPLAN (strategic picture changed)
  tool result contradicts plan assumptions
  required data unavailable, fundamental error
  → trigger full planning re-evaluation

PARTIAL ADJUST (tactical picture changed)
  result is unexpected but interpretable within current plan
  → re-evaluate current task scoring only

NO ACTION (execution-level noise)
  result matches expected branch
  routine success, expected data shape
  → input to current decision, not trigger to change it
```

Classification boundary: does this result change what I should be doing, or how I execute what I'm already doing?

Most tool results fall into NO ACTION. That's where the LOD dividend comes from.

---

## 4. The Three-Node Pattern in LangGraph

The architecture maps onto three nodes that wrap every tool call:

```
pre_execution_node → [tool call] → re_evaluation_node → post_processing_node
```

### pre_execution_node
Runs before every tool call. Lightweight by default.

```
input:   current goal stack, agent state, last tool result
output:  { tool, params, reasoning_enabled: bool }

reasoning_effort = low or none for routine steps
reasoning_effort = medium when tool selection is genuinely ambiguous
```

Responsibilities:
- load current goal stack state
- confirm current_task is still valid
- select tool and construct parameters
- decide if reasoning is needed for this specific call

### re_evaluation_node
The commitment threshold gate. This is where flip-flopping is prevented.

```
input:   tool result, current_task, success_condition, plan_investment
output:  { decision: continue | retry | replan | escalate }

reasoning_effort = low for clean results (binary condition check)
reasoning_effort = medium for anomalies (delta exceeds threshold)
```

Decision logic:

```
result satisfies success_condition          → continue, pop task
result fails, retries remaining             → retry
result fails, no retries                    → escalate
result unexpected, delta < threshold        → continue, log anomaly
result unexpected, delta > threshold        → replan remaining stack
plan deep + result ambiguous                → higher threshold, prefer continue
plan early + result ambiguous               → lower threshold, prefer replan
```

This node runs on every tool result. But it is cheap when results are clean — just a condition check. Expensive reasoning only triggers on genuine delta-exceeds-threshold cases.

### post_processing_node
Runs after re_evaluation confirms continuation. Routes to next action.

```
input:   re_evaluation decision, tool result, goal stack state
output:  { next_action, payload }

reasoning_effort = low for routing (deterministic based on re_evaluation)
reasoning_effort = medium for final output composition or replan handoff
```

Key property: context is never discarded on replan.

Prior tool results, completed steps, and confirmed facts are preserved in agent state and passed back to the planning layer as constraints on the new plan. This prevents the common failure of agents that replan from scratch and repeat already-completed work.

---

## 5. The Two Independent Knobs

A critical refinement: evaluation frequency and commitment threshold are separate concerns.

```
naive design (one knob):
  "rescore every N seconds"
  conflates "when do I think" with "when do I act"
  → either flip-flops (low N) or reacts too slowly (high N)

correct design (two knobs):
  knob 1: evaluation frequency  (when do I check)
  knob 2: switch threshold      (when do I act on what I see)

  rescore can happen frequently for accuracy
  but action only switches when delta > threshold

  score(current_action) = 0.7  → currently doing this
  score(alternative)    = 0.71 → don't switch, delta too small
  score(alternative)    = 0.85 → switch, delta significant
```

Decoupling gives:
- LOD applied to evaluation (cheap, frequent for active contexts)
- Commitment applied to action (sticky, prevents thrashing)

Same input data, two independent policies derived from it.

---

## 6. The Commitment Threshold — Variable, Not Fixed

The threshold for replanning should scale with context:

```
threshold = base_δ × f(plan_investment, reversibility, context_stability)
```

```
situation                              threshold adjustment
──────────────────────────────────────────────────────────────────
early in plan (few steps completed)    lower  — cheap to replan
deep in plan (many steps completed)    higher — high replan cost
reversible next action                 lower  — safe to try
irreversible next action (send email)  much higher — require confidence
stable context (data as expected)      higher — stay the course
unstable context (repeated failures)   lower  — current plan is likely wrong
```

An agent that received 3 consecutive tool failures should replan more readily than one executing smoothly — the failures are evidence the current plan is wrong.

---

## 7. Reasoning Effort as the Cost Lever

Modern providers (Anthropic, OpenAI, Google) expose reasoning effort as a per-call parameter:

```python
response = client.chat(
    model="...",
    reasoning_effort="low" | "medium" | "high",
    messages=[...]
)
```

The LOD mapping:

```
context                       reasoning_effort     game AI equivalent
──────────────────────────────────────────────────────────────────────
routine execution step        low or none          distant unit, slow tick
re_evaluation (clean)         low                  background unit check
re_evaluation (anomaly)       medium               in-zone unit, medium tick
escalation decision           medium-high          combat unit, fast tick
planning / replan             high                 full AI rescore, event
```

The provider manages inference internally. You set effort per node. No custom scheduler needed at the application layer.

---

## 8. The LOD Dividend — Quantified

Comparing naive ReAct against LOD-aware architecture on a 10-step task:

```
naive ReAct:
  every step → full reasoning call
  total: 10 high-effort reasoning calls
  cost:  10x baseline

LOD-aware (this architecture):
  planning:        1 high-effort call
  routine steps:   7 low-effort calls (or none)
  anomalies:       1-2 medium-effort calls
  total:           ~3-4 effective high-cost equivalents
  cost:            ~30-40% of naive

graceful degradation under tight budget:
  reasoning preserved for: planning, irreversible actions
  reasoning sacrificed:    routine steps, clean checks
  result: degraded coherence, not catastrophic failure
```

The practical argument for LOD is not "more elegant" — it is costs 3x less to run with better coherence properties.

---

## 9. Full Architecture

```
USER INQUIRY
      ↓
PLANNING LAYER  (high reasoning effort, runs once or on replan)
  LLM decomposes inquiry into hierarchical goal stack
  outputs structured plan:
    { goal, sub_goals[], current_task, success_conditions, termination_conditions }
      ↓
┌──────────────────────────────────────────────────────────────┐
│  EXECUTION LOOP (per task)                                   │
│                                                              │
│  pre_execution_node           (low reasoning default)        │
│    load state, select tool, construct params                 │
│         ↓                                                    │
│  [tool call: MCP / skill / user prompt]                      │
│         ↓                                                    │
│  re_evaluation_node           (low reasoning default)        │
│    check result against success_condition                    │
│    compute delta, apply commitment threshold                 │
│    → continue / retry / replan / escalate                    │
│         ↓                                                    │
│  post_processing_node         (low reasoning default)        │
│    update agent state                                        │
│    route to next action                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
      ↓
OUTPUT LAYER
  stack empty + all success met  → finalize
  stack blocked                  → explain + ask user
  replan triggered               → planning layer (context preserved)
```

---

## 10. The Test That Reveals Everything

To diagnose whether an existing DeepAgent implementation has these properties:

Give the agent a task requiring 5+ sequential tool calls where step 3 returns unexpected data.

```
expected behavior:
  adapt plan, handle unexpected data, continue coherently

failure modes and what they reveal:
  ignores unexpected data → produces wrong output
    → no event invalidation; everything treated as NO ACTION

  replans from scratch → loses prior context
    → no commitment threshold; every event triggers FULL REPLAN

  loops on step 3 forever
    → no termination condition; success/failure not explicit

  asks user about every minor uncertainty
    → no escalation threshold; partial uncertainty treated as blocking
```

Each failure mode maps to exactly one missing architectural component. The test is a diagnostic, not just a pass/fail check.

---

## 11. Open Implementation Questions

- Should `re_evaluation_node` be a separate LangGraph node or a conditional edge?
- How do you represent `plan_investment` as a numeric signal for threshold calculation?
- What is the right serialization format for goal stack handoff to planning layer on replan?
- Can `pre_execution_node` cache tool parameter construction for repeated identical tasks?
- How does this three-node pattern interact with streaming output to the user?
- Empirical test: same goal stack, same tool results, different effort allocations — measure output quality vs. token cost across runs

---

## References

Derived from analysis of:
- Pod-Bot architecture (Counter-Strike) — layered AI separation of concerns
- StarCraft / Dota bot architectures — hierarchical goal decomposition
- Halo 2 AI (Damian Isla, GDC 2005) — Behavior Tree popularization
- Prospect Theory (Kahneman & Tversky, 1979) — commitment threshold validation
- OpenAI Five — strategic layer as ML, execution as classical
