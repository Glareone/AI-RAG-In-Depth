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
