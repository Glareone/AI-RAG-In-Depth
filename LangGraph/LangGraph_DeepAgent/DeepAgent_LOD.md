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
