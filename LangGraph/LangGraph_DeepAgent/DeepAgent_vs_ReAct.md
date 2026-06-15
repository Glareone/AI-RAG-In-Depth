# DeepAgent + Todo List vs ReAct

> Comparing ReAct, ReAct+todo-list, and full DeepAgent architectures.
> Specifically: is dynamic replanning required, or can static plans work?

---

## 1. The Landscape — What These Things Actually Are

### ReAct
The original pattern. Loop of: reason → act → observe → reason → act → ...
The LLM decides what to do next at each step based on the latest observation.
LangGraph's `create_react_agent` implements this loop.

```
loop until done:
  llm reasons about next action
  llm picks a tool, calls it
  observes result
  back to top
```

No plan. No persistent state beyond message history. Pure local decision-making.

### ReAct + Todo List (the middle ground)
Adds a todo list as a state object the LLM can read and write.
The LLM creates a todo list at start, then executes it loop-style,
updating the list when things change.

```
init: llm creates todo list from user inquiry
loop until todo list empty:
  llm picks next todo item
  reasons about how to execute it
  calls tool, observes result
  optionally updates todo list (add, remove, reorder)
  marks item complete or revises it
```

### DeepAgent (LangChain's full implementation)
The `deepagents` library built on LangGraph. Adds three middleware on top of ReAct:

```
TodoListMiddleware     → provides write_todos tool
                         LLM prompted to plan and update list during execution

FilesystemMiddleware   → provides ls/read_file/write_file/edit_file tools
                         offloads context to virtual files instead of stuffing
                         everything into the message history

SubAgentMiddleware     → provides task tool to spawn subagents
                         each subagent has isolated context, own prompts/tools
```

All three are middleware on a standard `create_agent` (ReAct) call.
Each is composable — you can add or remove independently.

**Key insight:** DeepAgent is NOT a different runtime from ReAct.
It IS ReAct, plus extra tools the LLM can use to manage its own state.

---

## 2. The Static vs Dynamic Distinction

This terminology comes from academic literature on Deep Research Agents.

```
Static workflow:
  predefined task pipeline
  sequential subtasks executed in order
  no adaptation to runtime feedback
  examples: cron jobs, Airflow DAGs, n8n linear flows

Dynamic workflow:
  LLM-based task planning
  agent reconfigures plan based on iterative feedback
  adaptive to evolving context
  examples: DeepAgent, ReAct, plan-then-execute with replanner
```

The critical threshold:

> A workflow with LLM calls becomes an agent when the LLM gains autonomous
> control over tool selection, execution order, termination conditions,
> and replanning capability.

Static workflows with LLM nodes are "intelligent workflows" — not agents.
True agents are dynamic by definition.

---

## 3. Can a DeepAgent Be Static?

Short answer: **mostly no, and where it can be, you shouldn't use DeepAgent.**

### Why static DeepAgent rarely makes sense

If the plan can be fully known in advance:
- you don't need an agent — you need a workflow
- pre-define the steps as a LangGraph state machine
- use LLM nodes only where ambiguity genuinely exists
- much cheaper, faster, more reliable

The whole point of DeepAgent's `write_todos` tool is to handle cases where:
- you don't know all the steps upfront
- intermediate results affect downstream steps
- the agent must adapt to unexpected data

Pre-defining the todo list defeats the purpose.

### Where partial static makes sense (the "Static-DRA" approach)

One academic paper (arxiv 2512.03887) proposes Static-DRA — a hierarchical
tree-based static Deep Research Agent. The structure is predefined,
but the user can configure two parameters (depth, breadth) to control research scope.

This works because:
- Research follows a known structural pattern (broad → narrow)
- User wants predictable control over scope and cost
- Dynamic exploration would produce inconsistent results across runs

But this is a niche use case. The structure being static is a feature
of the specific domain (deep research with controllable depth), not a general principle.

### What about purely deterministic todo lists?

If you can write a fixed checklist of tool calls that will always succeed
in the same order, you don't need an LLM at all — write the code directly.
The LLM is justified only when:
- next step depends on prior tool output content
- some steps may fail and need adaptation
- the plan structure depends on the specific user inquiry

---

## 4. The Spectrum — How Dynamic Should an Agent Be?

This is the real question. Not "static or dynamic" — but **how much dynamism, where.**

```
fully static                                              fully dynamic
─────────────────────────────────────────────────────────────────────
LangGraph DAG    plan-then-execute    DeepAgent+todos    pure ReAct
no LLM control   LLM plans once,      LLM plans, can     LLM decides
fixed flow       executes fixed       update todos on    everything per
                 plan to completion   the fly            step
```

Each position has tradeoffs:

```
position              cost      reliability   adaptability   debuggability
─────────────────────────────────────────────────────────────────────────
LangGraph DAG         lowest    highest       lowest         highest
plan-then-execute     low       high          medium         medium
DeepAgent+todos       medium    medium        high           low
pure ReAct            highest   lowest        highest        lowest
```

The right position depends on the task profile:
- known workflow, infrequent edge cases → LangGraph DAG with LLM-only-where-needed
- known objective, unknown execution path → plan-then-execute
- known objective, partially unknown structure → DeepAgent+todos
- open-ended exploration → pure ReAct

---

## 5. Hybrid Architecture — Dedicated Todo Update Node

A common production pattern: ReAct cycle + separate node to manage todo_lists
when necessary. This is a **superior architecture** to either pure pattern.

```
naive DeepAgent:
  LLM has write_todos tool always available
  LLM decides when to update todos
  → unpredictable update timing
  → expensive (full reasoning on every cycle to decide)

dedicated node pattern:
  ReAct cycle handles execution
  dedicated node decides "should todo list be updated now?"
  todo list updates are explicit, scoped events
  → predictable
  → cheaper (todo update node can use low reasoning effort)
  → debuggable (you can see exactly when and why the list changed)
```

This pattern aligns precisely with the two-knob principle from the LOD note:
- knob 1: evaluation frequency (when does the LLM check progress)
- knob 2: switch threshold (when does the plan actually change)

The dedicated todo-update node IS knob 2. Pure DeepAgent collapses both into one.

---

## 6. Static List + Static Control — Is It Achievable?

A common question: can we use a static list with static control?
The honest answer: not really in any useful agent sense.

Walking through the four combinations:

**List static + control static:**
- you have a deterministic script with LLM-flavored tool calls
- this is a LangGraph DAG, not an agent
- valid pattern, but it's not a DeepAgent — call it what it is

**List static + control dynamic:**
- plan-then-execute pattern with a fixed plan
- works for narrow domains where the plan structure is universal
- dynamism is only in HOW each step is executed, not WHICH steps run
- this is the Static-DRA approach

**List dynamic + control static:**
- LLM proposes plan changes but a hardcoded policy decides whether to accept
- rarely useful — you've added LLM cost without LLM benefit
- the policy is doing the work the LLM was supposed to do

**List dynamic + control dynamic:**
- DeepAgent
- maximum flexibility, maximum cost, hardest to debug
- the standard LangChain deepagents library

The dedicated-node pattern (dynamic list, semi-static control via dedicated node)
sits between options 2 and 4 — and is genuinely the best engineering tradeoff
for most production use cases.

---

## 7. When DeepAgent Earns Its Cost

DeepAgent with full dynamic planning is justified when:

```
✓ task structure cannot be predicted from user inquiry alone
✓ intermediate results legitimately invalidate the original plan
✓ task duration is long enough that replanning cost is amortized
✓ failure cost is high enough to justify adaptation overhead
✓ context offloading to filesystem is needed (long-running, lots of data)
✓ subagent delegation is needed (parallel work, isolated contexts)
```

DeepAgent is overkill when:

```
✗ workflow is known — use LangGraph DAG
✗ task is short (1-5 steps) — use ReAct
✗ steps are deterministic — write code, no LLM
✗ failure is cheap — let it fail and retry rather than building adaptation
```

Supporting both ReAct and DeepAgent archetypes in a platform is correct —
they serve different task profiles. The decision should be made per agent based
on the task profile, not as a default.

---

## 8. The Cost / Complexity Curve

```
task complexity →
   simple     medium      complex     very complex
─────────────────────────────────────────────────────
ReAct        ✓ optimal    overkill    fails           fails badly
ReAct+todos  overkill     ✓ optimal   acceptable      struggles
DeepAgent    massive      overkill    ✓ optimal       acceptable
             overkill
```

For "simple" tasks — bypass to a direct tool call or LangGraph DAG.
For "very complex" tasks — DeepAgent + subagents pattern.

---

## 9. Recommendation for Platform Design

For an agent platform supporting multiple archetypes:

**Keep both ReAct and DeepAgent. Document decision criteria clearly.**

```
use ReAct when:
  - 1-5 tool calls expected
  - linear workflow, low branching
  - prototyping or exploring a new use case

use DeepAgent when:
  - 10+ tool calls likely
  - branching is data-dependent
  - production workflow with adaptation requirements
  - need persistent state across long runs

use neither (use LangGraph DAG):
  - workflow is known and fixed
  - LLM only needed for specific decision points
  - cost/latency are primary concerns
```

The mistake to avoid: defaulting users to DeepAgent because it's "more powerful."
Most use cases are simple ReAct or even classical workflow territory.
DeepAgent should be a deliberate choice, not a default.

---

## 10. Open Implementation Questions

- Measure cost difference: same task on ReAct vs ReAct+todos vs DeepAgent
- What is the failure rate of pure ReAct on tasks >10 steps in production?
- Can the dedicated todo-update node be triggered by tool result classification
  (FULL REPLAN / PARTIAL ADJUST / NO ACTION from the LOD pattern)?
- How does subagent delegation interact with hierarchical goal stacks?
- Should the platform expose Static-DRA as a third archetype for research tasks?
- Empirical comparison: production reliability of DeepAgent vs ReAct+structured-todos

---

## References

- LangChain Deep Agents Middleware docs: https://docs.langchain.com/oss/python/deepagents/middleware
- DeepAgent paper (Xiaohongshu, WWW 2026): https://arxiv.org/pdf/2510.21618
- Static-DRA (hierarchical tree static research agent): https://arxiv.org/pdf/2512.03887
- Deep Research Agents systematic review: https://arxiv.org/pdf/2506.18096
- Dynamic Planning vs Static Workflows: https://tao-hpu.medium.com/dynamic-planning-vs-static-workflows-what-truly-defines-an-ai-agent-b13ca5a2d110
- Plan-then-Execute Resilient Agents: https://arxiv.org/pdf/2509.08646
- LLM Dynamic Planner (neurosymbolic): https://arxiv.org/pdf/2308.06391

---

Related: [DeepAgent_LOD.md](./DeepAgent_LOD.md) — Level of Detail architecture for reducing reasoning cost in DeepAgent systems.
