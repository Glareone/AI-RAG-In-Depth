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
