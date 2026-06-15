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
FilesystemMiddleware   → provides ls/read_file/write_file/edit_file tools
SubAgentMiddleware     → provides task tool to spawn subagents
```

All three are middleware on a standard `create_agent` (ReAct) call.
Each is composable — you can add or remove independently.

**Key insight:** DeepAgent is NOT a different runtime from ReAct.
It IS ReAct, plus extra tools the LLM can use to manage its own state.

---

## 2. The Static vs Dynamic Distinction

```
Static workflow:
  predefined task pipeline, sequential subtasks, no adaptation
  examples: cron jobs, Airflow DAGs, n8n linear flows

Dynamic workflow:
  LLM-based task planning, reconfigures based on iterative feedback
  examples: DeepAgent, ReAct, plan-then-execute with replanner
```

> A workflow with LLM calls becomes an agent when the LLM gains autonomous
> control over tool selection, execution order, termination conditions,
> and replanning capability.

Static workflows with LLM nodes are "intelligent workflows" — not agents.
True agents are dynamic by definition.

---

## 3. Can a DeepAgent Be Static?

Short answer: **mostly no, and where it can be, you shouldn't use DeepAgent.**

If the plan can be fully known in advance — use a LangGraph DAG.
Pre-defining the todo list defeats the purpose of DeepAgent.

One niche exception: Static-DRA (arxiv 2512.03887) uses static structure
for deep research with user-configurable depth/breadth parameters.
Domain-specific, not a general principle.

---

## 4. The Spectrum

```
fully static                                              fully dynamic
─────────────────────────────────────────────────────────────────────
LangGraph DAG    plan-then-execute    DeepAgent+todos    pure ReAct

position              cost      reliability   adaptability   debuggability
─────────────────────────────────────────────────────────────────────────
LangGraph DAG         lowest    highest       lowest         highest
plan-then-execute     low       high          medium         medium
DeepAgent+todos       medium    medium        high           low
pure ReAct            highest   lowest        highest        lowest
```

---

## 5. Hybrid — Dedicated Todo Update Node

```
naive DeepAgent:
  LLM has write_todos tool always available
  → unpredictable update timing, expensive

dedicated node pattern:
  ReAct cycle handles execution
  dedicated node decides "should todo list be updated now?"
  → predictable, cheaper, debuggable
```

The dedicated todo-update node separates evaluation frequency from switch threshold.
Pure DeepAgent collapses both into one expensive decision per cycle.

---

## 6. Static List + Static Control — The Four Combinations

```
list static + control static   → LangGraph DAG (not an agent)
list static + control dynamic  → plan-then-execute (Static-DRA niche)
list dynamic + control static  → LLM cost without LLM benefit (avoid)
list dynamic + control dynamic → DeepAgent (full library)
```

Best production tradeoff: dynamic list + semi-static control via dedicated node.

---

## 7. When DeepAgent Earns Its Cost

```
justified:
✓ task structure cannot be predicted from user inquiry
✓ intermediate results legitimately invalidate the plan
✓ long-running tasks where replanning cost amortizes
✓ context offloading needed (filesystem middleware)
✓ subagent delegation needed

overkill:
✗ workflow is known → LangGraph DAG
✗ task is short (1-5 steps) → ReAct
✗ steps are deterministic → write code
✗ failure is cheap → retry, don't adapt
```

---

## 8. Hard Rails — When Agents Are The Wrong Tool

Wanting "an agent that follows hard rails" is wanting an LLM that behaves like code.
If it behaves like code — write code.

The fundamental incompatibility:

```
LLM autonomy                    Deterministic guarantees
────────────────────────────────────────────────────────
adapts to novelty               predictable cost
handles ambiguity               predictable latency
judgment under uncertainty      testable, debuggable
emergent behavior               hard rails enforce flow
```

You cannot have both in the same component.

### The Tool Spectrum

```
strongest rails ←────────────────────────────────→ weakest rails

state machine   n8n / Zapier   LangGraph DAG   LangGraph+LLM   ReAct   DeepAgent
hand-coded      visual flow    explicit graph  LLM at nodes    loop    full dynamic
no LLM          no/scoped LLM  LLM at          LLM at most     LLM     LLM owns
                               decision pts    decision pts    loop    everything
```

### n8n — When It's Genuinely the Right Answer

```
✓ workflow structure is known and stable
✓ steps are mostly integrations (API calls, data transforms)
✓ business stakeholders need to read/modify the flow
✓ failure modes are bounded (retry, skip, alert)
✓ cost predictability matters
```

n8n with scoped LLM nodes handles AI-heavy workflows correctly:
LLM handles ambiguity, n8n handles integration deterministically.

### The Three-Way Choice

```
1. LangGraph DAG with LLM decision nodes
   → known workflow with judgment-needed branch points

2. n8n with LLM nodes
   → business workflows, integration-heavy, non-engineer-editable

3. DeepAgent / ReAct
   → genuinely open-ended tasks, unknown structure
```

Most teams pick 3 because it's easiest to demo. Then they hit the wall.

### The Key Question

> What is the smallest amount of LLM autonomy I actually need?

```
none       → write code or use n8n
decision   → LangGraph DAG with LLM at specific nodes
adaptive   → plan-then-execute with replan node
open-ended → DeepAgent
```

### Platform Implication

"Use neither (use LangGraph DAG)" is not a footnote —
it's the third archetype the platform should probably support natively.

Users who want rails will build brittle workarounds if you don't give them the right tool.
Better to provide the right tool than have users fight the wrong one.

---

## 9. Cost / Complexity Curve

```
task complexity →
   simple     medium      complex     very complex
─────────────────────────────────────────────────────
ReAct        ✓ optimal    overkill    fails           fails badly
ReAct+todos  overkill     ✓ optimal   acceptable      struggles
DeepAgent    overkill     overkill    ✓ optimal       acceptable
```

---

## References

- LangChain Deep Agents Middleware: https://docs.langchain.com/oss/python/deepagents/middleware
- DeepAgent paper (Xiaohongshu, WWW 2026): https://arxiv.org/pdf/2510.21618
- Static-DRA: https://arxiv.org/pdf/2512.03887
- Deep Research Agents review: https://arxiv.org/pdf/2506.18096
- Dynamic Planning vs Static Workflows: https://tao-hpu.medium.com/dynamic-planning-vs-static-workflows-what-truly-defines-an-ai-agent-b13ca5a2d110
- Plan-then-Execute Resilient Agents: https://arxiv.org/pdf/2509.08646

Related: [DeepAgent_LOD.md](./DeepAgent_LOD.md)
