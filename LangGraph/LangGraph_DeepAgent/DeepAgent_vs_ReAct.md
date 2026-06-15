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
