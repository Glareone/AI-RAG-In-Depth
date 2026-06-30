---
name: adr
description: Architecture Decision Records for this project — naming, format, status lifecycle, and when to create one.
---

# ADR Skill

Architecture Decision Records live in `adr/` at project root.

## Naming convention

`YYMMDD-short-kebab-topic.md` — e.g. `260630-MCP-server-agent.md`

## Frontmatter

```yaml
---
status: proposed | accepted | superseded | deprecated
date: YYYY-MM-DD
author: Name
supersedes: YYMMDD-old-adr.md   # only if replacing another ADR
---
```

## When to create an ADR

Create one before implementing anything where the choice will be hard to reverse or where a future reader would ask "why did they do it this way?":
- Choosing a library, transport, or protocol
- Defining a convention (naming, error handling, span attributes)
- Rejecting a reasonable alternative

## When to check ADRs first

Before starting any task in this project, scan `adr/` for existing decisions that affect your work. If an ADR covers the area, follow it — do not re-litigate unless you are explicitly asked to supersede it.

## Format

```markdown
---
status: accepted
date: YYYY-MM-DD
author: Name
---

# Title (imperative: "Use X for Y")

## Context
What problem or constraint prompted this decision.

## Decision
What was decided and the key reason.

## Consequences
Trade-offs accepted. What becomes easier, what becomes harder.
```
