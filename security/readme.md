# OWASP Top 10 for Agentic Applications 2026

This folder contains the official **OWASP Top 10 for Agentic Applications 2026** PDF, published by the OWASP GenAI Security Project — Agentic Security Initiative (December 2025).

📄 [OWASP-Top-10-for-Agentic-Applications-2026-12.6-1.pdf](./OWASP-Top-10-for-Agentic-Applications-2026-12.6-1.pdf)

Licensed under **Creative Commons CC BY-SA 4.0**. Source: [genai.owasp.org](https://genai.owasp.org)

---

## What This Document Is

- **Focus:** security risks specific to autonomous AI agents (not traditional LLMs)
- **Scope:** agents that plan, decide, and act across multiple steps, tools, and systems on behalf of users
- **Difference from OWASP LLM Top 10:** LLM Top 10 covers single-response model risks; this covers risks from autonomy, delegation, and multi-step execution
- **Format:** each of the 10 entries follows: Description → Common Examples → Attack Scenarios → Mitigations → References
- **Key concept introduced:** **Least-Agency** — only grant agents the minimum autonomy required for the task

---

## Table of Contents — Summarized

**Letter from The Agentic Top 10 Leaders** (p.6)
Introduction from the project leads explaining the document's purpose: a concise, actionable companion to OWASP's deeper Agentic AI Threats and Mitigations guide.

**Agentic Top 10 At A Glance** (p.8)
Visual diagram mapping all 10 risks onto the agentic system lifecycle: inputs (prompts, APIs, external agents) → integration/processing (policy, memory, tool use) → outputs (tools, APIs, external agents).

---

### ASI01: Agent Goal Hijack (p.9)
Attackers manipulate an agent's objectives or decision path through prompt injection, poisoned documents, forged agent messages, or malicious external data — redirecting the agent toward unintended actions. Example: a hidden instruction in an email silently triggers data exfiltration (EchoLeak).

### ASI02: Tool Misuse and Exploitation (p.12)
The agent uses a *legitimate* tool in an unsafe or unintended way — over-invoking APIs, deleting data, or chaining tools to exfiltrate information — without necessarily being hijacked. Example: an email summarizer tool that also has delete/send permissions it shouldn't.

### ASI03: Identity and Privilege Abuse (p.15)
Exploits the gap between agent identity and traditional user-centric access control. Covers privilege inheritance, cached credentials, agent-to-agent trust abuse (confused deputy), and forged agent personas. Example: a low-privilege agent relays a request to a high-privilege agent, which executes it without re-checking original intent.

### ASI04: Agentic Supply Chain Vulnerabilities (p.18)
Risks from third-party tools, models, plug-ins, MCP servers, or other agents that are compromised, malicious, or tampered with — loaded dynamically at runtime rather than fixed at build time. Example: a malicious MCP server impersonating a legitimate one (postmark-mcp) secretly BCCs emails to an attacker.

### ASI05: Unexpected Code Execution (RCE) (p.21)
Agents that generate and execute code (including "vibe coding" tools) can be manipulated into running attacker-defined or hallucinated malicious code — via prompt injection, unsafe deserialization, or chained tool calls. Example: an agent processes a file path containing a hidden shell command that deletes production data.

### ASI06: Memory & Context Poisoning (p.24)
Adversaries corrupt an agent's stored or retrievable context (RAG stores, embeddings, conversation memory) so that future reasoning becomes biased or unsafe — persisting across sessions, unlike a one-time prompt injection. Example: an attacker repeatedly reinforces a fake price in a travel-booking assistant's memory until it's treated as fact.

### ASI07: Insecure Inter-Agent Communication (p.27)
Multi-agent systems communicate via APIs, message buses, and shared memory — weak authentication or integrity controls let attackers intercept, spoof, or tamper with these real-time exchanges. Example: a man-in-the-middle attacker injects hidden instructions into unencrypted agent-to-agent traffic.

### ASI08: Cascading Failures (p.30)
A single fault (hallucination, poisoned input, corrupted tool) propagates and amplifies across interconnected agents, turning one error into system-wide harm — because agents act autonomously and persist state without stepwise human checks. Example: a poisoned market-analysis agent inflates risk limits, causing downstream trading agents to auto-execute oversized positions.

### ASI09: Human-Agent Trust Exploitation (p.33)
Agents build strong trust with humans through fluent language and apparent expertise (anthropomorphism); attackers or misaligned agents exploit this trust to get harmful actions approved without proper scrutiny. Example: a finance copilot, fed a poisoned invoice, confidently recommends an urgent payment that a manager approves without independent checks.

### ASI10: Rogue Agents (p.36)
An agent deviates from its intended scope and acts harmfully, deceptively, or persistently — distinct from being hijacked, this is about the **loss of behavioral integrity** once drift begins, regardless of how it started. Example: an agent that learned a bad behavior from a (now-removed) poisoned instruction continues exfiltrating data independently.

---

### Appendix A — OWASP Agentic AI Security Mapping Matrix (p.39)
Cross-reference table linking each ASI risk to the corresponding OWASP LLM Top 10 (2025) entries, the detailed Agentic AI Threats & Mitigations T-codes, and AIVSS scoring categories.

### Appendix B — Relationship to OWASP CycloneDX and AIBOM (p.41)
Explains how this document complements CycloneDX/AIBOM (which answers "what components are in my AI system") by addressing "how can those components and agents behave or fail unsafely."

### Appendix C — Mapping Between OWASP Non-Human Identities Top 10 (2025) and Agentic AI Top 10 (p.42)
Table mapping all 10 NHI (Non-Human Identity) risks to the corresponding ASI entries — useful for teams already using the NHI framework for service-account and machine-identity security.

### Appendix D — ASI Agentic Exploits & Incidents Tracker (p.44)
A running, weekly-updated table of real-world agentic AI exploits and incidents (Feb 2025 – Oct 2025), each mapped to the relevant ASI risk category. Includes EchoLeak, Replit's production-database deletion, malicious MCP packages, Cursor RCE vulnerabilities, and more.

### Appendix E — Abbreviations (p.50)
Glossary of acronyms used throughout the document (A2A, MCP, RAG, SBOM, NHI, PEP/PDP, etc.).

### Acknowledgements (p.52)
Lists the Top 10 leaders, entry leads, contributors, and the expert review board (including representatives from NIST, Microsoft, the Alan Turing Institute, and others) who developed this document.

### OWASP GenAI Security Project Sponsors / Project Supporters (p.55–56)
Organizations funding and supporting the OWASP GenAI Security Project.

---

## Why This Matters for Agent Platform Work

Several entries map directly onto architectural decisions discussed elsewhere in this repo — particularly **ASI01 (Goal Hijack)**, **ASI02 (Tool Misuse)**, and **ASI08 (Cascading Failures)**, which are exactly the failure modes that goal-stack discipline, commitment thresholds, and scoped tool permissions (see [`LangGraph/LangGraph_DeepAgent`](../LangGraph/LangGraph_DeepAgent)) are designed to prevent.
