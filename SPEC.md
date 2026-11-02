# SKILL-JIT — just-in-time skill and tool loading

## 1. Thesis

Loading every skill, tool and MCP description into context on agent start degrades performance and wastes tokens. SKILL-JIT scores every available skill against the current task with a decision model in one batched pass (milliseconds), loads only the top-k plus a small core, and re-scores when the agent fails or the task shifts.

Analogy: a surgeon's instrument tray. The nurse doesn't wheel in every instrument in the hospital; they lay out what this operation needs and fetch more on request.

## 2. Why now

- LangChain lists "harnesses that dynamically assemble the right tools and context just-in-time" as an open problem, and describes skills' progressive disclosure as the current partial fix against context rot (https://www.langchain.com/blog/the-anatomy-of-an-agent-harness).
- Skill ecosystems are exploding (large public skill repos, skills registries), so skill counts per user are climbing.
- Decision models can score dozens of skills in a single batched call with no generated tokens.

## 3. Prior art & differentiation

Progressive disclosure loads only skill frontmatter; embedding retrieval picks by similarity; LLM-based selection is slow. SKILL-JIT uses a calibrated relevance score and measures the actual tradeoff: task success vs input tokens vs recall of needed skills.

## 4. Scope

- MVP: offline selector + evaluation harness over a corpus of 100+ public skills.
- v1: plugin for one harness (Claude Code skills directory management or a deepagents middleware).
- Non-goals: writing skills; skill security (see skillcheck idea / REDLOOP).

## 5. Architecture

```
task text + repo signals ─▶ candidate skills (frontmatter: name, description, triggers)
                                   ▼
          DecisionBackend: per skill → score(0–4) "needed for this task?"
                                   ▼
          select top-k with score ≥ s_min  ∪  core set
                                   ▼
          harness loads selected skills ─▶ agent runs
                                   ▼
          on tool-not-found / repeated failure ─▶ re-score with error context
```

## 6. Core method

- Selection as budgeted recall: maximize expected recall of needed skills subject to a token budget B: choose S maximizing Σ_{s∈S} p_s subject to Σ_{s∈S} t_s ≤ B (knapsack; greedy by p_s/t_s is fine at this scale).
- Ground truth "needed skills" per task: run the agent with all skills loaded and log which skills were actually invoked; validate by ablation on a subset.

## 7. Evaluation

- Compare: load-all, frontmatter-only progressive disclosure, embedding top-k, LLM selector, SKILL-JIT.
- Metrics: task success, input tokens per task, selection latency, recall@k of needed skills, failure-recovery rate after re-scoring.
- Stress test: grow the skill pool 10 → 50 → 200 and plot success vs pool size per method. The interesting result is where load-all starts degrading.

## 8. Milestones

- M0 Skill corpus ingester (frontmatter parser) with licenses recorded.
- M1 Scoring backends (d1, embeddings, LLM). Acceptance: identical interface, latency reported.
- M2 Task suite with ground-truth needed skills.
- M3 Agent runs across methods and pool sizes; report.
- M4 Harness plugin.

## 9. Risks & honest weaknesses

- "Needed skill" labels derived from load-all runs are biased toward what the model happened to use.
- Skill descriptions vary wildly in quality; selection quality may track description quality more than the selector.
- Harness-specific plugin APIs change frequently.

## 10. Verify before building

Current skill frontmatter conventions; how your target harness loads skills at runtime and whether it can unload them.

## 11. Claude Code kickoff

"Read SPEC.md. Implement M0 and M1 with tests. Before writing the ingester, check the current skill file format conventions in public skill repos."
