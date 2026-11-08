# SKILL-JIT

Just-in-time skill and tool loading: score every available skill against the task, load only what the task needs within a token budget. See [SPEC.md](SPEC.md).

Status: **M0–M2 done. M3 partly done:** the pool-size sweep and the agent-run harness are built, but the agent runs themselves and the `llm`/`d1` selectors have not run yet (no API keys; see M3 below). M4 not started.

## Install & run

```bash
pip install -e '.[test]'                      # add ,llm for the Claude backend
scripts/fetch_corpus.sh                        # pinned repos -> .corpus-src/, builds corpus.jsonl (376 skills)
python -m skilljit score corpus.jsonl --task "Fill in this PDF form" --backend lexical
python -m skilljit eval corpus.jsonl tasks/tasks.jsonl --backend lexical --s-min 0 --k 5
python -m skilljit label ~/.claude/projects/<proj>/<session>.jsonl --corpus corpus.jsonl  # transcript -> task row
python -m skilljit eval corpus.jsonl tasks/tasks.jsonl --backend embedding-bge --pool-size 10,50,200,0 --s-min 0 --k 5
python -m skilljit run corpus.jsonl tasks/tasks.jsonl -o runs.jsonl --dry-run   # episode count + token estimate
python -m skilljit run corpus.jsonl tasks/tasks.jsonl -o runs.jsonl             # needs ANTHROPIC_API_KEY; resumable
python -m skilljit report runs.jsonl
pytest -q
```

## Layout

| Path | What |
|---|---|
| `skilljit/ingest.py` | M0. Walks roots for `SKILL.md`, parses YAML frontmatter, validates against the [Agent Skills spec](https://agentskills.io/specification), records license (frontmatter `license`, else nearest `LICENSE*`/`COPYING*` up to the root) and token cost. Spec violations become `issues`, not exceptions. |
| `skilljit/backends.py` | M1. `Backend.score(task, skills) -> ScoreResult(scores 0–4, latency_s, input_tokens)`. Backends: `lexical` (BM25), `embedding` (pluggable embedder; default is an offline hashing embedder), `llm` (one Claude call, JSON-schema output), `d1` (Liquid AI d1 via AI/ML API, `score` questions batched ≤128 per request). |
| `skilljit/select.py` | Budgeted-recall knapsack: top-k by score if `k` is set, then greedy by score/tokens with a best-single-item fallback; `s_min` and an always-loaded core set. |
| `skilljit/evaluate.py` | M2. Loads the task suite (labels checked against the corpus) and computes recall@k, selected recall/precision, selected tokens, false loads on negatives, decoy rate, latency. |
| `skilljit/label.py` | M2. SPEC §6 ground truth: extracts invoked skills (`Skill` tool calls and `Read` of a `SKILL.md`) from Claude Code transcripts into task rows. |
| `skilljit/agent.py` | M3. Agent episodes: Claude gets a menu of skill cards and a `Skill` tool; with fallback on, `find_skills(need)` re-scores the pool (SPEC §5 recovery). Methods `all-cards`, `jit` and `jit-nofallback`, swept over per-task pool sizes. |
| `corpus/sources.txt` | 6 public skill repos pinned to commit SHAs, with licenses. |
| `corpus/decoys/` | 3 adversarial skills (self-promotion, keyword stuffing, prompt-structure injection), flagged `metadata.skilljit-decoy`. |
| `tasks/tasks.jsonl` | 49 hand-labeled tasks: 44 positives (2 multi-skill) and 5 negatives (`needed: []`). |

Credentials: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) for `llm` and `run`; `AIMLAPI_KEY` for `d1`. `embedding-bge` needs `pip install -e '.[embed]'` and downloads a ~65MB quantized ONNX model on first use.

## Task suite format

```json
{"id": "t21", "task": "...", "needed": [["paper-lookup", "literature-review"], "citation-management"], "label_source": "hand"}
```

Each `needed` entry is one requirement. An entry is either a skill name or a list of interchangeable names, and any one of them satisfies it. The pool has near-duplicates (`tdd` / `test-driven-development`, `grill-me` / `grilling`), and a single-name label would penalize a selector for picking the twin. `label_source` is `hand`, or `invoked` when the row comes from `skilljit label`.

## Baseline results (pool 376, budget 8000 tokens, `--s-min 0`)

| backend | k | recall@1 | recall@5 | selected recall | precision | selected tokens | false loads / negative | decoy rate | latency |
|---|---|---|---|---|---|---|---|---|---|
| lexical | 3 | 0.39 | 0.72 | 0.52 | 0.26 | 4.9k | 2.4 | 4% | 9 ms |
| lexical | 5 | 0.39 | 0.72 | 0.56 | 0.21 | 5.9k | 3.2 | 4% | 9 ms |
| embedding (hashing) | 3 | 0.28 | 0.36 | 0.35 | 0.16 | 5.4k | 2.4 | 2% | 43 ms |

For reference, per task, loading every skill body costs **1.13M tokens**, and frontmatter-only progressive disclosure costs **35k**.

The numbers worth reading:
- The selector gives up recall to stay inside the budget. Selected recall (0.56) sits below recall@5 (0.72) because large needed skills don't fit in 8k tokens.
- No offline backend can say "nothing is needed". On the negatives (trivia, a haiku) it still loads 2–3 skills.
- Keyword stuffing works on lexical. With `--s-min 0` and no `k` cap, `keyword-magnet` gets selected in 51% of tasks (any decoy: 59%). A `k` cap or `--s-min 2` brings that down to 2–10%, but only because the decoy scores lower than genuinely related skills, not because lexical recognizes it as a decoy.

## M3: pool-size stress test (selection level, offline backends)

Each task gets its own pool: every skill named in its labels (near-duplicate twins included), all 3 decoys, and seeded random distractors up to the target size. Run with `--k 5 --s-min 0 --budget 8000`, 49 tasks.

| backend | pool | recall@1 | recall@5 | selected recall | false loads/neg | decoy rate | load-all tokens | latency |
|---|---|---|---|---|---|---|---|---|
| lexical | 10 | 0.81 | 1.00 | 0.90 | 4.8 | 94% | 23k | 0.2 ms |
| lexical | 50 | 0.74 | 0.93 | 0.80 | 4.8 | 45% | 144k | 1.3 ms |
| lexical | 200 | 0.45 | 0.84 | 0.64 | 3.8 | 12% | 595k | 5.8 ms |
| lexical | 376 | 0.39 | 0.72 | 0.56 | 3.2 | 4% | 1.13M | 9.8 ms |
| embedding-bge | 10 | 0.78 | 1.00 | 0.83 | 4.4 | 100% | 23k | 61 ms |
| embedding-bge | 50 | 0.62 | 0.90 | 0.69 | 3.6 | 67% | 144k | 62 ms |
| embedding-bge | 200 | 0.51 | 0.80 | 0.53 | 3.0 | 39% | 595k | 63 ms |
| embedding-bge | 376 | 0.42 | 0.67 | 0.50 | 3.2 | 24% | 1.13M | 56 ms |
| embedding (hashing) | 376 | 0.28 | 0.36 | 0.30 | 3.6 | 2% | 1.13M | 46 ms |

How to read it:
- **Every selector degrades as the pool grows.** From 10 to 376 skills, recall@5 falls from 1.00 to 0.72 (lexical) and from 1.00 to 0.67 (bge). This is the SPEC §7 curve, measured for the selectors. Whether load-all degrades the *agent* needs the agent runs below.
- **The real embedding model doesn't beat BM25 on this suite.** bge's recall@1 is slightly higher at 376 (0.42 vs 0.39), but its recall@5 and selected recall are lower. One possible cause is the suite itself: I wrote the labels after reading the descriptions (see Known gaps). Labels from logged invocations would show whether it is.
- **Selection recall stays below recall@5.** The 8k budget drops large needed skills: the cost of the budget, paid in recall.
- **Neither offline selector can answer "nothing needed".** Both load 3–5 skills per negative task at every pool size.
- **Decoy rates at pools of 10 and 50 are an artifact of the setup.** All 3 decoys are always in the pool, so at pool 10 they make up 30% of it, and k=5 selects half the pool. Read the decoy rate at 200 and 376 only. At 376, lexical picks a decoy in 4% of tasks and bge in 24%. For bge it's always `keyword-magnet` (12 of 49 tasks); `universal-assistant` (self-promotion) and `injection-probe` are never selected. Similarity selectors fall for keyword stuffing, not persuasion. Persuasion is a threat only to the `llm`/`d1` scorers, which read the text.
- **Latency:** BM25 grows with the pool (0.2 → 10 ms). bge stays flat at ~60 ms, dominated by embedding the query, because card embeddings are precomputed.

## M3: agent runs (built, not yet run)

`skilljit run` plays out each (method, pool size, task) episode with Claude. The system prompt holds the menu of skill cards, the `Skill` tool returns a skill's SKILL.md, and with fallback on, `find_skills(need)` re-scores the rest of the pool against the task plus the stated need, then returns the top 3 cards as a tool result. History is append-only and the system prompt is cached. Episodes are written one JSONL row each, so a stopped run resumes where it left off.

- **What it measures:** skill-use success (all needed skills loaded), recall/precision of loaded skills, false loads on negatives, decoy loads, context and output tokens, `find_skills` calls, and episodes still unfinished at the turn limit.
- **What it does not measure:** end-task success. The agent can't execute anything, and the tasks have no fixtures or graders.
- **Cost (dry run, default grid of 4 pool sizes × 3 methods × 49 tasks = 588 episodes):** ~13.7M context tokens. The `all-cards` method at the full pool accounts for 5.9M of that, because it resends a ~35k-token menu on every call. Caching cuts the input cost on calls 2 and later, but thinking output adds to the bill. Run a `--limit 5` pilot and read the real token counts before the full grid.
- **Blocked:** no Anthropic API key is available to scripts in this environment, so no episodes have run. No made-up or simulated results stand in for them.

## Format conventions checked (2026-11)

Surveyed 6 repos (373 skills):

- Only `name`, `description` and `license` appear in practice; `metadata`/`compatibility`/`allowed-tools` are rare.
- Spec violations exist even in `anthropics/skills`: one description > 1024 chars and one `name` ≠ directory. Separately, one `mattpocock/skills` file isn't valid YAML at all: its description has an unquoted `: `.
- License lives either in frontmatter (`license: Complete terms in LICENSE.txt`) with a per-skill file, or only in a repo-root `LICENSE`. The ingester handles both.
- The SPEC's architecture diagram mentions "triggers"; no such field exists in the skill spec. Trigger phrasing lives inside `description`.
- No skill name appears in more than one of the 6 repos, so labels use bare names.

## Known gaps

- **Labels are hand-written, not from load-all runs.** SPEC §6 asks for labels from logged invocations, validated by ablation. `skilljit label` implements the extraction (checked on a real Claude Code transcript), but no load-all runs exist yet. Hand labels carry their own bias: I wrote them after reading the descriptions, so keyword overlap may favor `lexical`. I paraphrased tasks away from description wording to reduce that, but haven't measured how much it helped.
- The suite leans toward science, security and developer tooling, because that's what the pinned repos contain.
- Lexical and embedding scores are **not calibrated** (lexical is max-normalized; bge gives even unrelated text a cosine of ~0.4), so `s_min` only means something for `llm`/`d1`.
- `llm` and `d1` have only been tested against fakes. Neither has been called live: no keys in the dev environment.
- Token counts are a chars/4 estimate.
- Not run yet: agent episodes and the `llm`/`d1` selectors (both need keys). Not built: a load-all-*bodies* method (at pool ≥200 the bodies exceed a 1M context window anyway), end-task grading, and the harness plugin (M4).
