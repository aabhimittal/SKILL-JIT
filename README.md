# SKILL-JIT

Just-in-time skill and tool loading: score every available skill against the task, load only what the task needs within a token budget. See [SPEC.md](SPEC.md).

Status: **M0 (ingester), M1 (scoring backends) and M2 (task suite + offline eval) done.** M3–M4 not started.

## Install & run

```bash
pip install -e '.[test]'                      # add ,llm for the Claude backend
scripts/fetch_corpus.sh                        # pinned repos -> .corpus-src/, builds corpus.jsonl (376 skills)
python -m skilljit score corpus.jsonl --task "Fill in this PDF form" --backend lexical
python -m skilljit eval corpus.jsonl tasks/tasks.jsonl --backend lexical --s-min 0 --k 5
python -m skilljit label ~/.claude/projects/<proj>/<session>.jsonl --corpus corpus.jsonl  # transcript -> task row
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
| `corpus/sources.txt` | 6 public skill repos pinned to commit SHAs, with licenses. |
| `corpus/decoys/` | 3 adversarial skills (self-promotion, keyword stuffing, prompt-structure injection), flagged `metadata.skilljit-decoy`. |
| `tasks/tasks.jsonl` | 49 hand-labeled tasks: 44 positives (2 multi-skill) and 5 negatives (`needed: []`). |

Credentials: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) for `llm`; `AIMLAPI_KEY` for `d1`.

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
- Lexical and embedding scores are **not calibrated** (lexical is max-normalized; cosine sits near 0.3), so `s_min` only means something for `llm`/`d1`.
- `llm` and `d1` have only been tested against fakes. Neither has been called live: no keys in the dev environment.
- Token counts are a chars/4 estimate.
- Not built yet: re-scoring on failure (SPEC §5), agent runs across pool sizes (M3), harness plugin (M4).
