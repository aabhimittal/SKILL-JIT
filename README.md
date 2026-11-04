# SKILL-JIT

Just-in-time skill and tool loading: score every available skill against the task, load only what the task needs within a token budget. See [SPEC.md](SPEC.md).

Status: **M0 (ingester) and M1 (scoring backends) done.** M2–M4 not started.

## Install & run

```bash
pip install -e '.[test]'          # add ,llm for the Claude backend
git clone --depth 1 https://github.com/anthropics/skills /tmp/skills
python -m skilljit ingest /tmp/skills -o corpus.jsonl
python -m skilljit score corpus.jsonl --task "Fill in this PDF form" --backend lexical --budget 8000
pytest -q
```

## Layout

| Module | What |
|---|---|
| `skilljit/ingest.py` | M0. Walks roots for `SKILL.md`, parses YAML frontmatter, validates against the [Agent Skills spec](https://agentskills.io/specification), records license (frontmatter `license`, else nearest `LICENSE*`/`COPYING*` up to the root) and token cost. Spec violations become `issues`, not exceptions. |
| `skilljit/backends.py` | M1. `Backend.score(task, skills) -> ScoreResult(scores 0–4, latency_s, input_tokens)`. Backends: `lexical` (BM25), `embedding` (pluggable embedder; default is an offline hashing embedder), `llm` (one Claude call, JSON-schema output), `d1` (Liquid AI d1 via AI/ML API, `score` questions batched ≤128 per request). |
| `skilljit/select.py` | Budgeted-recall knapsack: greedy by score/tokens plus best-single-item fallback; `s_min`, `k`, and an always-loaded core set. |

Credentials: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) for `llm`; `AIMLAPI_KEY` for `d1`.

## Format conventions checked (2026-11)

Surveyed `anthropics/skills` and `obra/superpowers` (35 skills):

- Only `name`, `description` and `license` appear in practice; `metadata`/`compatibility`/`allowed-tools` are unused there.
- Spec violations exist even in the reference repo: one description > 1024 chars, one `name` ≠ directory.
- License lives either in frontmatter (`license: Complete terms in LICENSE.txt`) with a per-skill file, or only in a repo-root `LICENSE`. The ingester handles both.
- The spec's architecture diagram mentions "triggers"; no such field exists in the spec. Trigger phrasing lives inside `description`.
- Body sizes span 377 to 26k tokens, so the knapsack cost term matters: one outlier skill can use up the whole budget.

## Known gaps

- Lexical and embedding scores are **not calibrated**. Lexical is max-normalized (the top hit is always 4.0), and the embedding backend maps cosine to 0–4 (real cosines sit near 0.3, so they never reach `s_min=2`). On the real corpus this already causes a wrong selection: for a PDF task, lexical selects `canvas-design` at 2.49. Use these as rank baselines (top-k, `--s-min 0`). The threshold only means something for `llm`/`d1`.
- `llm` and `d1` have only been tested against fakes. Neither has been called live: no keys in the dev environment.
- Token counts are a chars/4 estimate.
- Not built yet: re-scoring on failure (SPEC §5) and harness loading/unloading (M4).
