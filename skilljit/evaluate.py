"""M2: task suite loading and offline selector evaluation (retrieval metrics, no agent runs).

A task's `needed` is a list of requirements; each is a skill name or a list of
interchangeable names (any-of), because public pools contain near-duplicates
(e.g. `tdd` vs `test-driven-development`). A requirement is met when any member
is retrieved. Tasks with `needed: []` are negatives: every selection is a false load.
Skills with metadata `skilljit-decoy: "true"` are adversarial and never needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

from .backends import Backend
from .ingest import Skill
from .select import select


@dataclass(frozen=True)
class Task:
    id: str
    task: str
    needed: tuple[frozenset[str], ...]
    label_source: str = "hand"


def load_tasks(path: str | Path, skills: Sequence[Skill] | None = None) -> list[Task]:
    """Load tasks JSONL; with `skills`, fail loudly on labels naming skills not in the corpus."""
    known = {s.name for s in skills} if skills is not None else None
    tasks = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        needed = tuple(frozenset([r] if isinstance(r, str) else r) for r in d["needed"])
        if known is not None:
            unknown = sorted(set().union(*needed) - known) if needed else []
            if unknown:
                raise ValueError(f"{path}:{n} ({d['id']}): unknown skills {unknown}")
        tasks.append(Task(d["id"], d["task"], needed, d.get("label_source", "hand")))
    return tasks


def is_decoy(s: Skill) -> bool:
    return str(s.metadata.get("skilljit-decoy", "")).lower() == "true"


def _met(needed, names: set[str]) -> float:
    return sum(1 for g in needed if g & names) / len(needed)


def evaluate(backend: Backend, skills: Sequence[Skill], tasks: Sequence[Task], budget: int,
             s_min: float = 2.0, k: int | None = None, core: Sequence[str] = (), ks=(1, 3, 5)) -> dict:
    rows = []
    for t in tasks:
        res = backend.score(t.task, skills)
        ranked = sorted(range(len(skills)), key=lambda i: -res.scores[i])
        picked = select(skills, res.scores, budget, k, s_min, core)
        names = {skills[i].name for i in picked}
        relevant = set().union(*t.needed) if t.needed else set()
        row = {
            "id": t.id,
            "latency_s": res.latency_s,
            "scorer_tokens": res.input_tokens,
            "selected": sorted(names),
            "selected_tokens": sum(skills[i].body_tokens for i in picked),
            "decoy": any(is_decoy(skills[i]) for i in picked),
            "positive": bool(t.needed),
        }
        if t.needed:
            row.update({f"recall@{n}": _met(t.needed, {skills[i].name for i in ranked[:n]}) for n in ks})
            row["recall_selected"] = _met(t.needed, names)
            row["precision"] = len(names & relevant) / len(names) if names else None
            row["missed"] = sorted(min(g) for g in t.needed if not g & names)
        rows.append(row)

    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if not r["positive"]]
    prec = [r["precision"] for r in pos if r["precision"] is not None]
    toks = [r["scorer_tokens"] for r in rows if r["scorer_tokens"] is not None]
    return {
        "backend": backend.name,
        "pool": len(skills),
        "tasks": len(tasks),
        **{f"recall@{n}": mean(r[f"recall@{n}"] for r in pos) for n in ks if pos},
        "recall_selected": mean(r["recall_selected"] for r in pos) if pos else None,
        "precision_selected": mean(prec) if prec else None,
        "selected_tokens": mean(r["selected_tokens"] for r in rows),
        "negatives_false_loads": mean(len(r["selected"]) for r in neg) if neg else None,
        "decoy_rate": mean(r["decoy"] for r in rows),
        "latency_ms": 1000 * mean(r["latency_s"] for r in rows),
        "scorer_tokens": mean(toks) if toks else None,
        # reference points: what loading everything / only frontmatter would cost per task
        "load_all_tokens": sum(s.body_tokens for s in skills),
        "frontmatter_tokens": sum(s.card_tokens for s in skills),
        "rows": rows,
    }
