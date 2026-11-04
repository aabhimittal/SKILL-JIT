"""Budgeted recall: pick skills maximizing sum(p_s) subject to sum(t_s) <= budget (0/1 knapsack).

Greedy by p/t, then compare against the best single item: the classic fix that
keeps greedy within 2x of optimal (plain greedy can be arbitrarily bad when one
large, highly relevant skill is crowded out by many small, weak ones).
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .ingest import Skill


def select(
    skills: Sequence[Skill],
    scores: Sequence[float],
    budget: int,
    k: int | None = None,
    s_min: float = 2.0,
    core: Iterable[str] = (),
) -> list[int]:
    """Return indices into `skills`: core skills (always, even over budget) then chosen ones."""
    core = set(core)
    chosen = [i for i, s in enumerate(skills) if s.name in core]
    left = budget - sum(skills[i].body_tokens for i in chosen)
    cand = [i for i, s in enumerate(skills) if s.name not in core and scores[i] >= s_min]
    if k is not None:
        # k caps a count, so cut by score first (SPEC §5: top-k, then fit the budget);
        # cutting by p/t would fill the k slots with tiny low-score skills
        cand = sorted(cand, key=lambda i: -scores[i])[:k]
    cand.sort(key=lambda i: (-scores[i] / skills[i].body_tokens, -scores[i]))

    greedy, used = [], 0
    for i in cand:
        if used + skills[i].body_tokens <= left:
            greedy.append(i)
            used += skills[i].body_tokens

    fits = [i for i in cand if skills[i].body_tokens <= left]
    best = max(fits, key=lambda i: scores[i], default=None)
    if best is not None and k != 0 and scores[best] > sum(scores[i] for i in greedy):
        greedy = [best]
    return chosen + greedy
