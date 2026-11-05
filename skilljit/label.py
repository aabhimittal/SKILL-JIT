"""Ground-truth labels from agent transcripts (SPEC §6): which skills did the agent invoke?

Reads Claude Code session JSONL. A skill counts as invoked when the assistant calls
the `Skill` tool with it, or `Read`s its SKILL.md (harnesses without a Skill tool load
skills by reading the file). Plugin-namespaced names (`plugin:skill`) are reduced
to the bare skill name.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def _blocks(entry: dict) -> list:
    c = (entry.get("message") or {}).get("content")
    return c if isinstance(c, list) else []


def invoked_skills(transcript: str | Path) -> list[str]:
    seen: dict[str, None] = {}  # ordered set: first-invocation order
    for line in Path(transcript).read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial last line of a live transcript
        if entry.get("type") != "assistant":
            continue
        for b in _blocks(entry):
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            inp = b.get("input") or {}
            if b.get("name") == "Skill" and isinstance(inp.get("skill"), str):
                seen[inp["skill"].rsplit(":", 1)[-1]] = None
            elif b.get("name") == "Read" and str(inp.get("file_path", "")).endswith("/SKILL.md"):
                seen[PurePosixPath(inp["file_path"]).parent.name] = None
    return list(seen)


def first_user_text(transcript: str | Path) -> str:
    for line in Path(transcript).read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user":
            continue
        c = (entry.get("message") or {}).get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        texts = [b.get("text", "") for b in _blocks(entry) if isinstance(b, dict) and b.get("type") == "text"]
        if any(t.strip() for t in texts):
            return "\n".join(texts).strip()
    return ""


def task_from_transcript(transcript: str | Path, known: set[str] | None = None) -> dict:
    """One tasks.jsonl row. `known` drops skills outside the corpus (harness built-ins etc.)."""
    names = [n for n in invoked_skills(transcript) if known is None or n in known]
    return {"id": Path(transcript).stem, "task": first_user_text(transcript), "needed": names,
            "label_source": "invoked"}
