"""M0: parse Agent Skills (SKILL.md) into selection candidates.

Format follows https://agentskills.io/specification: YAML frontmatter with required
`name` (<=64, lowercase/digits/single hyphens, matches dir) and `description`
(<=1024), optional `license`, `compatibility`, `metadata`, `allowed-tools`.
Public corpora violate the spec often, so violations are recorded as `issues`
instead of raised; only an unreadable/frontmatter-less file is an error.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

SKILL_FILE = "SKILL.md"
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
LICENSE_GLOBS = ("LICENSE*", "LICENCE*", "COPYING*")
MAX_FRONTMATTER = 16_384


class SkillParseError(ValueError):
    pass


def est_tokens(text: str) -> int:
    # shortcut: ~4 chars/token heuristic, swap for a real tokenizer when token metrics go into the M3 report
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: str  # directory containing SKILL.md
    body_tokens: int  # cost of loading the full SKILL.md body on activation
    license: str | None = None
    license_file: str | None = None
    metadata: dict = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    @property
    def card(self) -> str:
        """What a selector sees: the progressive-disclosure level-1 text."""
        return f"{self.name}: {self.description}"

    @property
    def card_tokens(self) -> int:
        return est_tokens(self.card)


def validate(fm: dict, dir_name: str) -> list[str]:
    issues = []
    name, desc = fm.get("name"), fm.get("description")
    if not isinstance(name, str) or not name:
        issues.append("missing name")
    else:
        if len(name) > 64:
            issues.append("name > 64 chars")
        if not NAME_RE.match(name):
            issues.append("name not lowercase-hyphenated")
        if name != dir_name:
            issues.append(f"name != directory ({dir_name})")
    if not isinstance(desc, str) or not desc.strip():
        issues.append("missing description")
    elif len(desc) > 1024:
        issues.append("description > 1024 chars")
    comp = fm.get("compatibility")
    if comp is not None and (not isinstance(comp, str) or len(comp) > 500):
        issues.append("compatibility not a string <= 500 chars")
    meta = fm.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        issues.append("metadata not a mapping")
    return issues


def find_license_file(skill_dir: Path, root: Path) -> Path | None:
    """Nearest LICENSE-like file from the skill dir up to (and including) root."""
    d = skill_dir.resolve()
    root = root.resolve()
    while True:
        for g in LICENSE_GLOBS:
            # untrusted repos: a LICENSE symlink to e.g. /etc/passwd must not leak into the corpus
            hits = sorted(p for p in d.glob(g) if p.is_file() and p.resolve().is_relative_to(root))
            if hits:
                return hits[0]
        if d == root or d.parent == d:
            return None
        d = d.parent


def _first_line(p: Path) -> str | None:
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                return line.strip()[:120]
    except OSError:
        pass
    return None


def parse_skill(skill_md: Path, root: Path | None = None) -> Skill:
    skill_md = Path(skill_md)
    root = Path(root) if root else skill_md.parent
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SkillParseError(f"{skill_md}: unreadable: {e}") from e
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise SkillParseError(f"{skill_md}: no YAML frontmatter")
    if len(m.group(1)) > MAX_FRONTMATTER:
        raise SkillParseError(f"{skill_md}: frontmatter > {MAX_FRONTMATTER} chars")
    try:
        # aliases are legit YAML but enable billion-laughs blowup once json.dumps expands shared refs
        if any(isinstance(t, yaml.AliasToken) for t in yaml.scan(m.group(1))):
            raise SkillParseError(f"{skill_md}: YAML aliases not allowed in frontmatter")
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise SkillParseError(f"{skill_md}: bad YAML: {e}") from e
    if not isinstance(fm, dict):
        raise SkillParseError(f"{skill_md}: frontmatter is not a mapping")

    lic_file = find_license_file(skill_md.parent, root)
    lic = fm.get("license")
    return Skill(
        name=str(fm.get("name") or skill_md.parent.name),
        description=" ".join(str(fm.get("description") or "").split()),
        path=str(skill_md.parent),
        body_tokens=est_tokens(text),
        license=str(lic) if lic else (_first_line(lic_file) if lic_file else None),
        license_file=str(lic_file) if lic_file else None,
        metadata=fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {},
        issues=tuple(validate(fm, skill_md.parent.name)),
    )


def ingest(roots: Iterable[str | Path]) -> tuple[list[Skill], list[str]]:
    """Walk roots for SKILL.md files. Returns (skills, errors)."""
    skills, errors = [], []
    for root in map(Path, roots):
        for p in sorted(root.rglob(SKILL_FILE)):
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue  # .git, .cache, etc.
            try:
                skills.append(parse_skill(p, root))
            except SkillParseError as e:
                errors.append(str(e))
    return skills, errors


def dump_jsonl(skills: Iterable[Skill], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in skills:
            f.write(json.dumps(asdict(s)) + "\n")


def load_jsonl(path: str | Path) -> list[Skill]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["issues"] = tuple(d.get("issues", ()))
                out.append(Skill(**d))
    return out
