from pathlib import Path

import pytest

from skilljit.ingest import Skill

POOL = {
    "pdf-processing": "Extract text and tables from PDF files, fill PDF forms, merge PDFs. Use when handling PDF documents.",
    "xlsx": "Create, edit and analyze Excel spreadsheets with formulas and charts.",
    "git-commit": "Write conventional commit messages for staged git changes.",
    "webapp-testing": "Test local web applications with Playwright: click, screenshot, read console logs.",
    "mcp-builder": "Build MCP servers that expose external APIs as tools for LLM agents.",
}


def write_skill(root: Path, dirname: str, frontmatter: str, body: str = "# Body\nDo the thing.\n") -> Path:
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return p


@pytest.fixture
def pool() -> list[Skill]:
    return [Skill(name=n, description=d, path=f"/x/{n}", body_tokens=500) for n, d in POOL.items()]
