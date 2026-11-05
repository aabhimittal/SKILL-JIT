import json
from pathlib import Path

import pytest

from skilljit.backends import Backend
from skilljit.cli import main
from skilljit.evaluate import evaluate, is_decoy, load_tasks
from skilljit.ingest import NAME_RE, Skill, dump_jsonl

SUITE = Path(__file__).parent.parent / "tasks" / "tasks.jsonl"


class Fixed(Backend):
    """Returns preset scores keyed by skill name, so metrics are hand-checkable."""
    name = "fixed"

    def __init__(self, by_task):
        self.by_task = by_task

    def _score(self, task, skills):
        table = self.by_task[task]
        return [table.get(s.name, 0.0) for s in skills], None


def mk(*names, decoy=()):
    return [Skill(name=n, description="", path="", body_tokens=100,
                  metadata={"skilljit-decoy": "true"} if n in decoy else {}) for n in names]


def write_tasks(tmp_path, rows):
    p = tmp_path / "tasks.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_any_of_groups_and_metrics(tmp_path):
    skills = mk("a", "a2", "b", "c", "evil", decoy={"evil"})
    tasks = load_tasks(write_tasks(tmp_path, [
        {"id": "p", "task": "P", "needed": [["a", "a2"], "b"]},
        {"id": "n", "task": "N", "needed": []},
    ]), skills)
    backend = Fixed({"P": {"a2": 4, "evil": 3.5, "c": 3, "b": 1}, "N": {"evil": 4}})
    rep = evaluate(backend, skills, tasks, budget=10_000, s_min=2)
    p, n = rep["rows"]
    assert p["recall@1"] == 0.5          # a2 satisfies the any-of group
    assert p["recall@3"] == 0.5          # b is ranked 4th
    assert p["recall@5"] == 1.0
    assert p["selected"] == ["a2", "c", "evil"] and p["recall_selected"] == 0.5
    assert p["precision"] == pytest.approx(1 / 3) and p["missed"] == ["b"]
    assert p["decoy"] and n["decoy"]
    assert rep["negatives_false_loads"] == 1 and rep["decoy_rate"] == 1.0
    assert rep["load_all_tokens"] == 500


def test_unknown_label_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown skills"):
        load_tasks(write_tasks(tmp_path, [{"id": "x", "task": "t", "needed": ["nope"]}]), mk("a"))


def test_decoy_flag():
    assert is_decoy(mk("d", decoy={"d"})[0]) and not is_decoy(mk("d")[0])


def test_cli_eval(tmp_path, capsys, pool):
    corpus = tmp_path / "c.jsonl"
    dump_jsonl(pool, corpus)
    tasks = write_tasks(tmp_path, [{"id": "t", "task": "fill a PDF form", "needed": ["pdf-processing"]}])
    assert main(["eval", str(corpus), str(tasks), "--s-min", "0", "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert "recall_selected        1.000" in out


def test_shipped_suite_is_well_formed():
    rows = [json.loads(line) for line in SUITE.read_text().splitlines() if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows)
    assert sum(1 for r in rows if not r["needed"]) >= 5  # negatives present
    for r in rows:
        assert r["task"].strip()
        for req in r["needed"]:
            names = [req] if isinstance(req, str) else req
            assert names and all(NAME_RE.match(n) for n in names), r["id"]
