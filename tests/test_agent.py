import json
from types import SimpleNamespace as NS

import pytest

from skilljit.agent import estimate_context_tokens, run_episode, run_experiment, summarize
from skilljit.backends import LexicalBackend
from skilljit.cli import main
from skilljit.evaluate import Task, sample_pool
from skilljit.ingest import Skill, dump_jsonl


def block(type_, **kw):
    return NS(type=type_, **kw)


class PolicyClient:
    """Fake messages.create: wants `want`; invokes it when visible, else searches once, else stops.

    `bogus` makes it first call a skill name that is not on the menu (tool-not-found path).
    """

    def __init__(self, want, bogus=None):
        self.want, self.bogus, self.calls = want, bogus, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        results = [b["content"] for m in kw["messages"] if m["role"] == "user" and isinstance(m["content"], list)
                   for b in m["content"]]
        seen = kw["system"][0]["text"] + "\n".join(results)
        tool_names = {t["name"] for t in kw["tools"]}
        n = len(self.calls)
        usage = NS(input_tokens=100, output_tokens=10, cache_read_input_tokens=50 * (n > 1),
                   cache_creation_input_tokens=0)
        loaded = any(getattr(b, "name", None) == "Skill" and b.input["skill"] == self.want
                     for m in kw["messages"] if m["role"] == "assistant" for b in m["content"])
        searched = any(getattr(b, "name", None) == "find_skills"
                       for m in kw["messages"] if m["role"] == "assistant" for b in m["content"])
        if self.bogus and n == 1:
            use = block("tool_use", id=f"u{n}", name="Skill", input={"skill": self.bogus})
        elif not loaded and f'"name": "{self.want}"' in seen:
            use = block("tool_use", id=f"u{n}", name="Skill", input={"skill": self.want})
        elif not loaded and not searched and "find_skills" in tool_names:
            use = block("tool_use", id=f"u{n}", name="find_skills", input={"need": "fill a PDF form"})
        else:
            return NS(stop_reason="end_turn", content=[block("text", text="plan")], usage=usage)
        return NS(stop_reason="tool_use", content=[block("thinking", thinking=""), use], usage=usage)


@pytest.fixture
def task():
    return Task("t", "Fill in this PDF form", (frozenset({"pdf-processing"}),))


def test_all_cards_invokes_needed(task, pool):
    c = PolicyClient("pdf-processing")
    row = run_episode(c, "m", task, pool, pool)
    assert row["invoked"] == ["pdf-processing"] and row["success"] and row["precision"] == 1.0
    assert row["turns"] == 2 and row["stop_reason"] == "end_turn"
    assert row["context_tokens"] == 100 + 150 and row["output_tokens"] == 20
    assert c.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert c.calls[0]["output_config"] == {"effort": "medium"}
    # append-only: the system prompt never changes within an episode
    assert len({call["system"][0]["text"] for call in c.calls}) == 1


def test_jit_fallback_recovers_missing_skill(task, pool):
    menu = [s for s in pool if s.name != "pdf-processing"]
    c = PolicyClient("pdf-processing")
    row = run_episode(c, "m", task, pool, menu, LexicalBackend(), fallback=True)
    assert row["finds"] == 1 and row["invoked"] == ["pdf-processing"] and row["success"]
    assert "pdf-processing" not in c.calls[0]["system"][0]["text"]  # surfaced via tool result only


def test_jit_without_fallback_fails_and_rejects_unknown(task, pool):
    menu = [s for s in pool if s.name != "pdf-processing"]
    c = PolicyClient("pdf-processing", bogus="pdf-processing")
    row = run_episode(c, "m", task, pool, menu, LexicalBackend(), fallback=False)
    assert row["not_found"] == 1 and row["invoked"] == [] and not row["success"]
    err = c.calls[1]["messages"][-1]["content"][0]
    assert err["is_error"] and "find_skills" not in err["content"]
    assert [t["name"] for t in c.calls[0]["tools"]] == ["Skill"]


def test_skill_body_is_returned(tmp_path, task):
    d = tmp_path / "pdf-processing"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: pdf-processing\ndescription: d\n---\nBODY TEXT")
    s = Skill(name="pdf-processing", description="d", path=str(d), body_tokens=5)
    c = PolicyClient("pdf-processing")
    run_episode(c, "m", task, [s], [s])
    assert "BODY TEXT" in c.calls[1]["messages"][-1]["content"][0]["content"]


def test_max_turns_marks_unfinished(task, pool):
    class Loop(PolicyClient):
        def create(self, **kw):
            self.calls.append(kw)
            return NS(stop_reason="tool_use", usage=NS(input_tokens=1, output_tokens=1),
                      content=[block("tool_use", id=f"u{len(self.calls)}", name="Skill", input={"skill": "xlsx"})])
    row = run_episode(Loop("x"), "m", task, pool, pool, max_turns=3)
    assert row["turns"] == 3 and row["stop_reason"] == "tool_use"


def test_sample_pool_keeps_labels_and_decoys_and_is_deterministic(pool, task):
    decoy = Skill(name="evil", description="", path="", body_tokens=1, metadata={"skilljit-decoy": "true"})
    big = pool + [decoy] + [Skill(name=f"x{i}", description="", path="", body_tokens=1) for i in range(50)]
    p = sample_pool(big, task, 10)
    assert len(p) == 10 and {"pdf-processing", "evil"} <= {s.name for s in p}
    assert [s.name for s in p] == [s.name for s in sample_pool(big, task, 10)]
    assert sample_pool(big, task, None) == big


def test_experiment_grid_skip_and_summary(pool, task):
    neg = Task("n", "Write a haiku", ())
    rows = list(run_experiment(PolicyClient("pdf-processing"), "m", LexicalBackend(), pool, [task, neg],
                               ["all-cards", "jit"], [None], budget=8000, k=2, s_min=0,
                               skip={("jit", None, "n")}))
    assert [(r["method"], r["id"]) for r in rows] == [("all-cards", "t"), ("jit", "t"), ("all-cards", "n")]
    assert rows[0]["selection_latency_s"] is None and rows[1]["selection_latency_s"] >= 0
    s = {x["method"]: x for x in summarize(rows)}
    assert s["all-cards"]["success"] == 1.0 and s["all-cards"]["episodes"] == 2
    assert estimate_context_tokens([task], pool, ["jit"], [None]) > 0


def test_cli_dry_run_and_report(tmp_path, capsys, pool):
    corpus, tasks = tmp_path / "c.jsonl", tmp_path / "t.jsonl"
    dump_jsonl(pool, corpus)
    tasks.write_text(json.dumps({"id": "t", "task": "fill a PDF form", "needed": ["pdf-processing"]}) + "\n")
    assert main(["run", str(corpus), str(tasks), "-o", str(tmp_path / "r.jsonl"),
                 "--pool-size", "3,0", "--dry-run"]) == 0
    assert capsys.readouterr().out.startswith("6 episodes")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"method": "jit", "pool": 5, "positive": True, "success": True, "recall": 1.0,
                                "precision": 1.0, "invoked": ["a"], "decoy": False, "context_tokens": 10,
                                "output_tokens": 2, "finds": 0, "stop_reason": "end_turn"}) + "\n")
    assert main(["report", str(rows)]) == 0
    assert "| jit | 5 | 1 | 1.00 | 1.00 |" in capsys.readouterr().out


def test_cli_eval_pool_sweep_table(tmp_path, capsys, pool):
    corpus, tasks = tmp_path / "c.jsonl", tmp_path / "t.jsonl"
    dump_jsonl(pool, corpus)
    tasks.write_text(json.dumps({"id": "t", "task": "fill a PDF form", "needed": ["pdf-processing"]}) + "\n")
    assert main(["eval", str(corpus), str(tasks), "--pool-size", "2,0", "--s-min", "0"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[2].startswith("| lexical | 2 |") and out[3].startswith("| lexical | 5 |")
