import json

from skilljit.cli import main
from skilljit.label import first_user_text, invoked_skills, task_from_transcript


def tool_use(name, inp):
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "ok"}, {"type": "tool_use", "id": "x", "name": name, "input": inp}]}}


def transcript(tmp_path, entries, partial_tail=True):
    p = tmp_path / "sess-1.jsonl"
    body = "".join(json.dumps(e) + "\n" for e in entries)
    p.write_text(body + ('{"type": "assist' if partial_tail else ""))  # live transcripts can end mid-line
    return p


ENTRIES = [
    {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Merge these PDFs"}]}},
    tool_use("Skill", {"skill": "pdf", "args": "merge"}),
    tool_use("Skill", {"skill": "document-skills:xlsx"}),
    tool_use("Read", {"file_path": "/home/u/.claude/skills/webapp-testing/SKILL.md"}),
    tool_use("Read", {"file_path": "/repo/README.md"}),
    tool_use("Skill", {"skill": "pdf"}),
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_use", "name": "Skill", "input": {"skill": "not-assistant"}}]}},
]


def test_invoked_skills_order_dedupe_and_sources(tmp_path):
    p = transcript(tmp_path, ENTRIES)
    assert invoked_skills(p) == ["pdf", "xlsx", "webapp-testing"]
    assert first_user_text(p) == "Merge these PDFs"


def test_task_row_filters_to_corpus(tmp_path, capsys, pool):
    from skilljit.ingest import dump_jsonl
    p = transcript(tmp_path, ENTRIES)
    row = task_from_transcript(p, known={"pdf", "webapp-testing"})
    assert row == {"id": "sess-1", "task": "Merge these PDFs", "needed": ["pdf", "webapp-testing"],
                   "label_source": "invoked"}
    corpus = tmp_path / "c.jsonl"
    dump_jsonl(pool, corpus)  # pool has xlsx and webapp-testing but not pdf
    assert main(["label", str(p), "--corpus", str(corpus)]) == 0
    assert json.loads(capsys.readouterr().out)["needed"] == ["xlsx", "webapp-testing"]


def test_string_user_content(tmp_path):
    p = transcript(tmp_path, [{"type": "user", "message": {"role": "user", "content": "plain text ask"}}])
    assert first_user_text(p) == "plain text ask" and invoked_skills(p) == []
