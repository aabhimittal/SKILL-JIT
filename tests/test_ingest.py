import pytest

from conftest import write_skill
from skilljit.cli import main
from skilljit.ingest import SkillParseError, dump_jsonl, ingest, load_jsonl, parse_skill


def test_valid_skill_with_inherited_license(tmp_path):
    (tmp_path / "LICENSE").write_text("MIT License\n\nCopyright ...")
    p = write_skill(tmp_path / "skills", "pdf-processing",
                    "name: pdf-processing\ndescription: Extract PDF text.\nmetadata:\n  version: '1.0'")
    s = parse_skill(p, root=tmp_path)
    assert (s.name, s.description, s.issues) == ("pdf-processing", "Extract PDF text.", ())
    assert s.license == "MIT License" and s.license_file.endswith("LICENSE")
    assert s.metadata == {"version": "1.0"}
    assert s.body_tokens > s.card_tokens > 0


def test_frontmatter_license_wins_and_skill_local_file_found(tmp_path):
    p = write_skill(tmp_path, "art", "name: art\ndescription: d\nlicense: Complete terms in LICENSE.txt")
    (p.parent / "LICENSE.txt").write_text("Apache License 2.0")
    s = parse_skill(p, root=tmp_path)
    assert s.license == "Complete terms in LICENSE.txt"
    assert s.license_file.endswith("art/LICENSE.txt")


def test_license_search_stops_at_root(tmp_path):
    (tmp_path / "LICENSE").write_text("outside root")
    p = write_skill(tmp_path / "repo", "x", "name: x\ndescription: d")
    s = parse_skill(p, root=tmp_path / "repo")
    assert s.license is None and s.license_file is None


@pytest.mark.parametrize("fm, issue", [
    ("name: Bad_Name\ndescription: d", "name not lowercase-hyphenated"),
    ("name: a--b\ndescription: d", "name not lowercase-hyphenated"),
    ("name: other\ndescription: d", "name != directory"),
    ("name: s\ndescription: " + "x" * 1025, "description > 1024 chars"),
    ("name: s", "missing description"),
    ("description: d", "missing name"),
    ("name: s\ndescription: d\nmetadata: nope", "metadata not a mapping"),
])
def test_spec_violations_recorded_not_raised(tmp_path, fm, issue):
    s = parse_skill(write_skill(tmp_path, "s", fm))
    assert any(i.startswith(issue) for i in s.issues), s.issues


def test_symlinked_license_outside_root_ignored(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    p = write_skill(tmp_path / "repo", "s", "name: s\ndescription: d")
    (p.parent / "LICENSE").symlink_to(secret)
    s = parse_skill(p, root=tmp_path / "repo")
    assert s.license is None and s.license_file is None


def test_yaml_alias_bomb_rejected(tmp_path):
    bomb = "name: s\ndescription: d\nmetadata:\n  a: &a [x, x, x, x, x, x, x, x, x]\n  b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a]\n  c: [*b, *b, *b, *b, *b, *b, *b, *b, *b]"
    with pytest.raises(SkillParseError, match="aliases"):
        parse_skill(write_skill(tmp_path, "s", bomb))


def test_oversized_frontmatter_rejected(tmp_path):
    with pytest.raises(SkillParseError, match="frontmatter >"):
        parse_skill(write_skill(tmp_path, "s", "name: s\ndescription: " + "x" * 20_000))


def test_multiline_description_collapsed(tmp_path):
    s = parse_skill(write_skill(tmp_path, "s", "name: s\ndescription: >\n  line one\n  line two"))
    assert s.description == "line one line two"


@pytest.mark.parametrize("text", ["# no frontmatter", "---\nname: [unclosed\n---\n", "---\n- a list\n---\n"])
def test_unparseable_raises(tmp_path, text):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "SKILL.md").write_text(text)
    with pytest.raises(SkillParseError):
        parse_skill(tmp_path / "s" / "SKILL.md")


def test_ingest_walks_tree_skips_hidden_and_collects_errors(tmp_path, capsys):
    write_skill(tmp_path / "a", "one", "name: one\ndescription: first")
    write_skill(tmp_path / "b" / "nested", "two", "name: two\ndescription: second")
    write_skill(tmp_path / ".git", "hidden", "name: hidden\ndescription: no")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("no frontmatter")
    skills, errors = ingest([tmp_path])
    assert sorted(s.name for s in skills) == ["one", "two"]
    assert len(errors) == 1 and "broken" in errors[0]

    out = tmp_path / "corpus.jsonl"
    dump_jsonl(skills, out)
    assert load_jsonl(out) == skills

    assert main(["ingest", str(tmp_path), "-o", str(out)]) == 0
    assert "2 skills" in capsys.readouterr().out
