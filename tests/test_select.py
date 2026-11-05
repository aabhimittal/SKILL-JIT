from skilljit.cli import main
from skilljit.ingest import Skill, dump_jsonl
from skilljit.select import select


def mk(costs):
    return [Skill(name=f"s{i}", description="", path="", body_tokens=c) for i, c in enumerate(costs)]


def test_budget_threshold_and_ratio_order():
    skills = mk([100, 100, 100, 100])
    picked = select(skills, [4, 3, 1, 2.5], budget=250, s_min=2)
    assert picked == [0, 1]  # s2 below s_min; s3 doesn't fit after 0,1


def test_k_limits_non_core():
    assert select(mk([10] * 5), [4] * 5, budget=10_000, k=2) == [0, 1]


def test_k_cuts_by_score_not_ratio():
    # regression: tiny low-score skills used to win every k slot on p/t and push out the needed one
    skills = mk([2000, 100, 100])
    assert select(skills, [4.0, 0.5, 0.5], budget=8000, k=1, s_min=0) == [0]


def test_core_always_included_and_consumes_budget():
    skills = mk([300, 100, 100])
    assert select(skills, [0, 4, 4], budget=400, core=["s0"]) == [0, 1]
    assert select(skills, [0, 4, 4], budget=100, core=["s0"]) == [0]  # core over budget, nothing else fits


def test_best_single_item_beats_bad_greedy():
    # greedy by ratio takes the small 2.0 item, which blocks the 4.0 item that alone fits
    skills = mk([10, 1000])
    assert select(skills, [2.0, 4.0], budget=1000) == [1]


def test_cli_score(tmp_path, capsys, pool):
    corpus = tmp_path / "c.jsonl"
    dump_jsonl(pool, corpus)
    assert main(["score", str(corpus), "--task", "fill a PDF form", "--budget", "600"]) == 0
    out = capsys.readouterr().out
    assert "* 4.00     500t  pdf-processing" in out and "selected 1" in out
