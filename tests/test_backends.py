import json
from types import SimpleNamespace

import pytest

from skilljit.backends import D1Backend, EmbeddingBackend, LexicalBackend, LLMBackend, SCALE
from skilljit.ingest import Skill

TASK = "Merge these three PDF invoices into one PDF and extract the totals table"


class FakeClaude:
    """Mimics anthropic.Anthropic().messages.create; scores 4 for skills whose name contains `hit`."""

    def __init__(self, hit="pdf", drop=None, stop="end_turn"):
        self.hit, self.drop, self.stop, self.calls = hit, drop, stop, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][0]["content"]
        names = [line.split("] ", 1)[1].split(":")[0] for line in prompt.splitlines() if line.startswith("[")]
        scores = [{"index": i, "score": 4 if self.hit in n else 0} for i, n in enumerate(names) if i != self.drop]
        return SimpleNamespace(
            stop_reason=self.stop,
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=json.dumps({"scores": scores}))],
            usage=SimpleNamespace(input_tokens=321),
        )


class FakeD1:
    def __init__(self, hit="pdf"):
        self.hit, self.calls = hit, []

    def __call__(self, url, payload, headers):
        self.calls.append(payload)
        ans = {k: {"type": "score", "score": 3.6 if self.hit in q["instructions"] else 0.2}
               for k, q in payload["questions"].items()}
        return {"answers": ans, "usage": {"input_tokens": 10 * len(ans), "output_tokens": 0}}


BACKENDS = {
    "lexical": lambda: LexicalBackend(),
    "embedding": lambda: EmbeddingBackend(),
    "llm": lambda: LLMBackend(client=FakeClaude()),
    "d1": lambda: D1Backend(api_key="k", post=FakeD1()),
}


@pytest.mark.parametrize("name", BACKENDS)
def test_identical_interface_ranks_relevant_skill_first(name, pool):
    res = BACKENDS[name]().score(TASK, pool)
    assert len(res.scores) == len(pool)
    assert all(0 <= s <= 4 for s in res.scores) and all(0 <= p <= 1 for p in res.probs)
    assert res.latency_s >= 0
    assert pool[max(range(len(pool)), key=res.scores.__getitem__)].name == "pdf-processing"


@pytest.mark.parametrize("name", BACKENDS)
def test_empty_pool(name):
    assert BACKENDS[name]().score(TASK, []).scores == []


def test_lexical_no_overlap_is_all_zero(pool):
    assert LexicalBackend().score("zzz qqq", pool).scores == [0.0] * len(pool)


def test_llm_request_shape_and_tokens(pool):
    fake = FakeClaude()
    res = LLMBackend(client=fake, model="m").score(TASK, pool)
    kw = fake.calls[0]
    assert kw["model"] == "m" and kw["output_config"]["format"]["type"] == "json_schema"
    assert TASK in kw["messages"][0]["content"]
    assert res.input_tokens == 321


def test_llm_missing_index_raises(pool):
    with pytest.raises(RuntimeError, match="no score"):
        LLMBackend(client=FakeClaude(drop=2)).score(TASK, pool)


def test_llm_truncated_raises(pool):
    with pytest.raises(RuntimeError, match="max_tokens"):
        LLMBackend(client=FakeClaude(stop="max_tokens")).score(TASK, pool)


def test_d1_batches_at_128_and_sums_tokens():
    skills = [Skill(name=f"s{i}", description="pdf" if i == 129 else "other", path="", body_tokens=1) for i in range(130)]
    fake = FakeD1()
    res = D1Backend(api_key="k", post=fake).score(TASK, skills)
    assert [len(c["questions"]) for c in fake.calls] == [128, 2]
    q = next(iter(fake.calls[0]["questions"].values()))
    assert q["type"] == "score" and q["criteria"] == list(SCALE)
    assert fake.calls[0]["state"] == TASK and fake.calls[0]["model"] == "liquid/d1"
    assert res.scores[129] == 3.6 and res.scores[0] == 0.2
    assert res.input_tokens == 1300


def test_d1_missing_answer_and_key(pool, monkeypatch):
    with pytest.raises(RuntimeError, match="missing answer"):
        D1Backend(api_key="k", post=lambda *a: {"answers": {}}).score(TASK, pool)
    monkeypatch.delenv("AIMLAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AIMLAPI_KEY"):
        D1Backend(post=FakeD1()).score(TASK, pool)
