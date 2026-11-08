import argparse
import json
import sys

from pathlib import Path

from .agent import METHODS, episodes, estimate_context_tokens, load_rows, run_experiment, summarize
from .backends import BACKENDS
from .evaluate import evaluate, load_tasks
from .ingest import dump_jsonl, ingest, load_jsonl
from .label import task_from_transcript
from .select import select


def _sizes(spec: str) -> list[int | None]:
    return [int(x) or None for x in spec.split(",") if x.strip()]


def _f(v, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _read_rows(path) -> list[dict]:
    return load_rows(path) if Path(path).exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="skilljit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ingest", help="parse SKILL.md files under roots into a JSONL corpus")
    a.add_argument("roots", nargs="+")
    a.add_argument("-o", "--out", required=True)

    s = sub.add_parser("score", help="score a corpus against a task and show the selection")
    s.add_argument("corpus")
    s.add_argument("--task", required=True)
    s.add_argument("--backend", choices=sorted(BACKENDS), default="lexical")
    s.add_argument("--budget", type=int, default=8000, help="token budget for loaded skill bodies")
    s.add_argument("--k", type=int, default=None)
    s.add_argument("--s-min", type=float, default=2.0)
    s.add_argument("--core", default="", help="comma-separated skill names always loaded")
    s.add_argument("--top", type=int, default=15)

    e = sub.add_parser("eval", help="retrieval metrics for a backend over a task suite")
    e.add_argument("corpus")
    e.add_argument("tasks")
    e.add_argument("--backend", choices=sorted(BACKENDS), default="lexical")
    e.add_argument("--budget", type=int, default=8000)
    e.add_argument("--k", type=int, default=None)
    e.add_argument("--s-min", type=float, default=2.0)
    e.add_argument("--core", default="")
    e.add_argument("--rows", action="store_true", help="also print per-task rows as JSONL")
    e.add_argument("--pool-size", default="", help="comma-separated per-task pool sizes, 0 = full pool "
                   "(several sizes print one markdown table row each)")
    e.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="M3 agent episodes (Claude API) across methods and pool sizes")
    r.add_argument("corpus")
    r.add_argument("tasks")
    r.add_argument("-o", "--out", required=True, help="episode rows JSONL (appended, resumable)")
    r.add_argument("--backend", choices=sorted(BACKENDS), default="lexical", help="selector for jit methods")
    r.add_argument("--methods", default=",".join(METHODS))
    r.add_argument("--pool-size", default="10,50,200,0")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--effort", default="medium")
    r.add_argument("--budget", type=int, default=8000)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--s-min", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    r.add_argument("--dry-run", action="store_true", help="print episode count and token estimate, call nothing")

    rp = sub.add_parser("report", help="aggregate run rows into a markdown table")
    rp.add_argument("rows")

    lb = sub.add_parser("label", help="turn agent transcripts into tasks.jsonl rows (invoked skills)")
    lb.add_argument("transcripts", nargs="+")
    lb.add_argument("--corpus", help="drop invoked skills not in this corpus")

    args = ap.parse_args(argv)
    if args.cmd == "eval":
        skills = load_jsonl(args.corpus)
        tasks = load_tasks(args.tasks, skills)
        core = [c for c in args.core.split(",") if c]
        sizes = _sizes(args.pool_size) or [None]
        backend = BACKENDS[args.backend]()
        if len(sizes) > 1:
            print("| backend | pool | recall@1 | recall@5 | selected recall | precision | selected tokens "
                  "| false loads/neg | decoy rate | load-all tokens | latency ms |")
            print("|---" * 11 + "|")
        for n in sizes:
            rep = evaluate(backend, skills, tasks, args.budget, args.s_min, args.k, core,
                           pool_size=n, seed=args.seed)
            rows = rep.pop("rows")
            if len(sizes) > 1:
                print(f"| {rep['backend']} | {rep['pool']} | {rep['recall@1']:.2f} | {rep['recall@5']:.2f} "
                      f"| {rep['recall_selected']:.2f} | {_f(rep['precision_selected'])} "
                      f"| {rep['selected_tokens']:.0f} | {_f(rep['negatives_false_loads'], 1)} "
                      f"| {rep['decoy_rate']:.0%} | {rep['load_all_tokens']:.0f} | {rep['latency_ms']:.1f} |")
                continue
            for key, v in rep.items():
                print(f"{key:22s} {v:.3f}" if isinstance(v, float) else f"{key:22s} {v}")
            for row in rows:
                if args.rows:
                    print(json.dumps(row))
                elif row.get("missed"):
                    print(f"miss {row['id']}: needed {row['missed']}, selected {row['selected']}", file=sys.stderr)
        return 0
    if args.cmd == "run":
        skills = load_jsonl(args.corpus)
        tasks = load_tasks(args.tasks, skills)[: args.limit]
        methods = [m for m in args.methods.split(",") if m]
        bad = sorted(set(methods) - set(METHODS))
        if bad:
            ap.error(f"unknown methods {bad}; choose from {METHODS}")
        sizes = _sizes(args.pool_size)
        if args.dry_run:
            n_ep = sum(1 for _ in episodes(tasks, skills, methods, sizes, args.seed))
            est = estimate_context_tokens(tasks, skills, methods, sizes, args.seed, k=args.k)
            print(f"{n_ep} episodes, ~{est / 1e6:.1f}M context tokens estimated "
                  f"(3 calls/episode; prompt caching lowers the bill)")
            return 0
        import anthropic  # optional dependency: pip install skill-jit[llm]

        client, backend = anthropic.Anthropic(), BACKENDS[args.backend]()
        done = {(d["method"], d["pool_size"], d["id"]) for d in _read_rows(args.out)}
        with open(args.out, "a", encoding="utf-8") as f:
            for row in run_experiment(client, args.model, backend, skills, tasks, methods, sizes,
                                      args.budget, args.k, args.s_min, args.seed, args.effort, skip=done):
                f.write(json.dumps(row) + "\n")
                f.flush()  # one row per episode so an interrupted run resumes where it stopped
                print(f"{row['method']:15s} pool={row['pool']:4d} {row['id']} invoked={row['invoked']} "
                      f"ctx={row['context_tokens']}", file=sys.stderr)
        return 0
    if args.cmd == "report":
        print("| method | pool | episodes | success | recall | precision | false loads/neg | decoy rate "
              "| context tokens | output tokens | find calls | unfinished |")
        print("|---" * 12 + "|")
        for s in summarize(load_rows(args.rows)):
            print(f"| {s['method']} | {s['pool']} | {s['episodes']} | {_f(s['success'])} | {_f(s['recall'])} "
                  f"| {_f(s['precision'])} | {_f(s['false_loads_neg'], 1)} | {s['decoy_rate']:.0%} "
                  f"| {s['context_tokens']:.0f} | {s['output_tokens']:.0f} | {s['finds']:.2f} | {s['unfinished']} |")
        return 0
    if args.cmd == "label":
        known = {s.name for s in load_jsonl(args.corpus)} if args.corpus else None
        for t in args.transcripts:
            print(json.dumps(task_from_transcript(t, known)))
        return 0
    if args.cmd == "ingest":
        skills, errors = ingest(args.roots)
        dump_jsonl(skills, args.out)
        flagged = [sk for sk in skills if sk.issues]
        unlicensed = sum(1 for sk in skills if not sk.license)
        for sk in flagged:
            print(f"issue {sk.path}: {'; '.join(sk.issues)}", file=sys.stderr)
        for e in errors:
            print(f"error {e}", file=sys.stderr)
        print(f"{len(skills)} skills -> {args.out} ({len(flagged)} with spec issues, "
              f"{unlicensed} without license, {len(errors)} unparseable)")
        return 0

    skills = load_jsonl(args.corpus)
    res = BACKENDS[args.backend]().score(args.task, skills)
    core = [c for c in args.core.split(",") if c]
    picked = set(select(skills, res.scores, args.budget, args.k, args.s_min, core))
    order = sorted(range(len(skills)), key=lambda i: -res.scores[i])
    for i in order[: args.top]:
        mark = "*" if i in picked else " "
        print(f"{mark} {res.scores[i]:4.2f}  {skills[i].body_tokens:6d}t  {skills[i].name}")
    print(f"{args.backend}: {len(skills)} skills in {res.latency_s * 1000:.1f} ms, "
          f"scorer input tokens={res.input_tokens}, selected {len(picked)} "
          f"({sum(skills[i].body_tokens for i in picked)}t / budget {args.budget})")
    return 0
