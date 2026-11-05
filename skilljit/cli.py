import argparse
import json
import sys

from .backends import BACKENDS
from .evaluate import evaluate, load_tasks
from .ingest import dump_jsonl, ingest, load_jsonl
from .label import task_from_transcript
from .select import select


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

    lb = sub.add_parser("label", help="turn agent transcripts into tasks.jsonl rows (invoked skills)")
    lb.add_argument("transcripts", nargs="+")
    lb.add_argument("--corpus", help="drop invoked skills not in this corpus")

    args = ap.parse_args(argv)
    if args.cmd == "eval":
        skills = load_jsonl(args.corpus)
        tasks = load_tasks(args.tasks, skills)
        core = [c for c in args.core.split(",") if c]
        rep = evaluate(BACKENDS[args.backend](), skills, tasks, args.budget, args.s_min, args.k, core)
        rows = rep.pop("rows")
        for key, v in rep.items():
            print(f"{key:22s} {v:.3f}" if isinstance(v, float) else f"{key:22s} {v}")
        for r in rows:
            if args.rows:
                print(json.dumps(r))
            elif r.get("missed"):
                print(f"miss {r['id']}: needed {r['missed']}, selected {r['selected']}", file=sys.stderr)
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
