import argparse
import sys

from .backends import BACKENDS
from .ingest import dump_jsonl, ingest, load_jsonl
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

    args = ap.parse_args(argv)
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
