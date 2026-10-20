"""Command-line entry points.

Usage:
    experimentation-platform demo
    experimentation-platform serve --port 8000
    experimentation-platform experiments
    experimentation-platform plan --baseline 0.10 --mde 0.02
"""

from __future__ import annotations

import argparse
import json

from .registry import ExperimentRegistry
from .simulator import build_demo_experiment
from .stats import required_n_continuous, required_n_proportions


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    run_demo()
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry()
    registry.register(build_demo_experiment())
    print(json.dumps([e.to_dict() for e in registry.list()], indent=2))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    if args.kind == "conversion":
        n = required_n_proportions(
            args.baseline, args.mde, alpha=args.alpha, power=args.power
        )
    else:
        n = required_n_continuous(
            args.mde, args.std, alpha=args.alpha, power=args.power
        )
    print(f"required per-variant sample size: {n}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serving import build_demo_app

    uvicorn.run(build_demo_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experimentation-platform", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the end-to-end demo")
    demo.set_defaults(func=_cmd_demo)

    exps = sub.add_parser("experiments", help="List the demo experiment as JSON")
    exps.set_defaults(func=_cmd_experiments)

    plan = sub.add_parser("plan", help="Closed-form sample-size calculation")
    plan.add_argument("--kind", choices=["conversion", "continuous"],
                      default="conversion")
    plan.add_argument("--baseline", type=float, default=0.10)
    plan.add_argument("--mde", type=float, required=True,
                      help="Minimum detectable effect (absolute)")
    plan.add_argument("--std", type=float, default=10.0,
                      help="Std dev (continuous planning)")
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--power", type=float, default=0.8)
    plan.set_defaults(func=_cmd_plan)

    serve = sub.add_parser("serve", help="Run the FastAPI serving app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
