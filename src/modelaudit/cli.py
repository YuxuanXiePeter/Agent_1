"""CLI modes: deterministic audit (no API) and LLM-driven analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

from .demo import make_demo
from .report import AuditEngine


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="modelaudit", description="Evidence-grounded ML/CV audit agent")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Generate sample CSV and audit without an API key")
    demo.add_argument("--data", default="data/demo_predictions.csv")
    demo.add_argument("--out", default="outputs/demo")

    audit = commands.add_parser("audit", help="Offline deterministic audit")
    audit.add_argument("--csv", required=True)
    audit.add_argument("--out", default="outputs/audit")

    agent = commands.add_parser("agent", help="Agent reasons about the dataset using tools")
    agent.add_argument("--csv", required=True)
    agent.add_argument("--out", default="outputs/agent")
    agent.add_argument("--question", default="Diagnose weaknesses, inspect leakage, and generate a report")
    agent.add_argument("--session-db", default="conversations.db")

    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            csv = make_demo(args.data)
            print(f"Demo CSV created: {csv}")
            engine = AuditEngine(csv, args.out)
            print(engine.write_all())
        elif args.command == "audit":
            print(AuditEngine(args.csv, args.out).write_all())
        elif args.command == "agent":
            from .agent import run_agent
            print(run_agent(AuditEngine(args.csv, args.out), args.question,
                            session_db=args.session_db))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
