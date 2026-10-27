from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .config import load_settings
from .stages import (
    evaluate_stage, evidence_stage, multihop_stage, preflight,
    retrieval_stage, verifier_stage,
)


def parse_hops(value: str) -> set[int]:
    hops = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not hops or not hops <= {2, 3, 4}:
        raise argparse.ArgumentTypeError("hops must be a comma-separated subset of 2,3,4")
    return hops


def parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[2] / "configs" / "base.json"
    result = argparse.ArgumentParser(description="HoVer E2E v2")
    result.add_argument("command", choices=(
        "preflight", "retrieval", "multihop", "evidence", "verifier", "evaluate"
    ))
    result.add_argument("--config", default=str(default_config))
    result.add_argument("--data-root")
    result.add_argument("--output-root")
    result.add_argument("--run-name", default="roberta_e2e_v2")
    result.add_argument("--hops", type=parse_hops, default={2})
    result.add_argument("--no-title-index", action="store_true")
    result.add_argument("--corenlp-url", default="http://127.0.0.1:9000")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.perf_counter()
    print(
        f"[stage] start command={args.command} run={args.run_name} "
        f"hops={','.join(map(str, sorted(args.hops)))}",
        flush=True,
    )
    settings = load_settings(
        args.config, data_root=args.data_root, output_root=args.output_root
    )
    if args.command == "preflight":
        output = preflight(settings, args.run_name, build_titles=not args.no_title_index)
    elif args.command == "retrieval":
        output = retrieval_stage(settings, args.run_name, hops=args.hops)
    elif args.command == "multihop":
        output = multihop_stage(settings, args.run_name, hops=args.hops)
    elif args.command == "evidence":
        output = evidence_stage(
            settings, args.run_name, hops=args.hops, corenlp_url=args.corenlp_url
        )
    elif args.command == "verifier":
        output = verifier_stage(settings, args.run_name, hops=args.hops)
    else:
        output = evaluate_stage(settings, args.run_name, hops=args.hops)
    print(
        f"[stage] complete command={args.command} "
        f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
        flush=True,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
