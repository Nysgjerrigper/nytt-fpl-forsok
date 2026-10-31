"""Re-register the pinned pilot; execute only with an explicit --execute flag."""
from __future__ import annotations

import argparse
from pathlib import Path

from fpl.research_workflow import register, report, run_next


def main() -> None:
    """Use the existing workflow; dataset/runtime/source admission remains intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.dataset is None:
        parser.error("--execute requires --dataset")
    root = register(Path(__file__).with_name("question.json"), args.output)
    if args.execute:
        run_next(root, args.dataset, "Explicit reproduction of the preregistered one-attempt workflow pilot.")
    print(report(root))


if __name__ == "__main__":
    main()
