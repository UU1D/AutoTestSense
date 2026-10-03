"""Copy the final catalog into the immutable-shaped retrieval directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.input_file.is_file():
        raise FileNotFoundError(args.input_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    shutil.copy2(args.input_file, temporary)
    temporary.replace(args.output_file)
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
