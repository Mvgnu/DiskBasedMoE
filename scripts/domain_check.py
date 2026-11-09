#!/usr/bin/env python3
"""Run the actual scoped verification commands using this Python environment."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["runtime", "service", "evaluation", "all"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    domains = ["runtime", "service", "evaluation"] if args.domain == "all" else [args.domain]
    for domain in domains:
        for command in (
            ["-m", "ruff", "check", f"src/disk_moe/{domain}", f"tests/{domain}"],
            ["-m", "pytest", f"tests/{domain}", "-q"],
        ):
            subprocess.run([sys.executable, *command], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/check_architecture.py"], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "build", "--no-isolation"], cwd=root, check=True)


if __name__ == "__main__":
    main()
