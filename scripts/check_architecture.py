#!/usr/bin/env python3
"""Small, dependency-free checks for the scoped package ownership contract."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "disk_moe"
DOMAINS = {"runtime", "service", "evaluation"}


def violations() -> list[str]:
    errors = []
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE)
        domain = rel.parts[0] if len(rel.parts) > 1 else None
        if domain is None and rel.name not in {"__init__.py", "contracts.py"}:
            errors.append(f"unowned package module: {rel}")
        elif domain is not None and domain not in DOMAINS:
            errors.append(f"unowned domain: {rel}")
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.level >= 2 and node.module:
                    modules.append("disk_moe." + node.module)
            else:
                continue
            for module in modules:
                if module.startswith(
                    (
                        "experiments",
                        "disk_moe_runtime",
                        "wikitext_dataset",
                        "glm4_disk_runtime",
                        "glm4_moe_lite_disk_backbone",
                    )
                ):
                    errors.append(f"{rel}:{node.lineno}: legacy import {module}")
                for other in DOMAINS - {domain}:
                    if module == f"disk_moe.{other}" or module.startswith(f"disk_moe.{other}."):
                        allowed = (
                            domain == "service"
                            and rel.as_posix() == "service/cli.py"
                            and other == "runtime"
                        ) or (
                            domain == "evaluation"
                            and rel.as_posix() == "evaluation/model_bench.py"
                            and other == "runtime"
                        )
                        if not allowed:
                            errors.append(f"{rel}:{node.lineno}: forbidden domain import {module}")
    for domain in DOMAINS:
        path = ROOT / "specs" / "domains" / f"{domain}.md"
        content = path.read_text()
        for field in (
            "owned:",
            "tests:",
            "exports:",
            "consumes:",
            "verification:",
            "reference:",
            "property:",
        ):
            if field not in content:
                errors.append(f"{path.relative_to(ROOT)} missing {field}")
    return errors


def main() -> None:
    errors = violations()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Architecture checks passed: ownership, import direction, legacy isolation, spec bars.")


if __name__ == "__main__":
    main()
