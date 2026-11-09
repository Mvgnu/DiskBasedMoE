"""Composition boundary for the optional native model and HTTP dependencies."""

import argparse


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Serve a native GLM4 disk export locally")
    result.add_argument("--model-dir", required=True, help="Validated GLM4 disk export")
    result.add_argument("--tokenizer", help="Tokenizer path/name, when not saved with export")
    result.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, cuda, mps")
    result.add_argument("--cache-mib", type=int, default=256, help="Global expert cache budget in MiB")
    result.add_argument("--max-new-tokens", type=int, default=128, help="Default per-request output limit")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.cache_mib < 0:
        parser().error("--cache-mib must be non-negative")
    if not 1 <= args.max_new_tokens <= 2048:
        parser().error("--max-new-tokens must be in 1..2048")
    if not 1 <= args.port <= 65535:
        parser().error("--port must be in 1..65535")

    import uvicorn

    from disk_moe.runtime import Glm4Engine

    from .app import create_app

    engine = Glm4Engine(
        args.model_dir,
        tokenizer_name=args.tokenizer,
        device=args.device,
        cache_budget_bytes=args.cache_mib * 1024 * 1024,
    )
    uvicorn.run(
        create_app(engine, default_max_new_tokens=args.max_new_tokens),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
