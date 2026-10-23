import argparse
import json
from dataclasses import asdict

import torch

from . import Glm4Engine, export_checkpoint, export_model
from .artifact import DTYPES
from .demo import tiny_model, tiny_tokenizer
from .export import export_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native GLM4 disk-backed reference tools")
    commands = parser.add_subparsers(dest="command", required=True)
    tiny = commands.add_parser("tiny-demo", help="Export and execute random tiny weights; this is not a trained chat model")
    tiny.add_argument("--output", required=True)
    tiny.add_argument("--dtype", choices=list(DTYPES), default="float32")
    tiny.add_argument("--cache-budget-bytes", type=int, default=2048)
    export = commands.add_parser("export", help="Incremental safetensors checkpoint export")
    export.add_argument("source")
    export.add_argument("output")
    export.add_argument("--dtype", choices=list(DTYPES))
    export.add_argument("--revision")
    args = parser.parse_args(argv)
    if args.command == "export":
        path = export_checkpoint(args.source, args.output, dtype=args.dtype, revision=args.revision)
        print(json.dumps(export_summary(path)))
        return
    torch.set_num_threads(1)
    model = tiny_model(DTYPES[args.dtype])
    export_model(model, args.output, tokenizer=tiny_tokenizer(), source={"kind": "random_tiny_fixture", "seed": 42})
    engine = Glm4Engine(args.output, cache_budget_bytes=args.cache_budget_bytes)
    print(json.dumps({"artifact_dir": args.output, "warning": "Random tiny fixture, no language-quality or large-model speed claim", "model": engine.describe()}))
    for event in engine.generate("hello disk", 6):
        print(json.dumps(asdict(event)))


if __name__ == "__main__":
    main()
