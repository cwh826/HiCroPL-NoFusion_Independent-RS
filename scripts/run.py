"""Portable experiment launcher: python scripts/run.py --help."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("patternnet", "rsicd", "resisc45", "mlrsnet",
            "patternnetv2", "rsicdv2", "resisc45v2", "mlrsnetv2")
METHODS = {"hicropl_nofusion": "HiCroPLNoFusion"}
DEFAULT_CONFIGS = {"hicropl_nofusion": "vit_b16_c16"}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", choices=["base2new", "crossdata", "domaingen"])
    parser.add_argument("phase", choices=["train", "test"])
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("seed", type=int)
    parser.add_argument("--root", required=True, help="Parent folder of the datasets")
    parser.add_argument("--source", choices=DATASETS,
                        help="Source training dataset for transfer evaluation")
    parser.add_argument("--config", help="Default configuration depends on the method")
    parser.add_argument("--method", type=str.lower, choices=METHODS, default="hicropl_nofusion",
                        help="Compatibility option; this project only runs HiCroPLNoFusion")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--subset", choices=["base", "new"], default="new",
                        help="Class subset for base2new evaluation")
    parser.add_argument("--load-epoch", type=int, help="Default: best validation checkpoint")
    parser.add_argument("--weights", help="Local CLIP JIT/state-dict weights; otherwise cached/downloaded")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args, overrides):
    if args.shots < 1:
        raise ValueError("--shots must be positive")
    if args.protocol == "domaingen":
        if not args.dataset.endswith("v2") or (args.source and not args.source.endswith("v2")):
            raise ValueError("domaingen uses the v2 datasets with aligned classes")
    if args.protocol == "crossdata":
        if args.dataset.endswith("v2") or (args.source and args.source.endswith("v2")):
            raise ValueError("crossdata uses the original datasets")
    if args.phase == "test" and args.protocol != "base2new" and not args.source:
        raise ValueError("Transfer testing requires --source (the dataset used for training)")
    if args.phase == "train" and args.source:
        raise ValueError("Training uses dataset directly; --source is for transfer testing")
    if args.phase == "test" and args.protocol == "base2new" and args.source:
        raise ValueError("base2new uses one dataset; omit --source")
    source = args.source or args.dataset
    trainer = METHODS[args.method]
    config_name = args.config or DEFAULT_CONFIGS[args.method]
    out = Path(args.output_root).expanduser().resolve()
    tag = Path(f"shots_{args.shots}") / trainer / config_name / f"seed{args.seed}"
    if args.protocol == "base2new":
        model_dir = out / "base2new" / "train_base" / args.dataset / tag
        output = model_dir if args.phase == "train" else out / "base2new" / f"test_{args.subset}" / args.dataset / tag
        subset = "base" if args.phase == "train" else args.subset
    else:
        model_dir = out / args.protocol / "train" / source / tag
        output = model_dir if args.phase == "train" else out / args.protocol / "test" / source / args.dataset / tag
        subset = "all"
    cmd = [sys.executable, "-B", str(ROOT / "train.py"),
           "--root", str(Path(args.root).expanduser().resolve()),
           "--seed", str(args.seed), "--trainer", trainer,
           "--dataset-config-file", str(ROOT / "yaml" / "datasets" / f"{args.dataset}.yaml"),
           "--config-file", str(ROOT / "yaml" / "trainers" / trainer / f"{config_name}.yaml"),
           "--output-dir", str(output)]
    if args.phase == "test":
        cmd += ["--eval-only", "--model-dir", str(model_dir)]
        if args.load_epoch is not None:
            cmd += ["--load-epoch", str(args.load_epoch)]
    cmd += ["DATASET.NUM_SHOTS", str(args.shots), "DATASET.SUBSAMPLE_CLASSES", subset]
    if args.weights:
        cmd += ["MODEL.CLIP_WEIGHTS", str(Path(args.weights).expanduser().resolve())]
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if len(overrides) % 2:
        raise ValueError("Configuration overrides must be KEY VALUE pairs after --")
    cmd += overrides
    return cmd


def main():
    parser = make_parser()
    args, overrides = parser.parse_known_args()
    cmd = build_command(args, overrides)
    print(subprocess.list2cmdline(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
