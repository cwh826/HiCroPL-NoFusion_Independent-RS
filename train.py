"""HiCroPL-Independent-RS: independent deep prompts for remote sensing."""
import argparse
from pathlib import Path
import torch
from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger
from files import patternnet, rsicd, resisc45, mlrsnet
from files import patternnetv2, rsicdv2, resisc45v2, mlrsnetv2
from models import hicropl_nofusion

PROJECT_ROOT = Path(__file__).resolve().parent


def extend_cfg(cfg):
    cfg.TRAINER.NAME = "HiCroPLNoFusion"
    cfg.TRAINER.HICROPL_NF = CN()
    cfg.TRAINER.HICROPL_NF.N_CTX = 16
    cfg.TRAINER.HICROPL_NF.CTX_INIT = "a photo of a"
    cfg.TRAINER.HICROPL_NF.PREC = "fp32"
    cfg.TRAINER.HICROPL_NF.PROMPT_DEPTH = 12
    cfg.TRAINER.HICROPL_NF.LAMBD = 12.0
    cfg.TRAINER.HICROPL_NF.ADAM_EPS = 0.001
    cfg.TRAINER.HICROPL_NF.TEXT_PROMPTS = ""
    cfg.TRAINER.HICROPL_NF.TEMPLATES = ["a satellite image of {}.",
                                      "an aerial photograph of {}.",
                                      "a remote sensing image of {}."]
    cfg.TRAINER.HICROPL_NF.TEACHER_BATCH_SIZE = 32
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.MODEL.CLIP_WEIGHTS = ""
    cfg.MODEL.CLIP_CACHE = str(PROJECT_ROOT / ".cache" / "clip")
    cfg.TEST.FINAL_MODEL = "best_val"
    cfg.TEST.SPLIT = "test"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.dataset_config_file:
        cfg.merge_from_other_cfg(CN.load_cfg(Path(args.dataset_config_file).read_text(encoding="utf-8-sig")))
    if args.config_file:
        cfg.merge_from_other_cfg(CN.load_cfg(Path(args.config_file).read_text(encoding="utf-8-sig")))
    for arg, key in [("root", "DATASET.ROOT"), ("output_dir", "OUTPUT_DIR"),
                     ("resume", "RESUME"), ("trainer", "TRAINER.NAME"),
                     ("backbone", "MODEL.BACKBONE.NAME")]:
        value = getattr(args, arg)
        if value:
            node = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = getattr(node, part)
            setattr(node, parts[-1], value)
    if args.seed is not None:
        cfg.SEED = args.seed
    cfg.merge_from_list(args.opts)
    if cfg.TRAINER.NAME != "HiCroPLNoFusion":
        raise ValueError("This project only supports HiCroPLNoFusion")
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    setup_logger(cfg.OUTPUT_DIR)
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    print(cfg)
    print(collect_env_info())
    trainer = build_trainer(cfg)
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
    elif not args.no_train:
        trainer.train()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="", help="Parent folder of the datasets")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trainer", default="")
    parser.add_argument("--backbone", default="")
    parser.add_argument("--dataset-config-file", default="")
    parser.add_argument("--config-file", default=str(PROJECT_ROOT / "yaml/trainers/HiCroPLNoFusion/vit_b16_c16.yaml"))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
