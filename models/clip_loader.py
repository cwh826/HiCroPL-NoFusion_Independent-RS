"""CLIP weight loading, extracted unchanged from the source project."""
from pathlib import Path
import torch
from clip import clip


def load_clip_to_cpu(cfg):
    weights = cfg.MODEL.CLIP_WEIGHTS
    if weights:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CLIP weights not found: {path}")
    else:
        path = clip._download(clip._MODELS[cfg.MODEL.BACKBONE.NAME],
                              root=cfg.MODEL.CLIP_CACHE)
    try:
        state = torch.jit.load(str(path), map_location="cpu").eval().state_dict()
    except RuntimeError:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    return clip.build_model(state)
