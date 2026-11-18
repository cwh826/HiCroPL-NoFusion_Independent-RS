"""HiCroPL-derived baseline without either direction of prompt interaction.

Removed: text2visual_net, visual2text_net, proxy tokens and attention pooling.
Retained: independent deep prompts, frozen-feature residuals and cosine
distillation. RSICD teacher text uses declared fixed templates by default.
This is an adapted ablation, not complete HiCroPL or an exact peer-code copy.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_checkpoint
from clip import clip
from clip.model import convert_weights
from clip.hicropl_model import IndependentTextEncoder, IndependentVisionEncoder
from .clip_loader import load_clip_to_cpu


@torch.no_grad()
def frozen_text_features(clip_model, classnames, options):
    names = [name.replace("_", " ") for name in classnames]
    if options.TEXT_PROMPTS:
        path = Path(options.TEXT_PROMPTS).expanduser()
        descriptions = json.loads(path.read_text(encoding="utf-8-sig"))
        mode = f"description file: {path}"
    else:
        if not options.TEMPLATES:
            raise ValueError("At least one fixed teacher template is required")
        descriptions = {name: [template.format(name) for template in options.TEMPLATES] for name in names}
        mode = f"fixed remote-sensing templates ({len(options.TEMPLATES)})"
    groups = []
    for name in names:
        texts = descriptions.get(name)
        if not isinstance(texts, list) or not texts or not all(isinstance(s, str) and s.strip() for s in texts):
            raise ValueError(f"Teacher descriptions missing or invalid for class {name!r}")
        groups.append(texts)
    tokens = clip.tokenize([text for texts in groups for text in texts])
    device = clip_model.visual.conv1.weight.device
    # HiCroPL normalizes each description before averaging, unlike PromptSRC.
    features = torch.cat([F.normalize(clip_model.encode_text(batch.to(device)).float(), dim=-1)
                          for batch in tokens.split(options.TEACHER_BATCH_SIZE)])
    result = []
    start = 0
    for texts in groups:
        result.append(F.normalize(features[start:start + len(texts)].mean(dim=0), dim=0))
        start += len(texts)
    print(f"HiCroPLNoFusion teacher text: {mode}")
    return torch.stack(result)


class IndependentPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        options = cfg.TRAINER.HICROPL_NF
        depth, n_ctx = options.PROMPT_DEPTH, options.N_CTX
        max_depth = min(len(clip_model.visual.transformer.resblocks), len(clip_model.transformer.resblocks))
        if not 1 <= depth <= max_depth:
            raise ValueError(f"PROMPT_DEPTH must be between 1 and {max_depth}")
        if not 1 <= n_ctx < clip_model.context_length - 2:
            raise ValueError("Invalid HiCroPL text context length")
        if tuple(cfg.INPUT.SIZE) != (clip_model.visual.input_resolution,) * 2:
            raise ValueError("INPUT.SIZE must match the CLIP weights' resolution")
        dtype, device = clip_model.dtype, clip_model.visual.conv1.weight.device
        text_width = clip_model.ln_final.weight.numel()
        vision_width = clip_model.visual.conv1.out_channels
        prefix = options.CTX_INIT.replace("_", " ").strip()
        if prefix and n_ctx <= 4:
            tokens = clip.tokenize(prefix).to(device)
            if int(tokens[0].argmax()) - 1 < n_ctx:
                raise ValueError("Initialization text has too few tokens")
            with torch.no_grad():
                ctx = clip_model.token_embedding(tokens)[0, 1:1+n_ctx].to(dtype).clone()
        else:
            ctx = torch.empty(n_ctx, text_width, dtype=dtype, device=device)
            nn.init.normal_(ctx, std=0.02)
            prefix = " ".join(["X"] * n_ctx)
        self.text_prompts = nn.ParameterList([nn.Parameter(ctx)])
        for _ in range(depth - 1):
            ctx = torch.empty(n_ctx, text_width, dtype=dtype, device=device)
            nn.init.normal_(ctx, std=0.02)
            self.text_prompts.append(nn.Parameter(ctx))
        visual = torch.empty(n_ctx, vision_width, dtype=dtype, device=device)
        nn.init.normal_(visual, std=0.02)
        # Retain the reference initialization: equal values, independent storage.
        self.visual_prompts = nn.ParameterList([nn.Parameter(visual.clone()) for _ in range(depth)])
        tokens = clip.tokenize([f"{prefix} {name.replace('_', ' ')}." for name in classnames]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokens).to(dtype)
        self.register_buffer("tokenized_prompts", tokens, persistent=False)
        self.register_buffer("token_prefix", embedding[:, :1], persistent=False)
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:], persistent=False)
        self.n_cls = len(classnames)

    def forward(self):
        ctx = self.text_prompts[0].unsqueeze(0).expand(self.n_cls, -1, -1)
        text = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        # Neither branch reads the other branch's parameters.
        return text, self.visual_prompts[0], list(self.text_prompts[1:]), list(self.visual_prompts[1:])


class HiCroPLNoFusionModel(nn.Module):
    def __init__(self, cfg, classnames, backbone, teacher_image, fixed_text):
        super().__init__()
        self.prompt_learner = IndependentPromptLearner(cfg, classnames, backbone)
        self.image_encoder = IndependentVisionEncoder(backbone.visual)
        self.text_encoder = IndependentTextEncoder(backbone)
        self.logit_scale = backbone.logit_scale
        self.teacher_image = teacher_image
        self.register_buffer("fixed_text", fixed_text, persistent=False)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("prompt_learner."))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.teacher_image.eval()
        return self

    def prompted_features(self, image):
        text, shallow_visual, deep_text, deep_visual = self.prompt_learner()
        image_features = self.image_encoder(image.to(self.image_encoder.conv1.weight.dtype),
                                             shallow_visual, deep_visual)
        text_features = self.text_encoder(text, self.prompt_learner.tokenized_prompts, deep_text)
        return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1)

    def training_outputs(self, image):
        image_features, text_features = self.prompted_features(image)
        with torch.no_grad():
            fixed_image = F.normalize(self.teacher_image(image.to(self.teacher_image.conv1.weight.dtype)), dim=-1)
        # Retain HiCroPL's within-branch residuals with frozen features.
        mixed_image = F.normalize(image_features + fixed_image, dim=-1)
        mixed_text = F.normalize(text_features + self.fixed_text.to(text_features.dtype), dim=-1)
        logits = self.logit_scale.exp() * mixed_image @ mixed_text.t()
        return logits, mixed_image, mixed_text, fixed_image

    def forward(self, image):
        return self.training_outputs(image)[0]


def hicropl_nofusion_loss(outputs, labels, fixed_text, weight):
    logits, image_features, text_features, fixed_image = outputs
    ce = F.cross_entropy(logits.float(), labels)
    text = 1 - F.cosine_similarity(text_features.float(), fixed_text.float(), dim=1, eps=1e-7).mean()
    image = 1 - F.cosine_similarity(image_features.float(), fixed_image.float(), dim=1, eps=1e-7).mean()
    return ce + weight * (text + image), {"ce": ce, "distill_text": text, "distill_image": image}


@TRAINER_REGISTRY.register()
class HiCroPLNoFusion(TrainerX):
    def check_cfg(self, cfg):
        options = cfg.TRAINER.HICROPL_NF
        if options.PREC not in {"fp32", "fp16", "amp"}:
            raise ValueError("Unsupported precision")
        if not cfg.MODEL.BACKBONE.NAME.startswith("ViT-"):
            raise ValueError("HiCroPLNoFusion supports CLIP ViT backbones")
        if options.TEACHER_BATCH_SIZE < 1 or options.LAMBD < 0:
            raise ValueError("Invalid teacher batch size or distillation weight")
        if options.ADAM_EPS <= 0:
            raise ValueError("ADAM_EPS must be positive")
        if cfg.MODEL.INIT_WEIGHTS:
            raise ValueError("Use --resume or --model-dir with this adapted method")

    def build_model(self):
        cfg = self.cfg
        options = cfg.TRAINER.HICROPL_NF
        if self.device.type == "cpu" and options.PREC != "fp32":
            raise ValueError("CPU execution requires TRAINER.HICROPL_NF.PREC fp32")
        backbone = load_clip_to_cpu(cfg).float().to(self.device).eval().requires_grad_(False)
        fixed_text = frozen_text_features(backbone, self.dm.dataset.classnames, options)
        if options.PREC == "fp16":
            convert_weights(backbone)
        teacher_image = copy.deepcopy(backbone.visual).requires_grad_(False).eval()
        self.model = HiCroPLNoFusionModel(cfg, self.dm.dataset.classnames, backbone,
                                        teacher_image, fixed_text).to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        if cfg.OPTIM.NAME in {"adam", "adamw", "amsgrad"}:
            # The supplied HiCroPL Dassl fork exposes Adam epsilon separately.
            # Keep its value without changing the other methods' optimizer code.
            self.optim.defaults["eps"] = options.ADAM_EPS
            for group in self.optim.param_groups:
                group["eps"] = options.ADAM_EPS
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if options.PREC == "amp" else None
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"HiCroPLNoFusion trainable parameters: {count}; both prompt mappings/proxies removed")

    def forward_backward(self, batch):
        image, labels = batch["img"].to(self.device), batch["label"].to(self.device)
        self.optim.zero_grad(set_to_none=True)
        self.model.train()
        with autocast(enabled=self.scaler is not None):
            outputs = self.model.training_outputs(image)
            loss, terms = hicropl_nofusion_loss(outputs, labels, self.model.fixed_text,
                                               self.cfg.TRAINER.HICROPL_NF.LAMBD)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite HiCroPLNoFusion loss")
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            self.optim.step()
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {"loss": loss.item(), "acc": compute_accuracy(outputs[0], labels)[0].item(),
                **{name: value.item() for name, value in terms.items()}}

    def after_epoch(self):
        if not self.cfg.TEST.NO_TEST and self.cfg.TEST.FINAL_MODEL == "best_val" and self.val_loader is None:
            raise RuntimeError("best_val requires a validation split")
        super().after_epoch()

    def load_model(self, directory, epoch=None):
        if not directory:
            raise ValueError("HiCroPLNoFusion evaluation requires --model-dir")
        filename = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        path = Path(directory) / "prompt_learner" / filename
        checkpoint = load_checkpoint(str(path))
        # Only independent prompts are stored; class embeddings are rebuilt.
        self.model.prompt_learner.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"Loaded HiCroPLNoFusion prompts from {path} (epoch={checkpoint['epoch']})")
