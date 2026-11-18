"""HiCroPL's external deep-prompt encoders, with no cross-modal mapping.

Adapted from the supplied HiCroPL CLIP implementation. Precision follows the
input rather than forcing fp16. See licenses/HiCroPL-LICENSE.txt.
"""
import torch
from torch import nn
from .model import VisionTransformer


class IndependentTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        for name in ["transformer", "positional_embedding", "ln_final", "text_projection"]:
            setattr(self, name, getattr(clip_model, name))

    def forward(self, prompts, tokens, deep_prompts):
        dtype = self.text_projection.dtype
        n_ctx = deep_prompts[0].shape[0] if deep_prompts else None
        x = (prompts.to(dtype) + self.positional_embedding.to(dtype)).permute(1, 0, 2)
        for index, block in enumerate(self.transformer.resblocks):
            if 1 <= index <= len(deep_prompts):
                ctx = deep_prompts[index - 1].to(x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x[:1], ctx, x[1 + n_ctx:]], dim=0)
            x = block(x)
        x = self.ln_final(x.permute(1, 0, 2)).to(dtype)
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, tokens.argmax(dim=-1)] @ self.text_projection


class IndependentVisionEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not isinstance(original, VisionTransformer):
            raise ValueError("HiCroPLNoFusion requires a CLIP ViT backbone")
        self.input_resolution = original.input_resolution
        self.output_dim = original.output_dim
        for name in ["conv1", "class_embedding", "positional_embedding", "ln_pre",
                     "transformer", "ln_post", "proj"]:
            setattr(self, name, getattr(original, name))

    def forward(self, images, shallow_prompt, deep_prompts):
        x = self.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        n_ctx = shallow_prompt.shape[0]
        ctx = shallow_prompt.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self.ln_pre(torch.cat([x, ctx], dim=1)).permute(1, 0, 2)
        for index, block in enumerate(self.transformer.resblocks):
            if 1 <= index <= len(deep_prompts):
                ctx = deep_prompts[index - 1].to(x.dtype).unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x[:-n_ctx], ctx], dim=0)
            x = block(x)
        return self.ln_post(x.permute(1, 0, 2)[:, 0]) @ self.proj
