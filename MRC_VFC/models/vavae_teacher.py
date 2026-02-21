import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nonlinearity(x):
    return x * torch.sigmoid(x)


def _normalize(in_channels, num_groups=32):
    groups = min(num_groups, in_channels)
    while groups > 1 and in_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6, affine=True)


class _Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.with_conv:
            # Same padding style as LDM/AutoencoderKL.
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.avg_pool(x)
        return x


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.conv1(_nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(_nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k) * (c ** -0.5)
        w_ = torch.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b, hw, hw
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class _Encoder(nn.Module):
    """
    LDM-style encoder used as a pragmatic VA-VAE teacher adapter.
    """

    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        z_channels=32,
        dropout=0.0,
        attn_levels=(4,),
        resamp_with_conv=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(_ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(_AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in, dropout=dropout)

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(_nonlinearity(self.norm_out(h)))
        return h


def _parse_int_seq(value, default: Tuple[int, ...]):
    if value is None:
        return tuple(default)
    if isinstance(value, (tuple, list)):
        out = [int(v) for v in value]
        return tuple(out) if out else tuple(default)
    txt = str(value).strip()
    if not txt:
        return tuple(default)
    parts = [p.strip() for p in txt.split(",")]
    out = [int(p) for p in parts if p]
    return tuple(out) if out else tuple(default)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "module", "ema", "params_ema"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        tensor_like = [k for k, v in ckpt_obj.items() if torch.is_tensor(v)]
        if tensor_like:
            return ckpt_obj
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    raise ValueError("Unsupported checkpoint format. Expected dict/state_dict or nn.Module.")


def _strip_prefixes(key: str, prefixes: Iterable[str]):
    out = key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p):]
                changed = True
    return out


class VAVAETeacherEncoder(nn.Module):
    """
    Lightweight adapter for VA-VAE checkpoints.
    Returns pooled latent feature vectors for KD.
    """

    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult="1,1,2,2,4",
        num_res_blocks=2,
        z_channels=32,
        attn_levels="4",
        input_size=256,
        resize_input=False,
        pool="avg",
        feature_from="mu",
    ):
        super().__init__()
        ch_mult_tuple = _parse_int_seq(ch_mult, default=(1, 1, 2, 2, 4))
        attn_levels_tuple = _parse_int_seq(attn_levels, default=(len(ch_mult_tuple) - 1,))
        self.input_size = int(input_size)
        self.resize_input = bool(resize_input)
        self.pool = str(pool).lower()
        self.feature_from = str(feature_from).lower()
        self.z_channels = int(z_channels)

        self.encoder = _Encoder(
            in_channels=in_channels,
            ch=int(ch),
            ch_mult=ch_mult_tuple,
            num_res_blocks=int(num_res_blocks),
            z_channels=self.z_channels,
            attn_levels=attn_levels_tuple,
        )
        self.quant_conv = nn.Conv2d(2 * self.z_channels, 2 * self.z_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.resize_input and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if moments.size(1) % 2 == 0:
            mu, logvar = torch.chunk(moments, 2, dim=1)
            feat_map = mu if self.feature_from in ("mu", "mean") else moments
        else:
            feat_map = moments

        if self.pool == "flat":
            return feat_map.flatten(1)
        if self.pool == "max":
            return F.adaptive_max_pool2d(feat_map, (1, 1)).flatten(1)
        return F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)

    def load_pretrained(self, ckpt_path, strict=False, partial=True, map_location="cpu") -> Dict[str, int]:
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"VA-VAE checkpoint not found: {ckpt_path}")
        ckpt_obj = torch.load(ckpt_path, map_location=map_location)
        state_dict = _extract_state_dict(ckpt_obj)

        cleaned = {}
        prefixes = (
            "module.",
            "model.",
            "autoencoder.",
            "first_stage_model.",
            "vae.",
        )
        for k, v in state_dict.items():
            nk = _strip_prefixes(str(k), prefixes)
            if nk.startswith("encoder.") or nk.startswith("quant_conv."):
                cleaned[nk] = v

        if not cleaned:
            raise ValueError("No encoder/quant_conv keys found in VA-VAE checkpoint.")

        if not partial:
            missing, unexpected = self.load_state_dict(cleaned, strict=bool(strict))
            return {
                "loaded": len(cleaned),
                "total_model_keys": len(self.state_dict()),
                "ckpt_candidate_keys": len(cleaned),
                "missing": len(missing),
                "unexpected": len(unexpected),
            }

        model_sd = self.state_dict()
        loadable = {}
        skipped = 0
        for k, v in cleaned.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                loadable[k] = v
            else:
                skipped += 1

        missing, unexpected = self.load_state_dict(loadable, strict=False)
        return {
            "loaded": len(loadable),
            "skipped": skipped,
            "total_model_keys": len(model_sd),
            "ckpt_candidate_keys": len(cleaned),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }
