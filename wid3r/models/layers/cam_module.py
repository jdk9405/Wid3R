import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FlashAttentionRope, FlashCrossAttentionRope
from ..dinov2.layers import Mlp
from ..dinov2.layers import LayerScale
from .upsample import ResUpsampleBil
from cam_utils.coordinate import flat_interpolate
from cam_utils.positional_embedding import generate_fourier_features

from einops import rearrange

import ipdb


class AngularModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        init_values=None,
    ):
        super().__init__()
        self.pin_params = 3
        self.deg1_params = 3
        self.deg2_params = 5
        self.deg3_params = 7
        self.num_params = self.pin_params + self.deg1_params + self.deg2_params + self.deg3_params      # 18

        self.aggregate1 = FlashAttentionRope(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=True,
                proj_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope=rope,
                )
        self.aggregate2 = FlashAttentionRope(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=True,
                proj_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope=rope,
                )

        self.latents_pos = nn.Parameter(torch.randn(1, self.num_params, hidden_dim), requires_grad=True)
        self.in_features = nn.Identity()

        self.project_pin = nn.Linear(hidden_dim, self.pin_params * hidden_dim, bias=False)
        self.project_deg1 = nn.Linear(hidden_dim, self.deg1_params * hidden_dim, bias=False)
        self.project_deg2 = nn.Linear(hidden_dim, self.deg2_params * hidden_dim, bias=False)
        self.project_deg3 = nn.Linear(hidden_dim, self.deg3_params * hidden_dim, bias=False)

        self.out_pinhole = Mlp(hidden_dim, hidden_dim, out_features=1, drop=proj_drop, bias=proj_bias)
        self.out_deg1 = Mlp(hidden_dim, hidden_dim, out_features=3, drop=proj_drop, bias=proj_bias)
        self.out_deg2 = Mlp(hidden_dim, hidden_dim, out_features=3, drop=proj_drop, bias=proj_bias)
        self.out_deg3 = Mlp(hidden_dim, hidden_dim, out_features=3, drop=proj_drop, bias=proj_bias)    

        self.mlp1 = Mlp(hidden_dim, hidden_dim, out_features=hidden_dim, drop=proj_drop, bias=proj_bias)
        self.mlp2 = Mlp(hidden_dim, hidden_dim, out_features=hidden_dim, drop=proj_drop, bias=proj_bias)
        self.ls1 = LayerScale(hidden_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls1_1 = LayerScale(hidden_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls1_2 = LayerScale(hidden_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(hidden_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2_1 = LayerScale(hidden_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2_2 = LayerScale(hidden_dim, init_values=init_values) if init_values else nn.Identity()


    def fill_intrinsics(self, x):
        hfov, cx, cy = x.unbind(dim=-1)
        hfov = torch.sigmoid(hfov - 1.1)    # 1.1 magic number s.t. hfov = pi/2 for x = 0
        ratio = self.shapes[0] / self.shapes[1]
        vfov = hfov * ratio
        cx = torch.sigmoid(cx)
        cy = torch.sigmoid(cy)
        correction_tensor = torch.tensor([2 * torch.pi, 2 * torch.pi, self.shapes[1], self.shapes[0]], device=x.device, dtype=x.dtype,)

        intrinsics = torch.stack([hfov, vfov, cx, cy], dim=1)
        intrinsics = correction_tensor.unsqueeze(0) * intrinsics
        return intrinsics

    def forward(self, cls_tokens) -> torch.Tensor:
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)

        pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens = cls_tokens.chunk(4, dim=1)
        # [b, 1, 1024] x 4
        pin_tokens = rearrange(self.project_pin(pin_tokens), "b n (h c) -> b (n h) c", h=self.pin_params)
        # [b, 3, 1024]
        deg1_tokens = rearrange(self.project_deg1(deg1_tokens), "b n (h c) -> b (n h) c", h=self.deg1_params)
        # [b, 3, 1024]
        deg2_tokens = rearrange(self.project_deg2(deg2_tokens), "b n (h c) -> b (n h) c", h=self.deg2_params)
        # [b, 5, 1024]
        deg3_tokens = rearrange(self.project_deg3(deg3_tokens), "b n (h c) -> b (n h) c", h=self.deg3_params)
        # [b, 7, 1024]
        tokens = torch.cat([pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens], dim=1)
        # [b, 18, 1024]

        tokens = self.ls1(self.mlp1(self.ls1_1(self.aggregate1(tokens, xpos=latents_pos)) + self.ls1_2(tokens))) + tokens
        tokens = self.ls2(self.mlp2(self.ls2_1(self.aggregate2(tokens, xpos=latents_pos)) + self.ls2_2(tokens))) + tokens     # [b, 18, 1024]

        tokens_pinhole, tokens_deg1, tokens_deg2, tokens_deg3 = torch.split(
            tokens, [self.pin_params, self.deg1_params, self.deg2_params, self.deg3_params], dim=1,
        )
        x = self.out_pinhole(tokens_pinhole).squeeze(-1)            # [b, 3, 1]
        d1 = self.out_deg1(tokens_deg1)
        d2 = self.out_deg2(tokens_deg2)
        d3 = self.out_deg3(tokens_deg3)

        camera_intrinsics = self.fill_intrinsics(x)
        return camera_intrinsics, torch.cat([d1, d2, d3], dim=1)

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes



class RadialModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        init_values=None,

        depths: int | list[int] = 4,
        kernel_size: int = 7,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.ups = nn.ModuleList([])
        self.process_features = nn.ModuleList([])
        mult = 2
        self.to_latents = nn.Linear(hidden_dim, hidden_dim)

        self.prompt_camera = FlashAttentionRope(
                dim=hidden_dim,
                num_heads=num_heads,
                qkv_bias=True,
                proj_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope=rope,
                )

        for i, depth in enumerate(depths):
            current_dim = min(hidden_dim, mult * hidden_dim // int(2**i))
            next_dim = mult * hidden_dim // int(2 ** (i + 1))
            output_dim = max(next_dim, out_dim)
            self.process_features.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    current_dim,
                    kernel_size=max(1, 2 * i),
                    stride=max(1, 2 * i),
                    padding=0,
                )
            )
            self.ups.append(
                ResUpsampleBil(
                    current_dim,
                    output_dim=output_dim,
                    expansion=1,
                    layer_scale=init_values,
                    kernel_size=kernel_size,
                    num_layers=depth,
                    use_norm=False,
                )
            )
        self.depth_mlp = nn.Sequential(
            nn.LayerNorm(next_dim), 
            nn.Linear(next_dim, output_dim)
        )


        self.confidence_mlp = nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))

        self.to_depth_lr = nn.Conv2d(output_dim, output_dim // 2, kernel_size=3, padding=1, padding_mode="reflect")
        self.to_confidence_lr = nn.Conv2d(output_dim, output_dim // 2, kernel_size=3, padding=1, padding_mode="reflect")
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.to_confidence_hr = nn.Sequential(
            nn.Conv2d(output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def embed_rays(self, rays):
        rays_embedding = flat_interpolate(rays, old=self.original_shapes, new=self.shapes, antialias=True)
        rays_embedding = rays_embedding / torch.norm(rays_embedding, dim=-1, keepdim=True).clip(min=1e-4)
        x, y, z = rays_embedding[..., 0], rays_embedding[..., 1], rays_embedding[..., 2]
        polar = torch.acos(z)
        x_clipped = x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1)
        azimuth = torch.atan2(y, x_clipped)
        rays_embedding = torch.stack([polar, azimuth], dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.hidden_dim,
            max_freq=max(self.shapes) // 2,
            use_log=True,
            cat_orig=False,
        )
        return rays_embedding

    def condition(self, feat, rays_embedding):
        conditioned_features = self.prompt_camera(rearrange(feat, "b h w c -> b (h w) c"), rays_embedding)
        return conditioned_features

    def process(self, features_list, rays_embedding):
        conditioned_features = self.condition(features_list, rays_embedding)
        init_latents = self.to_latents(conditioned_features)
        init_latents = rearrange(init_latents, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]).contiguous()
        conditioned_features = rearrange(conditioned_features, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]).contiguous()
        latents = init_latents

        for i, up in enumerate(self.ups):
            latents = latents + self.process_features[i](conditioned_features)
            latents = up(latents)

        return latents

    def depth_proj(self, features):
        h_out, w_out = features.shape[-2:]
        # aggregate output and project to depth
        out_depth_features = self.depth_mlp(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out_depth_features = F.interpolate(out_depth_features, size=(h_out, w_out), mode="bilinear", align_corners=True)
        # [S, 256, h_out, w_out]
        logdepth = self.to_depth_lr(out_depth_features)
        logdepth = F.interpolate(logdepth, size=self.original_shapes, mode="bilinear", align_corners=True)
        logdepth = self.to_depth_hr(logdepth)
        # [S, 1, H, W]
        return logdepth

    def confidence_proj(self, features):
        features = features.permute(0, 2, 3, 1)
        confidence = self.confidence_mlp(features).permute(0, 3, 1, 2)
        confidence = self.to_confidence_lr(confidence)
        confidence = F.interpolate(confidence, size=self.original_shapes, mode="bilinear", align_corners=True)
        confidence = self.to_confidence_hr(confidence)
        return confidence       # [S, 1, H, W]

    def decode(self, features):
        logdepth = self.depth_proj(features)
        confidence = self.confidence_proj(features)
        return logdepth, confidence

    def forward(self, features: list[torch.Tensor], rays_hr: torch.Tensor,) -> torch.Tensor:
        rays_embedding = self.embed_rays(rays_hr)           # [b, 925, 1024]
        features = self.process(features, rays_embedding)
        logdepth, logconf = self.decode(features)
        return logdepth, logconf, rays_embedding
