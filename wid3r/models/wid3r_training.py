import sys
sys.path.append('.')

import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

from .layers.cam_module import AngularModule, RadialModule
from .layers.grad_choker import GradChoker

from cam_utils.coordinate import coords_grid
from cam_utils.sht import rsh_cart_3
from einops import rearrange

import ipdb

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False

class Wid3R(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            use_camera_gt=False,
            ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError


        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
                ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        # self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)


        # ----------------------
        #   Camera Model token
        # ----------------------
        num_camera_model_tokens = 4

        # camera model token for each camera model
        # 0: pinhole, 1: eucm, 2: fisheye, 3: spherical, 4: mei, 5: opencv
        self.camera_model_token = nn.Parameter(torch.randn(6, num_camera_model_tokens, self.dec_embed_dim))

        self.camera_model_token_adapter1 = nn.Linear(self.dec_embed_dim*2, self.dec_embed_dim)
        self.camera_model_token_adapter2 = nn.Linear(self.dec_embed_dim*2, self.dec_embed_dim)

        self.patch_start_idx = num_register_tokens + num_camera_model_tokens


        # ----------------------
        #  Local Points Decoder
        # ----------------------
        if False:
            self.point_decoder = TransformerDecoder(
                    in_dim=2*self.dec_embed_dim,            # 1024 x 2
                    dec_embed_dim=1024,
                    dec_num_heads=16,
                    out_dim=1024,
                    rope=self.rope,
                    )
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Ray modules
        # ----------------------
        init_values = 0.01
        self.ray_module = AngularModule(hidden_dim=self.dec_embed_dim, init_values=init_values,) # rope=self.rope)
        self.distance_module = RadialModule(hidden_dim=self.dec_embed_dim, 
                                            depths=[2, 2, 2], 
                                            out_dim=64, 
                                            kernel_size=3, 
                                            init_values=init_values,
                                            )
        self.choker = GradChoker(alpha=0.1)
        self.use_camera_gt = use_camera_gt


        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,                # 8
                out_dim=512,
                rope=self.rope,
                use_checkpoint=False
                )
        self.camera_head = CameraHead(dim=512)


        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points      # False
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                    in_dim=2*self.dec_embed_dim, 
                    dec_embed_dim=1024,
                    dec_num_heads=16,
                    out_dim=1024,
                    rope=self.rope,
                    )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if load_vggt:
            # vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            # vggt_weight = load_file('./ckpts/model.safetensors')
            vggt_weight = load_file('/fs/gamma-projects/ANYCAM2/workspace/Wid3R/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        self.train_conf = train_conf            # False

        if freeze_encoder:      # False
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint      # 4

        if ckpt is not None:

            if ckpt.endswith('.safetensors'):
                checkpoint = load_file(ckpt, device='cpu')
            else:
                checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[Wid3R] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()

    def decode(self, hidden, N, H, W, camera_names):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])   # [B*N, 5, 1024]

        camera_model_mask = []
        # 0: pinhole, 1: eucm, 2: fisheye, 3: spherical, 4: mei, 5:opencv
        for camera_name in camera_names:
            if camera_name == 'Pinhole':
                camera_model_mask.append(0)         # [1, 4, 1024]
            elif camera_name == 'EUCM':
                camera_model_mask.append(1)
            elif camera_name == 'Fisheye624':
                camera_model_mask.append(2)
            elif camera_name == 'Spherical':
                camera_model_mask.append(3)
            elif camera_name == 'MEI':
                camera_model_mask.append(4)
            elif camera_name == 'OPENCV':
                camera_model_mask.append(5)
            else:
                NotImplementedError(f"Camera model {camera_name} is not implemented")
        assert len(camera_model_mask) == B*N
        camera_model_token = self.camera_model_token[camera_model_mask].reshape(B*N, *self.camera_model_token.shape[-2:])   # [B*N, 4, 1024]

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([camera_model_token, register_token, hidden], dim=1)  # [B*N, 4+5+256, 1024]
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))        # [B*N, 265, 1024]

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    # [B*N, 265, 2048], [B*N, 265, 2]

    def forward(self, imgs, cameras=None, where_use_gt_rays=None,):

        predictions = {}

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode by dinov3
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        torch.cuda.empty_cache()

        camera_names = [cameras[i].__class__.__name__ for i in range(len(cameras))]
        hidden, pos = self.decode(hidden, N, H, W, camera_names)
        # hidden: [B*N, 265, 2048], pos: [B*N, 265, 2]


        if cameras is not None:

            cameras = cameras.to(hidden.device)

            """
            ray module. start.
            """
            # TODO: make options for selecting our module or the original one.
            camera_model_token = hidden[..., :4, :]

            with torch.amp.autocast(device_type='cuda', enabled=False):
                camera_model_token = self.choker(camera_model_token)                            # [B*N, 4, 2048]
                camera_model_token = self.camera_model_token_adapter1(camera_model_token)       # [B*N, 4, 1024]
                self.ray_module.set_shapes((H, W))

                intrinsics, sh_coeffs = self.ray_module(cls_tokens=camera_model_token)
                # intrinsics: [B*N, 4], sh_coeffs: [B*N, 15, 3]
                # del camera_model_token

                # id_coords = coords_grid(B*N, H, W, device=sh_coeffs.device)                     # [B*N, 2, H, W]
                id_coords = coords_grid(1, H, W, device=sh_coeffs.device)                       # [1, 2, H, W]

                # This is fov based
                longitude = (id_coords[:, 0] - intrinsics[:, 2].view(-1, 1, 1)) / W * intrinsics[:, 0].view(-1, 1, 1)   # [B*N, H, W]
                latitude = (id_coords[:, 1] - intrinsics[:, 3].view(-1, 1, 1)) / H * intrinsics[:, 1].view(-1, 1, 1)    # [B*N, H, W]

                x = torch.cos(latitude) * torch.sin(longitude)          # [B*N, H, W]
                y = -torch.sin(latitude)                                # [B*N, H, W]
                z = torch.cos(latitude) * torch.cos(longitude)          # [B*N, H, W]
                unit_sphere = torch.stack([x, y, z], dim=-1)            # [B*N, H, W, 3]
                unit_sphere = unit_sphere / torch.norm(unit_sphere, dim=-1, keepdim=True).clip(min=1e-5)

                harmonics = rsh_cart_3(unit_sphere)[..., 1:]            # remove constant-value harmonic
                # [B*N, H, W, 15]

                rays_pred = torch.einsum("bhwc, bcd->bhwd", harmonics, sh_coeffs)                       # [B*N, H, W, 3]
                rays_pred = rays_pred / torch.norm(rays_pred, dim=-1, keepdim=True).clip(min=1e-5)
                rays_pred = rays_pred.permute(0, 3, 1, 2)                                  # [B*N, 3, H, W]
                rays_pred = rearrange(rays_pred, "b c h w -> b (h w) c")          # [B*N, H*W, 3] 

                rays_gt = cameras.get_rays((B*N, H, W)).view(B*N, 3, -1).contiguous().permute(0, 2, 1)  # [B*N, H*W, 3]
                # del harmonics, sh_coeffs, unit_sphere

                # should clean also nans
                if self.training:
                    if where_use_gt_rays is not None:
                        where_use_gt_rays = where_use_gt_rays.int()
                        rays = rays_gt * where_use_gt_rays + rays_pred * (1 - where_use_gt_rays)
                    else:
                        rays = rays_pred
                elif self.use_camera_gt:
                    rays = rays_gt if rays_gt is not None else rays_pred
                else:
                    rays = rays_pred


            predictions["cam_rays"] = rays
            predictions["cam_rays_pred"] = rays_pred
            predictions["cam_rays_gt"] = rays_gt.detach()        # GT
            """
            ray module. end.
            """

            camera_pose_hidden = self.camera_decoder(hidden, xpos=pos)

            """
            distance module. start.
            """
            with torch.amp.autocast(device_type='cuda', enabled=False):
                hidden_ray = self.camera_model_token_adapter2(rearrange(hidden, "(b n) p c -> b n p c", b=B, n=N))

                common_shape = (H // self.patch_size, W // self.patch_size)
                self.distance_module.set_shapes(common_shape)
                self.distance_module.set_original_shapes((H, W))
                B, N, P, C = hidden_ray.shape
                patch_hidden_ray = hidden_ray.contiguous().view(B*N, P, C)[:, self.patch_start_idx:, :].reshape(B*N, common_shape[0], common_shape[1], C)

                logradius, loguncertainty, _ = self.distance_module(features=patch_hidden_ray, rays_hr=rays.detach())

                radius = torch.exp(logradius.clip(min=-8.0, max=8.0) + 2.0)
                uncertainty = torch.exp(loguncertainty.clip(min=-8.0, max=10.0))                  # [B*N, 1, H, W]

                predictions["logradius"] = logradius
                predictions["radius"] = radius

                rays = rearrange(rays, "b (h w) c -> b c h w", h=H, w=W)                        # [B*N, 3, H, W]
                local_points = rays * radius                                                    # [B*N, 3, H, W]
                local_points = rearrange(local_points, "(b n) c h w -> b n h w c", b=B, n=N)    # [B, N, H, W, 3]

                camera_pose_hidden = camera_pose_hidden.float()
                camera_poses = self.camera_head(camera_pose_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
                
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
                uncertainty = rearrange(uncertainty, "(b n) c h w -> b n h w c", b=B, n=N)              # [B, N, H, W, 3]
                global_points = None
                conf = None

                del rays, radius, camera_pose_hidden


        else:

            point_hidden = self.point_decoder(hidden, xpos=pos)
            if self.train_conf:     # False
                conf_hidden = self.conf_decoder(hidden, xpos=pos)
            camera_hidden = self.camera_decoder(hidden, xpos=pos)
            if self.use_global_points:
                context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
                global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

            with torch.amp.autocast(device_type='cuda', enabled=False):
                # local points
                point_hidden = point_hidden.float()
                ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                xy, z = ret.split([2, 1], dim=-1)
                z = torch.exp(z)
                local_points = torch.cat([xy * z, z], dim=-1)

                # confidence
                if self.train_conf:
                    conf_hidden = conf_hidden.float()
                    conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                else:
                    conf = None
                    
                # camera
                camera_hidden = camera_hidden.float()
                camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

                # Global points
                if self.use_global_points:
                    global_point_hidden = global_point_hidden.float()
                    global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                else:
                    global_points = None
                
                # unproject local points using camera poses
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

                uncertainty=None


        predictions["points"] = points
        predictions["local_points"] = local_points
        predictions["conf"] = conf
        predictions["uncertain"] = uncertainty
        predictions["camera_poses"] = camera_poses 
        predictions["global_points"] = global_points

        return predictions
