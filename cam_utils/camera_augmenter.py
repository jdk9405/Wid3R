"""
inspired by https://github.com/lpiccinelli-eth/UniK3D
"""

import numpy as np
import torch
from einops import rearrange

from .camera import CameraSampler
from .camera import BatchCamera
from .camera import EUCM, Fisheye624, OPENCV, MEI, Pinhole, Spherical
name2cls = {
    "EUCM": EUCM,
    "Fisheye624": Fisheye624,
    "OPENCV": OPENCV,
    "MEI": MEI,
    "Pinhole": Pinhole,
    "Spherical": Spherical
}
from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__, __LOW_QUALITY_DATASETS__
from .coordinate import coords_grid

import ipdb

try:
    from splatting import splatting_function
except Exception as e:
    splatting_function = None
    print(
        f"Splatting not available, please install it from github.com/hperrot/splatting"
    )

@torch.jit.script
def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    device = mask1.device

    # Ensure the masks are binary (0 or 1)
    mask1 = mask1.to(torch.bool)
    mask2 = mask2.to(torch.bool)

    # Compute intersection and union
    intersection = torch.sum(mask1 & mask2).to(torch.float32)
    union = torch.sum(mask1 | mask2).to(torch.float32)

    # Compute IoU
    iou = intersection / union.clip(min=1.0)

    return iou

def fill(rgb, mask):
    def fill_noise(size, device):
        return torch.rand(size, device=device)

    def fill_black(size, device):
        return torch.zeros(size, device=device, dtype=torch.float32)

    def fill_white(size, device):
        return 1 * torch.ones(size, device=device, dtype=torch.float32)

    def fill_zero(size, device):
        return torch.zeros(size, device=device, dtype=torch.float32)

    B, C = rgb.shape[:2]
    validity_mask = mask.repeat(1, C, 1, 1).bool()
    filler_fn = fill_black
    for i in range(B):
        rgb[i][~validity_mask[i]] = filler_fn(
            size=rgb[i][~validity_mask[i]].shape, device=rgb.device
        )
    return rgb


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32)
def optional_augment_camera(batch, dataset_names, camera_sampler):

    rgb = torch.stack([view['img'] for view in batch], dim=1)      # [b(3), v(19), 3, 224, 224]
    B, N, _, H, W = rgb.shape
    rgb = rgb.reshape(B*N, 3, H, W)
    depth = torch.stack([view['depthmap'] for view in batch], dim=1).reshape(B*N, 1, H, W)         # [BN, 1, H, W]
    validity_mask = torch.stack([view['valid_mask'] for view in batch], dim=1).reshape(B*N, 1, H, W)  # [BN, 1, H, W]
    cameras = torch.cat([batch[i1]['camera'][i0] for i0 in range(B) for i1 in range(N)])
    cameras = cameras.to(rgb.device)

    do_augment = torch.rand(1) > 0.75
    # do_augment = True   # debug
    # do_augment = False
    if not do_augment:
        return rgb, depth, cameras, validity_mask

    guidance = depth.clone()
    dtype, device = depth.dtype, depth.device
    BN, C, H, W = rgb.shape
    id_coords = coords_grid(BN, H, W, device=device)     # [b, 2, h, w] 

    is_pinhole = np.array(cameras.original_class) == 'Pinhole'    # [b]
    is_highquality = np.array([name in __HIGH_QUALITY_DATASETS__ for name in dataset_names])
    is_pinhole = is_pinhole & is_highquality
    if not is_pinhole.any():
        do_augment = False
        return rgb, depth, cameras, validity_mask

    selected_rgb = rgb[is_pinhole]       # [b', 3, h, w]    
    selected_depth = depth[is_pinhole]   # [b', 1, h, w]
    selected_guidance = guidance[is_pinhole]   # [b', 1, h, w]
    pinhole_cameras = torch.cat([cameras[i] for i in range(len(cameras)) if is_pinhole[i]])
    selected_validity_mask = validity_mask[is_pinhole]   # [b', 1, h, w]
    selected_id_coords = id_coords[is_pinhole]   # [b', 2, h, w]

    fovs = torch.tensor(max(pinhole_cameras.hfov, pinhole_cameras.vfov)) * 180 / np.pi      # [b']
    ratios = (70.0 / fovs).clip(max=1.0).to(device)     # decrease effect for larger fov               # [b']
    if (fovs < 40.0).any():  # skips ~5%
        do_augment = False

    if (selected_depth < 0.0).any():
        do_augment = False

    selected_depth[~selected_validity_mask] = selected_depth.max() * 2.0

    fx, fy, cx, cy = pinhole_cameras.params[:, :4].unbind(dim=-1)
    new_cameras = camera_sampler(fx, fy, cx, cy, mult=1.0, ratio=ratios, H=H)
    unprojected = pinhole_cameras.reconstruct(selected_depth)     # [b', 3, h, w]
    projected = new_cameras.project(unprojected)         # [b', 2, h, w]
    projection_mask = new_cameras.projection_mask        # [b', 1, h, w]
    overlap_mask = (
        new_cameras.overlap_mask
        if new_cameras.overlap_mask is not None
        else torch.ones_like(projection_mask)
    )                                                   # [b', 1, h, w]
    mask = selected_validity_mask & overlap_mask                 # [b', 1, h, w]

    # if it is actually going out, we need to remember the regions
    # remember when the tengetial distortion was keeping the validaty_mask border after re-warpingi
    # need a better way to define overlap class, in case of vortex style if will mask wrong parts...
    # also is_collapse does not take into consideration when we have vortex effect,
    # how can we avoid vortex in the first place????
    is_collapse = (projected[:, 1, 0, :] >= 0.0).all(dim=1)  # [b]
    if is_collapse.any():
        projected[is_collapse][~mask.repeat(1, 2, 1, 1)[is_collapse]] = selected_id_coords[is_collapse][~mask.repeat(1, 2, 1, 1)[is_collapse]]
    flow = projected - selected_id_coords        # [b', 2, h, w]        
    selected_depth[~mask] = selected_depth.max() * 2.0

    if flow.norm(dim=1).median() / max(H, W) > 0.1:  # extreme cases
        do_augment = False

    # warp via soft splat
    depth_image = torch.cat([selected_rgb, selected_guidance, mask], dim=1)
    depth_image = splatting_function(
        "softmax", depth_image, flow, -torch.log(1 + selected_depth.clip(0.01))
    )
    selected_rgb_warp = depth_image[:, :3]
    selected_guidance_warp = depth_image[:, 3:4]
    selected_validity_mask_new = depth_image[:, -1:] > 0.0

    expanding = selected_validity_mask_new.sum() > selected_validity_mask.sum()
    threshold = 0.7 if expanding else 0.25
    _iou = iou(selected_validity_mask_new, selected_validity_mask)
    if _iou < threshold:  # too strong augmentation, lose most of the image
        do_augment = False

    # where it goes out
    mask_unwarpable = projection_mask & overlap_mask     # [b', 1, h, w]
    selected_validity_mask = selected_validity_mask & mask_unwarpable     # [b', 1, h, w]

    if do_augment:
        aug_mask = selected_validity_mask & mask & selected_validity_mask_new     # [b', 1, h, w]
        aug_rgb = fill(selected_rgb_warp, aug_mask)     # [b', 3, h, w]
        aug_depth = selected_guidance_warp     # [b', 1, h, w]
        aug_depth[~aug_mask] = 0

        rgb[is_pinhole] = aug_rgb.to(rgb)
        depth[is_pinhole] = aug_depth.to(depth)
        validity_mask[is_pinhole] = aug_mask.to(validity_mask)

        new_cameras_names = [new_cameras.__class__.__name__] * len(pinhole_cameras)
        new_cameras = torch.cat([name2cls[name](new_cameras.params[i:i+1]) for i, name in enumerate(new_cameras_names)])

        aug_rgb = aug_rgb.reshape(-1, N, 3, H, W)
        aug_depth = aug_depth.reshape(-1, N, H, W)
        aug_mask = aug_mask.reshape(-1, N, H, W)

        j = 0
        for i, stat in enumerate(is_pinhole):
            if stat:
                cameras[i] = new_cameras[j]
                j += 1

        is_pinhole = np.where(is_pinhole.reshape(B, N).all(axis=-1))[0]
        temp_cameras = np.array([cam for cam in cameras]).reshape(B, N).T.reshape(-1)
        for j in range(N):
            # try:
            batch[j]['img'][is_pinhole] = aug_rgb[:, j].to(rgb)
            batch[j]['depthmap'][is_pinhole] = aug_depth[:, j].to(depth)
            batch[j]['camera'] = [cam for cam in temp_cameras[j*B:(j+1)*B]]
            batch[j]['valid_mask'][is_pinhole] = aug_mask[:, j].to(validity_mask)
            # except:
            #     ipdb.set_trace()
            # if batch[j]['depthmap'].dtype != torch.float32:
            #     ipdb.set_trace()

    return rgb, depth, cameras, validity_mask
        
        
    




@torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32)
def augment_camera(rgb, depth, cameras, validity_mask, camera_sampler):
    guidance = depth.clone()
    dtype, device = depth.dtype, depth.device
    B, C, H, W = rgb.shape

    augment_indices = torch.rand(B, 1, 1, device=device, dtype=dtype) > 0.9
    id_coords = coords_grid(B, H, W, device=device)     # [b, 2, h, w] 
    # get rescaled depth
    augment_indices = augment_indices.reshape(-1)

    final_rgb = rgb.clone()
    final_depth = depth.clone()
    final_mask = validity_mask.clone()
    for i, is_augment in enumerate(augment_indices):
        if not is_augment:
            continue

        # pinhole_camera = inputs["camera"][i]
        pinhole_camera = cameras[i]
        fov = max(pinhole_camera.hfov[0], pinhole_camera.vfov[0]) * 180 / np.pi
        ratio = min(70.0 / fov, 1.0)  # decrease effect for larger fov
        if fov < 40.0:  # skips ~5%
            augment_indices[i] = False
            continue

        rgb_i = rgb[i : i + 1]
        id_coords_i = id_coords[i : i + 1]

        validity_mask_i = validity_mask[i : i + 1]
        depth = guidance[i : i + 1]

        if (depth < 0.0).any():
            augment_indices[i] = False
            continue

        depth = depth.sqrt()  # why sqrt??
        depth[~validity_mask_i] = depth.max() * 2.0

        fx, fy, cx, cy = pinhole_camera.params[:, :4].unbind(dim=-1)
        new_camera = camera_sampler(fx, fy, cx, cy, mult=1.0, ratio=ratio, H=H)
        unprojected = pinhole_camera.reconstruct(depth)     # [b, 3, h, w]
        projected = new_camera.project(unprojected)         # [b, 2, h, w]
        projection_mask = new_camera.projection_mask        # [b, 1, h, w]
        overlap_mask = (
            new_camera.overlap_mask
            if new_camera.overlap_mask is not None
            else torch.ones_like(projection_mask)
        )                                                   # [b, 1, h, w]
        mask = validity_mask_i & overlap_mask               # [b, 1, h, w]

        # if it is actually going out, we need to remember the regions
        # remember when the tengetial distortion was keeping the validaty_mask border after re-warpingi
        # need a better way to define overlap class, in case of vortex style if will mask wrong parts...
        # also is_collapse does not take into consideration when we have vortex effect,
        # how can we avoid vortex in the first place????
        is_collapse = (projected[0, 1, 0, :] >= 0.0).all()
        if is_collapse:
            projected[~mask.repeat(1, 2, 1, 1)] = id_coords_i[~mask.repeat(1, 2, 1, 1)]
        flow = projected - id_coords_i
        depth[~mask] = depth.max() * 2.0

        if flow.norm(dim=1).median() / max(H, W) > 0.1:  # extreme cases
            augment_indices[i] = False
            continue

        # warp via soft splat
        depth_image = torch.cat([rgb_i, guidance[i : i + 1], mask], dim=1)
        depth_image = splatting_function(
            "softmax", depth_image, flow, -torch.log(1 + depth.clip(0.01))
        )
        rgb_warp = depth_image[:, :3]
        guidance_warp = depth_image[:, 3:4]
        validity_mask_i = depth_image[:, -1:] > 0.0

        expanding = validity_mask_i.sum() > validity_mask[i : i + 1].sum()
        threshold = 0.7 if expanding else 0.25
        _iou = iou(validity_mask_i, validity_mask[i : i + 1])
        if _iou < threshold:  # too strong augmentation, lose most of the image
            augment_indices[i] = False
            continue

        # where it goes out
        mask_unwarpable = projection_mask & overlap_mask
        validity_mask[i] = validity_mask[i] & mask_unwarpable.squeeze(0)

        final_mask[i] = validity_mask[i] & mask & validity_mask_i
        final_rgb[i] = fill(rgb_warp, final_mask[i])[0]
        final_depth[i] = guidance_warp[0]
        final_depth[i][~final_mask[i]] = 0
        cameras[i] = new_camera

    return final_rgb, final_depth, cameras, final_mask


if __name__ == "__main__":

    from .camera import CameraSampler

    camera_sampler = CameraSampler(file_path="cam_utils/camera_sampler.json")

    augment_camera(None, None, camera_sampler)