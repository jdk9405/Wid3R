import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

from typing import List, Union

import ipdb

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from wid3r.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        upxleft = upxleft / (upxleft.norm(dim=-1, keepdim=True) + 1e-10)
        leftxdown = leftxdown / (leftxdown.norm(dim=-1, keepdim=True) + 1e-10)
        downxright = downxright / (downxright.norm(dim=-1, keepdim=True) + 1e-10)
        rightxup = rightxup / (rightxup.norm(dim=-1, keepdim=True) + 1e-10)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        gt_upxleft = gt_upxleft / (gt_upxleft.norm(dim=-1, keepdim=True) + 1e-10)
        gt_leftxdown = gt_leftxdown / (gt_leftxdown.norm(dim=-1, keepdim=True) + 1e-10)
        gt_downxright = gt_downxright / (gt_downxright.norm(dim=-1, keepdim=True) + 1e-10)
        gt_rightxup = gt_rightxup / (gt_rightxup.norm(dim=-1, keepdim=True) + 1e-10)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        if pred['uncertain'] is None:
            weights_ = gt_local_pts[..., 2]
        else:
            weights_ = gt_local_pts.norm(dim=-1)
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = 10. * self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:       # False
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
    # def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)                  # [B, N, 1, 4, 4]
        pred_pose_exp = pred_pose_align.unsqueeze(1)          # [B, 1, N, 4, 4]
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)                    # [B, N, 1, 4, 4]
        gt_pose_exp = gt_pose.unsqueeze(1)                  # [B, 1, N, 4, 4]
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)    # [B, N, N, 4, 4]
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)          # [B, N, N, 4, 4]

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)  # [N, N]

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Wid3RLoss(nn.Module):
    def __init__(
        self,
        train_conf=False,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)             # [B, N, H, W, 3]
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)         # [B, N, H, W]
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)        # [B, N, 4, 4]

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])                                   # [B, 4, 4]
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]    # [B, N, H, W, 3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)            # [B, N, 4, 4]

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)     # [B, N, 4, 4]
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        # [B, N, H, W, 3]
        
        dataset_names = gt[0]['dataset']
        # ipdb.set_trace()

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']                   # [b, n, h, w]

        # normalize prediction points
        camera_poses_normalized = camera_poses.clone()
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = local_points[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            local_points[valid_batch] = local_points[valid_batch] / norm_factor[..., None, None, None, None]

            if 'global_points' in pred and pred['global_points'] is not None:
                pred['global_points'][valid_batch] /= norm_factor[..., None, None, None, None]

            camera_poses_normalized[valid_batch, ..., :3, 3] /= norm_factor.view(B_, 1, 1)

        pred['local_points'] = local_points          # [b, n, h, w, 3]
        pred['camera_poses'] = camera_poses_normalized # [b, n, 4, 4]

        return pred


    def forward(self, pred, gt_raw):
        # ipdb.set_trace()
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1
        # final_loss += camera_loss
        details.update(camera_loss_details)

        # Ray Loss
        if "cam_rays" in pred:
            B, N, _, H, W = gt['imgs'].shape
            valid_masks = gt['valid_masks'].reshape(B*N, H*W)
            ray_loss, ray_loss_details = compute_ray_loss(pred, valid_masks)
            final_loss += ray_loss
            details.update(ray_loss_details)

            distance_pred = pred['local_points'].view(B*N, H, W, -1).norm(dim=-1)             # [B*N, H, W]
            distance_gt = gt['local_points'].reshape(B*N, H, W, -1).norm(dim=-1)       # [B*N, H, W]
            uncertain_loss, uncertain_loss_details = compute_uncertain_loss(uncertainty=pred['uncertain'].view(B*N, 1, H, W),
                                                                            target_pred=distance_pred.detach(),
                                                                            target_gt=distance_gt.detach(),
                                                                            mask=gt['valid_masks'].view(B*N, H, W))
            final_loss += 0.1 * uncertain_loss
            details.update(uncertain_loss_details)

        # Distance Loss (Radial Loss)
        radial_loss, radial_loss_details = compute_radial_loss(pred, gt, scale)
        final_loss += radial_loss
        details.update(radial_loss_details)

        return final_loss, details


def compute_ray_loss(pred_dict, valid_masks, polar_weight=3.0, polar_asym=0.7, **kwargs):

    details = dict()

    input = pred_dict['cam_rays_pred']               # [B*S, H*W, 3]
    target = pred_dict['cam_rays_gt']           # [B*S, H*W, 3]

    input = input / torch.norm(input, dim=-1, keepdim=True).clamp(min=1e-5)
    target = target / torch.norm(target, dim=-1, keepdim=True).clamp(min=1e-5)

    x_target, y_target, z_target = target.unbind(dim=-1)
    z_clipped = z_target.clip(min=-0.99999, max=0.99999)
    x_clipped = x_target.abs().clip(min=1e-5) * (2 * (x_target > 0).float() - 1)
    polar_target = torch.arccos(z_clipped)
    azimuth_target = torch.atan2(y_target, x_clipped)           # [B*S, H*W]

    x_input, y_input, z_input = input.unbind(dim=-1)
    z_clipped = z_input.clip(min=-0.99999, max=0.99999)
    x_clipped = x_input.abs().clip(min=1e-5) * (2 * (x_input > 0).float() - 1)
    polar_input = torch.arccos(z_clipped)
    azimuth_input = torch.atan2(y_input, x_clipped)

    polar_error = l1(polar_input - polar_target)
    azimuth_error = l1(azimuth_input - azimuth_target)

    quantile_weight = torch.ones_like(polar_input)
    quantile_weight[(polar_target > polar_input) & (polar_target > torch.pi / 2)] = (2 * polar_asym)
    quantile_weight[(polar_target <= polar_input) & (polar_target > torch.pi / 2)] = 2 * (1 - polar_asym)

    mean_polar_error = masked_mean(polar_error * quantile_weight, valid_masks, dim=[1]).squeeze(1)
    mean_azimuth_error = masked_mean(azimuth_error * quantile_weight, valid_masks, dim=[1]).squeeze(1)
    mean_error = (polar_weight * mean_polar_error + mean_azimuth_error) / (1 + polar_weight)

    mean_error = torch.sqrt(mean_error + 1e-4).mean(dim=0)
    details['ray_loss'] = mean_error

    # Return loss dictionary
    return mean_error, details

def compute_radial_loss(pred, gt, scale):

    details = dict()

    B, N, H, W, _ = gt['local_points'].shape
    aligned_radial_pred = pred['radius'].reshape(B, N, H, W) * scale.view(B, 1, 1, 1)
    radial_gt = gt['local_points'].norm(dim=-1)                 # [B, N, H, W]  

    mask = gt['valid_masks']                     # [B, N, H, W] 

    error = l1(aligned_radial_pred - radial_gt)  # [B, N, H, W]

    error = error.reshape(B*N, H, W)             # [BN, H, W]
    mask = mask.reshape(B*N, H, W)               # [BN, H, W]
    error_image = masked_mean(data=error, mask=mask, dim=[-2, -1]).squeeze(1, 2)

    error_image = error_image.mean(dim=0)
    details['radial_loss'] = error_image

    return error_image, details


def compute_uncertain_loss(uncertainty, target_pred, target_gt, mask, rescale=True, margin=0.5):

    details = {}

    BN = target_gt.shape[0]
    mask = mask.bool()                                                  # [BN, H, W]

    target_gt = target_gt.float().reshape(BN, -1)                       # [BN, HW]
    target_pred = target_pred.float().reshape(BN, -1)                   # [BN, HW]
    uncertainty = uncertainty.float().reshape(BN, -1)                   # [BN, HW]
    mask = mask.reshape(BN, -1)                                         # [BN, HW]

    masked_gt = target_gt.masked_fill(~mask, float('nan'))
    med_gt = torch.nanmedian(masked_gt, dim=1, keepdim=True).values
    masked_pred = target_pred.masked_fill(~mask, float('nan'))
    med_pred = torch.nanmedian(masked_pred, dim=1, keepdim=True).values

    if rescale:
        target_pred = target_pred / med_pred.clamp(min=1e-12) * med_gt

    error = torch.abs((log(target_pred) - log(target_gt)).abs() - uncertainty)    # [BN, HW]
    valid_losses = masked_mean(error, dim=[-1], mask=mask).squeeze(dim=-1)        # [BN]

    unvalid_prob = F.softplus(margin - uncertainty)
    unvalid_losses = masked_mean(unvalid_prob, dim=[-1], mask=~mask).squeeze(dim=-1)       # [BN]

    losses = valid_losses + unvalid_losses
    losses = losses.mean(dim=0)

    details["uncertain_loss"] = losses

    return losses, details


def l1(input_tensor: torch.Tensor, gamma: float = 1.0, *args, **kwargs) -> torch.Tensor:
    return torch.abs(input_tensor)

def log(t, eps: float = 1e-5):
    return torch.log(t.clamp(min=eps))

def masked_mean(data: torch.Tensor, mask: torch.Tensor | None, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(
        torch.nan_to_num(data, nan=0.0) * mask, dim=dim, keepdim=True
    ) / mask_sum.clamp(min=1.0)
    return mask_mean
