
import os
import os.path as osp
import numpy as np
import torch
import hydra
import logging
import json
from itertools import combinations
import open3d as o3d

from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from time import time

from wid3r.models.wid3r_training import Wid3R
from evaluation.utils.interfaces import infer_cameras_w2c, infer_cameras_c2w
from evaluation.utils.messages import set_default_arg, write_csv
from evaluation.relpose.metric import se3_to_relative_pose_error, calculate_auc_np
from evaluation.utils.files import list_imgs_a_sequence, list_imgs_cams_a_sequence, get_all_sequences
from evaluation.mv_recon.moge_alignment import align_points_scale_xyz_shift
from evaluation.mv_recon.utils import umeyama

from cam_utils.camera import Spherical, Fisheye624, Pinhole

import ipdb

torch.manual_seed(42)

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets     # see configs/evaluation/relpose-colmap-auc.yaml
    all_data_info: DictConfig = hydra_cfg.data                  # see configs/data  
    pretrained_model_name_or_path: str = hydra_cfg.wid3r.pretrained_model_name_or_path    # see configs/evaluation/relpose-colmap-auc.yaml

    # 0. create model
    model = Wid3R(pos_type="rope100", decoder_size="large", load_vggt=False, freeze_encoder=True, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None, use_camera_gt=False)
    model = model.to(hydra_cfg.device).eval()
    checkpoint = torch.load(pretrained_model_name_or_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    logger = logging.getLogger("relpose-angular")
    logger.info(f"Loaded Wid3R from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 2. ready to read, and look up sampled ids from sequence name
        model.eval()
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")


        seq_list = get_all_sequences(dataset_info)
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] COLMAP AUC evaluation on {dataset_name} dataset ..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
        for seq_idx, seq in enumerate(seq_list):
            # 3.1 list the images in the sequence
            filelist, cameras = list_imgs_cams_a_sequence(dataset_info, seq, width=hydra_cfg.width, height=hydra_cfg.height)
            cameras = cameras.to(hydra_cfg.device)
            gt = dataset.get_data(seq, filelist, target_height=hydra_cfg.height, target_width=hydra_cfg.width, device=hydra_cfg.device)
            gt_extrs = gt["camera_poses"]    # world saw cam


            save_dir = osp.join(output_root, seq) if seq is not None else output_root
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")
            with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                # 6. infer cameras
                start_time = time()
                pred_extrs, pred = infer_cameras_c2w(filelist, cameras, model, hydra_cfg)      # world saw cam
                pred_extrs = pred_extrs.numpy()
                end_time = time()
                logger.info(f"Inference time: {end_time - start_time} seconds")
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            errors = []
            assert len(pred_extrs) == len(gt_extrs), f"pred_extrs: {pred_extrs.shape}, gt_extrs: {gt_extrs.shape}"
            # make all pairs
            pair_ids = list(combinations(range(len(filelist)), 2))

            for pair_id in pair_ids:

                pred_extr1 = pred_extrs[pair_id[0]]
                pred_extr2 = pred_extrs[pair_id[1]]
                sfm_pose = np.linalg.inv(pred_extr2) @ pred_extr1                 # 2 saw 1
                gt_extr1 = gt_extrs[pair_id[0]]
                gt_extr2 = gt_extrs[pair_id[1]]
                gt_pose = np.linalg.inv(gt_extr2) @ gt_extr1                 # 2 saw 1

                e_t, e_R = compute_pose_error(gt_pose, sfm_pose[:3, :3], sfm_pose[:3, 3:4])
                e_pose = max(e_t, e_R)

                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)

            error_dict = {"e_t": tot_e_t, "e_R": tot_e_R, "e_pose": tot_e_pose}

            thresholds = [1, 3, 5, 10, 20]
            auc = pose_auc(error_dict["e_pose"], thresholds)

            os.makedirs(save_dir, exist_ok=True)
            txt = open(osp.join(save_dir, "AUC_{}_{}.txt".format(dataset_name, seq)), "w")
            txt.write("final result\n")
            txt.write("auc_1 : {}\n".format(auc[0]))
            txt.write("auc_3 : {}\n".format(auc[1]))
            txt.write("auc_5 : {}\n".format(auc[2]))
            txt.write("auc_10 : {}\n".format(auc[3]))
            txt.write("auc_20 : {}\n".format(auc[4]))
            txt.write("inference time: {}\n".format(end_time - start_time))
            txt.close()


            # visual results
            """
            visual results
            """
            colors = gt['images'].permute(0, 2, 3, 1).cpu().numpy() # [N, H, W, 3]

            pred_pts = pred['points'].squeeze(0).cpu().numpy()      # (N, H, W, 3)
            gt_pts = gt['pointclouds']                              # (N, H, W, 3)
            valid_mask = gt['valid_mask']                           # (N, H, W)

            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # moge refine
            pred_pts_torch = torch.from_numpy(pred_pts[valid_mask]).to(hydra_cfg.device)
            gt_pts_torch = torch.from_numpy(gt_pts[valid_mask]).to(hydra_cfg.device)        # [N', 3]
            # sample
            N_sample = 5000
            indices = torch.randperm(gt_pts_torch.shape[0])[:N_sample]
            pred_pts_torch = pred_pts_torch[indices]
            gt_pts_torch = gt_pts_torch[indices]
            scale, shift = align_points_scale_xyz_shift(pred_pts_torch, gt_pts_torch, weight=1 / gt_pts_torch.norm(dim=-1))
            pred_pts = pred_pts * scale.cpu().numpy() + shift.cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts[valid_mask])
            pcd.colors = o3d.utility.Vector3dVector(colors[valid_mask])
            o3d.io.write_point_cloud(osp.join(save_dir, f"{seq}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts[valid_mask])
            pcd_gt.colors = o3d.utility.Vector3dVector(colors[valid_mask])
            o3d.io.write_point_cloud(osp.join(save_dir, f"{seq}-gt.ply"), pcd_gt)

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs



if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-colmap-auc")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()

