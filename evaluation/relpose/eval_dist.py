
import os
import os.path as osp
import numpy as np
import torch
import hydra
import logging
import json

from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wid3r.models.wid3r_training import Wid3R
from evaluation.utils.interfaces import infer_cameras_w2c, infer_cameras_c2w
from evaluation.utils.messages import set_default_arg, write_csv
from evaluation.relpose.metric import se3_to_relative_pose_error, calculate_auc_np
from evaluation.utils.files import list_imgs_a_sequence, list_imgs_cams_a_sequence, get_all_sequences
from evaluation.relpose.evo_utils import calculate_averages, load_traj, eval_metrics, plot_trajectory, get_tum_poses, save_tum_poses

from cam_utils.camera import Spherical, Fisheye624, Pinhole
from evo.core.trajectory import PosePath3D, PoseTrajectory3D

import ipdb

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-distance.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    pretrained_model_name_or_path: str = hydra_cfg.wid3r.pretrained_model_name_or_path  # see configs/evaluation/relpose-distance.yaml

    # 0. create model
    model = Wid3R(pos_type="rope100", decoder_size="large", load_vggt=False, freeze_encoder=True, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None, use_camera_gt=False)
    model = model.to(hydra_cfg.device).eval()
    checkpoint = torch.load(pretrained_model_name_or_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    logger = logging.getLogger("relpose-distance")
    logger.info(f"Loaded Wid3R from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 2. get the sequence list
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map = json.load(f)
        sequence_list = list(seq_id_map.keys())
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)

        # 3. infer for each sequence
        model.eval()
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering relpose(c2w) on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        results = []
        tbar = tqdm(sequence_list, desc=f"[{dataset_name} eval]")
        for seq_name in tbar:
            # 4.1 list all images of this sequence
            ids_list = seq_id_map[seq_name]

            for case_num, ids in enumerate(ids_list, start=1):
                dataset.set_sequence_name(seq_name)
                filelist, gt_extrs = dataset.load_image_files_and_poses(ids)
                filelist = filelist[:: hydra_cfg.pose_eval_stride]
                gt_extrs = gt_extrs[:: hydra_cfg.pose_eval_stride]      # [N, 3, 4]  # cam saw world


                """
                traj_tum: (N, 7), in 
                timestamps_mat: (N, 1)
                """
                # gt_extrs: (N, 4, 4) with last row [0, 0, 0, 1]
                gt_extrs = np.concatenate([gt_extrs, np.zeros((gt_extrs.shape[0], 1, 4))], axis=1)
                gt_extrs[:, 3, 3] = 1
                gt_poses = np.linalg.inv(gt_extrs)       # world saw cam

                pose_path = PosePath3D(poses_se3=gt_poses)
                timestamps_mat = np.arange(gt_poses.shape[0]).astype(float)

                traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)
                xyz = traj.positions_xyz
                # shift -1 column -> w in back column
                # quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
                # uncomment this line if the quaternion is in scalar-first format
                quat = traj.orientations_quat_wxyz

                traj_tum = np.column_stack((xyz, quat))
                gt_traj = (traj_tum, timestamps_mat)



                # 4.2 real inference
                # pr_poses: c2w poses, (N, 3, 4), in torch
                # pr_intrs: focals + pps, (N, 3, 3), in numpy
                cam_name = dataset_info.camera.type
                width, height = hydra_cfg.width, hydra_cfg.height
                if cam_name == "Pinhole":
                    cam_params = np.array([1, 1, width, height])
                elif cam_name == "Spherical":
                    cam_params = np.array([1., 1., 1., 1., width, height, np.pi, np.pi / 2.])
                elif cam_name == "Fisheye624":
                    cam_params = np.zeros(16)
                else:
                    raise AssertionError(f"Camera name {cam_name} is not supported.")
                cameras = torch.cat([eval(cam_name)(params=torch.from_numpy(cam_params).float()) for _ in range(len(filelist))])

                pr_poses, pr_intrs = infer_cameras_c2w(filelist, cameras, model, hydra_cfg)
                pred_traj = get_tum_poses(pr_poses)

                os.makedirs(osp.join(output_root, seq_name), exist_ok=True)
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, gt_traj,
                    seq = seq_name,
                    filename = osp.join(output_root, seq_name, f"case_{case_num}_eval_metric.txt")
                )
                plot_trajectory(pred_traj, gt_traj, title=seq_name, filename=osp.join(output_root, seq_name, f"case_{case_num}_vis.png"), verbose=hydra_cfg.verbose)
                
                # 4.3 save sequence case metrics to csv
                seq_metrics = {
                    "dataset": dataset_name,
                    "seq": seq_name,
                    "case": case_num,
                    "ATE": ate,
                    "RPE trans": rpe_trans,
                    "RPE rot": rpe_rot,
                }
                write_csv(osp.join(output_root, "seq_metrics.csv"), seq_metrics)
                results.append((seq_name, ate, rpe_trans, rpe_rot))

                # 4.4 update metrics for a sequence to tqdm bar
                tbar.set_postfix_str(f"Seq {seq_name} Case {case_num} ATE: {ate:5.2f} | RPE-trans: {rpe_trans:5.2f} | RPE-rot: {rpe_rot:5.2f}")
        
        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

        dataset_metrics = {
            "ATE": avg_ate,
            "RPE trans": avg_rpe_trans,
            "RPE rot": avg_rpe_rot,
        }
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, dataset_metrics)
        logger.info(f"{dataset_name} - Average pose estimation metrics: {dataset_metrics}")

    del model
    torch.cuda.empty_cache()



if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-distance")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
