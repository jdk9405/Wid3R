
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

from cam_utils.camera import Spherical, Fisheye624, Pinhole

import ipdb

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets     # see configs/evaluation/relpose-angular.yaml
    all_data_info: DictConfig = hydra_cfg.data                  # see configs/data  
    pretrained_model_name_or_path: str = hydra_cfg.wid3r.pretrained_model_name_or_path    # see configs/evaluation/relpose-angular.yaml

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
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map = json.load(f)
        sequence_list = list(seq_id_map.keys())

        # 3. prepare for metrics
        rError = []
        tError = []
        metric_dict: dict = {}
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating {dataset_name} with Wid3R...")
        tbar = tqdm(sequence_list, desc=f"[{dataset_name} eval]")
        for seq_name in tbar:
            # 4. decide sampling strategy to choose sample frames, from all frames (seq_num_frames) of a sequence
            ids_list = seq_id_map[seq_name]

            for case_num, ids in enumerate(ids_list, start=1):

                # 5. load data sample (only extrinsics are used)
                dataset.set_sequence_name(seq_name)
                image_paths, gt_extrs = dataset.load_image_files_and_poses(ids)

                # 5-1. load cameras
                cam_name = dataset_info.camera.type
                width, height = hydra_cfg.width, hydra_cfg.height
                if cam_name == "Pinhole":
                    cam_params = np.array([1, 1, width, height])
                elif cam_name == "Spherical":
                    cam_params = np.array([1., 1., 1., 1., width, height, np.pi, np.pi / 2.])
                elif cam_name == "Fisheye624":
                    cam_params = np.zeros(16)
                    # debug = [610.9410078676575,610.9410078676575,690.2852877470175,715.1148341104505,0.4060356696288849,-0.489948419647729,0.1745652818132035,1.132983686620576,-1.701635218233742,0.6511555293441647,0.0006211469747214578,1.932200015697112e-05,-1.485525650871087e-05,0.0002601225712292815,-0.0006582109778598294,3.761395141407565e-05]
                    # cam_params = np.array(debug)
                else:
                    raise AssertionError(f"Camera name {cam_name} is not supported.")
                cameras = torch.cat([eval(cam_name)(params=torch.from_numpy(cam_params).float()) for _ in range(len(image_paths))])

                with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                    # 6. infer cameras
                    pred_extrs, pred_intrs = infer_cameras_w2c(image_paths, cameras, model, hydra_cfg)      # cam saw world

                    # 7. compute metrics
                    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                        pred_se3   = pred_extrs,
                        gt_se3     = gt_extrs,
                        num_frames = len(ids),
                    )     

                # 8. update metric for a sequence for each sample
                tbar.set_postfix_str(f"Sequence {seq_name} Case {case_num} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")
                logger.info(f"Sequence {seq_name} Case {case_num} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")

                rError.extend(rel_rangle_deg.cpu().numpy())
                tError.extend(rel_tangle_deg.cpu().numpy())


        rError = np.array(rError)
        tError = np.array(tError)
        # 9. arrange all intermediate results to metrics
        for threshold in dataset_info.metric_thresholds:
            metric_dict[f"Racc_{threshold}"] = np.mean(rError < threshold).item() * 100
            metric_dict[f"Tacc_{threshold}"] = np.mean(tError < threshold).item() * 100
            Auc, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
            metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100

        logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")

        # 9. save evaluation metrics to csv
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)

    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating model Wid3R on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()




if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    with torch.no_grad():
        main()
