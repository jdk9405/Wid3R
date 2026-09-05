import hydra
import os
import os.path as osp
import numpy as np
import cv2
import logging
import torch
import ipdb

from tqdm import tqdm
from omegaconf import DictConfig, ListConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wid3r.models.wid3r_training import Wid3R
from evaluation.utils.interfaces import infer_monodepth
from evaluation.utils.files import list_imgs_a_sequence, list_imgs_cams_a_sequence, get_all_sequences
from evaluation.utils.messages import set_default_arg

@hydra.main(version_base="1.2", config_path='../configs', config_name='eval')
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets
    all_data_info: DictConfig = hydra_cfg.data
    pretrained_model_name_or_path: str = hydra_cfg.wid3r.pretrained_model_name_or_path

    # 0. create model
    model = Wid3R(pos_type="rope100", decoder_size="large", load_vggt=False, freeze_encoder=True, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None, use_camera_gt=False)
    model = model.to(hydra_cfg.device).eval()
    checkpoint = torch.load(pretrained_model_name_or_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)

    logger = logging.getLogger("monodepth-infer")
    logger.info(f"Loaded Wid3R from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            # some datasets have only a set of images, only for monodepth
            seq_list = [None]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        # 3. infer for each sequence
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Inferring monodepth on {dataset_name} dataset ..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
        for seq_idx, seq in enumerate(seq_list):
            # 3.1 list the images in the sequence
            filelist, cameras = list_imgs_cams_a_sequence(dataset_info, seq, width=hydra_cfg.width, height=hydra_cfg.height)
            cameras = cameras.to(hydra_cfg.device)
            save_dir = osp.join(output_root, seq) if seq is not None else output_root
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")

            # 3.2 infer for each image
            for i, file in enumerate(tqdm(filelist)):
                # 3.2.1 skip if the file already exists
                npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
                png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
                if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                    continue

                # 3.2.2 infer the depth map
                camera = torch.cat([cameras[i:i+1]])
                distance_map = infer_monodepth(file, camera, model, hydra_cfg)
                depth_map = distance_map

                # 3.2.3 save the depth map to the save_dir as npy
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
                elif not isinstance(depth_map, np.ndarray):
                    raise ValueError(f"Unknown depth map type: {type(depth_map)}")
                np.save(npy_save_path, depth_map)

                # 3.2.4 also save the png
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
                cv2.imwrite(png_save_path, depth_color)
        # for each dataset
        logger.info(f"Monodepth inference for dataset {dataset_name} finished!")

    del model
    torch.cuda.empty_cache()
    logger.info(f"Monodepth inference finished!")

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
