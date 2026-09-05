import os
import os.path as osp
import glob
import numpy as np
import torch
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from evaluation.utils.read_write_model import read_cameras_binary, read_images_binary, read_cameras_text, read_images_text, qvec2rotmat
from torch.utils.data import Dataset
import ipdb
from cam_utils.camera import Fisheye624

class SmerfEval(Dataset):
    def __init__(self, img_dir, colmap_file):
        self.img_dir = img_dir
        self.colmap_file = colmap_file
        self.sequence_list = ['alameda', 'berlin', 'london', 'nyc']

    def set_sequence_name(self, sequence_name):
        self.sequence_name = sequence_name
        self.seq_colmap_file = self.colmap_file.format(sequence_name)
        self.seq_img_dir = self.img_dir.format(sequence_name)

    def get_seq_framenum(self, sequence_name):
        self.set_sequence_name(sequence_name)
        img_exts = ["*.JPG", "*.jpg", "*.png", "*.jpeg"]
        for img_ext in img_exts:
            img_files = glob.glob(osp.join(self.seq_img_dir, img_ext))
            if len(img_files) > 0:
                return len(img_files)
        raise ValueError(f"No image files found in {self.seq_img_dir}")

    def load_image_files_and_poses(self, ids):
        if self.seq_colmap_file.endswith(".txt"):
            model_img = read_images_text(self.seq_colmap_file)
        elif self.seq_colmap_file.endswith(".bin"):
            model_img = read_images_binary(self.seq_colmap_file)
        else:
            raise ValueError(f"Unknown colmap file format: {self.seq_colmap_file}")

        image_paths = []
        gt_extrs = torch.zeros((len(ids), 3, 4))
        t_idx = 0
        for i_idx, id in enumerate(sorted(model_img.keys())):
            if i_idx in ids:
                name = model_img[id].name
                qvec, tvec = model_img[id].qvec, model_img[id].tvec
                pose = np.eye(4)
                rotmat = qvec2rotmat(qvec)
                pose[:3, :3] = rotmat
                pose[:3, 3] = tvec          # cam saw world
                # pose = np.linalg.inv(pose)  # world saw cam
                image_paths.append(osp.join(self.seq_img_dir, name))
                gt_extrs[t_idx] = torch.from_numpy(pose[:3, :])
                t_idx += 1

        assert len(image_paths) == len(gt_extrs)
        return image_paths, gt_extrs        # cam saw world
