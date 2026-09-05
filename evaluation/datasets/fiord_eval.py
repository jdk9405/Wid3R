import os
import os.path as osp
import glob
import numpy as np
import torch
import json
from math import cos, sin
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from evaluation.utils.read_write_model import read_cameras_binary, read_images_binary, read_cameras_text, read_images_text, qvec2rotmat
from torch.utils.data import Dataset
import ipdb

class FIORDEval(Dataset):
    def __init__(self, img_dir, colmap_file):
        self.img_dir = img_dir
        self.colmap_file = colmap_file
        self.sequence_list = ["Kitchen_In", "meetingroom", "parakennus"]

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

        # dict
        temp_dict = {}
        for i_idx, id in enumerate(sorted(model_img.keys())):
            name = model_img[id].name
            temp_dict[name] = model_img[id]

        selected_names = []
        for temp in ids:
            selected_names.append(osp.join(*temp.split("/")[-2:]))

        for i_idx, name in enumerate(sorted(temp_dict.keys())):
            if name in selected_names:
                id = temp_dict[name].id
                name = model_img[id].name
                qvec, tvec = model_img[id].qvec, model_img[id].tvec
                pose = np.eye(4)
                rotmat = qvec2rotmat(qvec)
                pose[:3, :3] = rotmat
                pose[:3, 3] = tvec          # cam saw world

                pose = np.linalg.inv(pose)  # world saw cam
                pose = pose @ np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # rotate 180 degree
                pose = np.linalg.inv(pose)  # cam saw world

                image_paths.append(osp.join(self.seq_img_dir, name))
                gt_extrs[t_idx] = torch.from_numpy(pose[:3, :])
                t_idx += 1

        try:
            assert len(image_paths) == len(gt_extrs)
        except AssertionError:
            ipdb.set_trace()    
        return image_paths, gt_extrs        # cam saw world

