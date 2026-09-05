import os
import os.path as osp
import glob
import numpy as np
import torch
import json
from math import cos, sin
import sys
from cam_utils.camera import Spherical
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from torch.utils.data import Dataset
from PIL import Image
import cv2
import ipdb

class MP3DEval(Dataset):
    DEPTH_SCALE = 4000.
    MIN_DEPTH = 0.01
    MAX_DEPTH = 20.0

    def __init__(self, root_dir, img_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.sequence_list = [
            "2t7WUuJeko7",
            "5ZKStnWn8Zo",
            "ARNzJeq3xxb",
            "fzynW3qQPVF",
            "jtcxE69GiFV",
            "pa4otMbVnkk",
            "q9vSo1VnCiC",
            "rqfALeAoiTq",
            "UwV83HsGsw3",
            "wc2JMjhGNzB",
            "WYY7iVyf5p8",
            "YFuZgdQ5vWj",
            "yqstnuAEVhm",
            "YVUC4YcDtcY",
            "gxdoqLR6rwA",
            "gYvKGZ5eRqb",
            "RPmz2sHmrrY",
            "Vt2qJdWjCF2",
        ]

    def set_sequence_name(self, sequence_name):
        self.sequence_name = sequence_name
        self.seq_img_dir = self.img_dir.format(sequence_name)

    def get_seq_framenum(self, sequence_name):
        self.set_sequence_name(sequence_name)
        img_exts = ["*.JPG", "*.jpg", "*.png", "*.jpeg"]
        for img_ext in img_exts:
            img_files = glob.glob(osp.join(self.seq_img_dir, img_ext))
            if len(img_files) > 0:
                return len(img_files)
        raise ValueError(f"No image files found in {self.seq_img_dir}")


    def get_data(self, sequence_name, ids, target_height, target_width, device):

        image_paths = []
        images = []
        pointclouds = []
        valid_mask = []
        camera_poses = []

        cam_params = np.array([1., 1., 1., 1., target_width, target_height, np.pi, np.pi / 2.])
        camera = Spherical(params=torch.from_numpy(cam_params).float())
        camera = camera.to(device)


        for sub_dir in ids:
            image_path = osp.join(self.root_dir.format(sequence_name), sub_dir)
            image_paths.append(image_path)

            image = np.array(Image.open(image_path))
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            # H, W = image.shape[:2]
            images.append(torch.from_numpy(image).float().permute(2, 0, 1) / 255.)     # [3, H, W]

            depth_path = image_path.replace("dk_erp_images", "dk_erp_depth_images")

            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / self.DEPTH_SCALE
            depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            mask = (depth > self.MIN_DEPTH) & (depth < self.MAX_DEPTH) & np.isfinite(depth)
            depth[~mask] = 0.
            depth = torch.from_numpy(depth).float().to(device)
            valid_mask.append(mask)

            # world saw cam [4, 4]
            camera_pose_path = image_path.replace("dk_erp_images", "dk_erp_poses").replace(".png", ".txt")
            camera_pose = np.identity(4)
            with open(camera_pose_path, "r") as f:
                temp = f.readlines()
                camera_pose[0] = np.array(temp[0].strip().split())
                camera_pose[1] = np.array(temp[1].strip().split())
                camera_pose[2] = np.array(temp[2].strip().split())
            camera_poses.append(camera_pose)
            camera_pose = torch.from_numpy(camera_pose).float().to(device)

            rays = camera.get_rays([1, target_height, target_width])   # [1, 3, H, W]
            c_points = (rays * depth)[0].reshape(3, -1)     # [3, HW]
            w_points = camera_pose @ torch.cat([c_points, torch.ones((1, c_points.shape[1])).to(device)], dim=0)
            w_points = w_points[:3, :].T             # [HW, 3]
            w_points = w_points.reshape(target_height, target_width, 3)
            pointclouds.append(w_points.cpu().numpy())      # [H, W, 3]


        images = torch.stack(images, dim=0)    # [N, 3, H, W]
        pointclouds = np.stack(pointclouds, axis=0)    # [N, H, W, 3]
        valid_mask = np.stack(valid_mask, axis=0)   # [N, H, W]
        camera_poses = np.stack(camera_poses, axis=0)   # [N, 4, 4]

        output_dict = {}
        output_dict["image_paths"] = image_paths
        output_dict["images"] = images
        output_dict["pointclouds"] = pointclouds
        output_dict["valid_mask"] = valid_mask
        output_dict["camera_poses"] = camera_poses
        return output_dict

