import os
import os.path as osp
import glob
import numpy as np
import torch
import json
from math import cos, sin
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from cam_utils.camera import Spherical
from torch.utils.data import Dataset
import cv2
from PIL import Image

import ipdb

class Stanford2D3DEval(Dataset):
    DEPTH_SCALE = 512.
    MIN_DEPTH = 0.01
    MAX_DEPTH = 20.0

    def __init__(self, img_dir, root_dir=None):
        self.img_dir = img_dir
        self.root_dir = root_dir
        self.sequence_list = ['area_5a', 'area_5b']

    def set_sequence_name(self, sequence_name):
        self.sequence_name = sequence_name
        self.seq_img_dir = self.img_dir.format(sequence_name)


    def load_pose(self, pose_file):
        # opencv opengl converter
        gl2cv = torch.tensor([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]]).float()

        T = torch.eye(4)
        with open(pose_file, 'r') as f:
            temp = json.load(f)
            location = np.array(temp['camera_location'])
            final_camera_rotation = np.array(temp['final_camera_rotation'])
            phi, theta, psi = final_camera_rotation
            tt = torch.from_numpy(location[: ,None]).float()               # [3, 1]
            ax = np.array([[1,0,0],
                        [0, cos(phi), -sin(phi)],
                        [0, sin(phi), cos(phi)]])
            ay = np.array([[cos(theta), 0, sin(theta)],
                        [0,1,0],
                        [-sin(theta), 0, cos(theta)]])
            az = np.array([[cos(psi), -sin(psi), 0],
                        [sin(psi), cos(psi), 0],
                        [0,0,1]])
            R = np.matmul(np.matmul(az, ay), ax)
            R = torch.from_numpy(R).float()                         # [3, 3]
            T[:3, :3] = R
            T[:3, 3:4] = tt                                        # opengl

            T = gl2cv @ T @ gl2cv                                 # opencv, world saw cam
            T = T.inverse().to(torch.float32)                     # cam saw world

        return T


    def load_image_files_and_poses(self, names):

        image_paths = []
        gt_extrs = torch.zeros((len(names), 3, 4))
        for idx, name in enumerate(names):

            image_file = osp.join(self.seq_img_dir, name)
            pose_file = image_file.replace('rgb', 'pose').replace('.png', '.json')
            T = self.load_pose(pose_file)                      # cam saw world

            image_paths.append(image_file)
            gt_extrs[idx] = T[:3, :]

        assert len(image_paths) == len(gt_extrs)
        return image_paths, gt_extrs        # cam saw world



    # for mv-recon evaluation
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

            depth_path = image_path.replace("rgb", "depth")

            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / self.DEPTH_SCALE
            depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            mask = (depth > self.MIN_DEPTH) & (depth < self.MAX_DEPTH) & np.isfinite(depth)
            depth[~mask] = 0.
            depth = torch.from_numpy(depth).float().to(device)
            valid_mask.append(mask)

            # world saw cam [4, 4]
            camera_pose_path = image_path.replace("rgb", "pose").replace(".png", ".json")
            camera_pose = self.load_pose(camera_pose_path)
            camera_pose = camera_pose.inverse().to(device)    # world saw cam
            camera_poses.append(camera_pose.cpu().numpy())

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