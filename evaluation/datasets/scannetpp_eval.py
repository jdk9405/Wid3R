import os
import os.path as osp
import glob
import numpy as np
import torch
import json
from math import cos, sin
import sys
from cam_utils.camera import Fisheye624
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
# from evaluation.utils.read_write_model import read_cameras_binary, read_images_binary, read_cameras_text, read_images_text, qvec2rotmat
from torch.utils.data import Dataset
from PIL import Image
import cv2
import ipdb

OPENCV_OPENGL = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


class ScanNetppEval(Dataset):
    DEPTH_SCALE = 1000.0
    MIN_DEPTH = 0.01
    MAX_DEPTH = 20.0
    def __init__(self, root_dir, img_dir, transform_file):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.transform_file = transform_file
        self.sequence_list = [
            "7b6477cb95", "c50d2d1d42", "cc5237fd77", "acd95847c5", "fb5a96b1a2", "a24f64f7fb",
            "1ada7a0617", "5eb31827b7", "3e8bba0176", "3f15a9266d", "21d970d8de", "5748ce6f01", 
            "c4c04e6d6c", "7831862f02", "bde1e479ad", "38d58a7a31", "5ee7c22ba0", "f9f95681fd",
            "3864514494", "40aec5fffa", "13c3e046d7", "e398684d27", "a8bf42d646", "45b0dac5e3", 
            "31a2c91c43", "e7af285f7d", "286b55a2bf", "7bc286c1b6", "f3685d06a9", "b0a08200c9", 
            "825d228aec", "a980334473", "f2dc06b1d2", "5942004064", "25f3b7a318", "bcd2436daf", 
            "f3d64c30f8", "0d2ee665be", "3db0a1c8f3", "ac48a9b736", "c5439f4607", "578511c8a9", 
            "d755b3d9d8", "99fa5c25e1", "09c1414f1b", "5f99900f09", "9071e139d9", "6115eddb86",
            "27dd4da69e", "c49a8c6cff"
        ]

    def set_sequence_name(self, sequence_name):
        self.sequence_name = sequence_name
        # self.seq_colmap_file = self.colmap_file.format(sequence_name)
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

        transforms_file = osp.join(self.root_dir, "data", sequence_name, "dslr", "nerfstudio", "transforms.json")
        with open(transforms_file, "r") as f:
            transforms = json.load(f)
        frames = transforms["frames"]
        cam_params = np.array([
            transforms["fl_x"],
            transforms["fl_y"],
            transforms["cx"],
            transforms["cy"],
            transforms["k1"],
            transforms["k2"],
            transforms["k3"],
            transforms["k4"],
            0., 0., 0., 0., 0., 0., 0., 0.,
        ])
        camera = Fisheye624(params=torch.from_numpy(cam_params).float())
        factor_x = target_width / transforms["w"]
        factor_y = target_height / transforms["h"]
        camera.resize_v2(factor_x, factor_y)
        camera = camera.to(device)

        FILE_DICT = {}
        for frame in frames:
            file_path = frame["file_path"]
            FILE_DICT[file_path] = frame

        for sub_dir in ids:
            image_path = osp.join(self.root_dir.format(sequence_name), sub_dir)
            image_paths.append(image_path)

            image = np.array(Image.open(image_path))
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            # H, W = image.shape[:2]
            images.append(torch.from_numpy(image).float().permute(2, 0, 1) / 255.)     # [3, H, W]

            depth_path = image_path.replace("scannetpp/data", "scannetpp/render").replace("resized_images", "render_depth").replace("JPG", "png")
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / self.DEPTH_SCALE
            depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            mask = (depth > self.MIN_DEPTH) & (depth < self.MAX_DEPTH) & np.isfinite(depth)
            depth[~mask] = 0.
            depth = torch.from_numpy(depth).float().to(device)
            valid_mask.append(mask)

            img_name = osp.basename(sub_dir)
            camera_pose = np.array(FILE_DICT[img_name]["transform_matrix"])    # world saw cam [4, 4]
            camera_pose = OPENCV_OPENGL @ camera_pose @ OPENCV_OPENGL
            camera_pose = torch.from_numpy(camera_pose).float().to(device)

            c_points = camera.reconstruct(depth)[0].reshape(3, -1)
            w_points = camera_pose @ torch.cat([c_points, torch.ones((1, c_points.shape[1])).to(device)], dim=0)
            w_points = w_points[:3, :].T             # [N, 3]
            w_points = w_points.reshape(target_height, target_width, 3)
            pointclouds.append(w_points.cpu().numpy())             # [H, W, 3]

        images = torch.stack(images, dim=0)    # [N, 3, H, W]
        pointclouds = np.stack(pointclouds, axis=0)    # [N, H, W, 3]
        valid_mask = np.stack(valid_mask, axis=0)   # [N, H, W]

        output_dict = {}
        output_dict["image_paths"] = image_paths
        output_dict["images"] = images
        output_dict["pointclouds"] = pointclouds
        output_dict["valid_mask"] = valid_mask
        return output_dict




