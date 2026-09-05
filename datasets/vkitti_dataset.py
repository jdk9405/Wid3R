import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
import h5py
from utils.basic import seed_anything
from PIL import Image
from tqdm import tqdm
from datasets.base.transforms import *
import cv2
from scipy.spatial.transform import Rotation as R

import gzip
import json
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import ipdb

from cam_utils.camera import Pinhole
import random
from scipy.io import loadmat
from copy import deepcopy


class vKITTI(BaseDataset):
    min_depth = 0.01
    max_depth = 100.0

    def __init__(self,
                 data_root=None,
                 verbose=False,
                 mode="train",
                 min_num_frames=24,
                 allow_duplicate_img=False,
                 augs_dk=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.verbose = verbose
        self.dataset_label = "vKITTI"
        self.data_root = data_root

        self.invalid_sequence = []
        self.allow_duplicate_img = allow_duplicate_img

        if mode == "train":
            mode_name_list = ["train"]
            self.training = True
        elif mode == "test":
            mode_name_list = ["test"]
            self.training = False
        else:
            raise ValueError(f"Invalid mode: {mode}")


        total_frame_num = 0
        for mode_name in mode_name_list:
            annotation_file = os.path.join(os.path.normpath(data_root) + "_anycam", f"{mode_name}_dk_v2.jgz")

            try:
                with gzip.open(annotation_file, "r") as fin:
                    annotation = json.load(fin)
            except FileNotFoundError:
                raise ValueError(f"Invalid annotation file: {annotation_file}")

            self.data_store = {}
            self.num_imgs = {}
            for seq_name, seq_data in annotation.items():
                if len(seq_data) < min_num_frames:
                    continue
                if seq_name in self.invalid_sequence:
                    continue
                total_frame_num += len(seq_data)
                self.data_store[seq_name] = seq_data
                self.num_imgs[seq_name] = len(seq_data)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        print(f"{status}: vKITTI Data Size: {self.sequence_list_len}")
        print(f"{status}: vKITTI Data dataset length: {len(self)}")

        if self.training:
            for k, v in augs_dk.items():
                setattr(self, k, v)


    def __len__(self):
        return self.sequence_list_len



    def preprocess_inputs(self, image, depth, mask, extrinsic, camera_name, camera_params, target_size=(224, 224), mode="crop", image_path=None, is_flip=False):
        # extrinsic: world saw cam

        # Resize images and depth maps
        height, width = image.shape[:2]
        height_d, width_d = depth.shape[:2]
        height_m, width_m = mask.shape[:2]
        assert height == height_d == height_m and width == width_d == width_m, f"Image, depth, and mask dimensions do not match for {image_path}"

        if mode == "pad":
            raise NotImplementedError("Pad mode is not implemented")
        else:
            new_width = target_size[0]
            new_height = target_size[1]

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_LINEAR).astype(bool)

        # camera resize
        factor_x = new_width / width
        factor_y = new_height / height
        camera = eval(camera_name)(params=torch.from_numpy(camera_params).float())
        camera.resize_v2(factor_x, factor_y)


        # augmentations
        image = torch.from_numpy(image).contiguous().permute(2, 0, 1)
        if self.training:
            depth = torch.from_numpy(depth)[None]
            mask = torch.from_numpy(mask)[None]

            if is_flip:
                TF_flip = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                image = TF.hflip(image)
                depth = TF.hflip(depth)
                mask = TF.hflip(mask)
                camera.flip(H=new_height, W=new_width)
                extrinsic = TF_flip @ extrinsic @ TF_flip
            extrinsic = torch.from_numpy(extrinsic).to(torch.float32)


            for op_name in self.current_ops:
                op_meta = self.augmentations_dict[op_name]
                if op_meta["geometrical"]:
                    image = op_meta["function"](image, interpolation=TF.InterpolationMode.BILINEAR, **op_meta["kwargs"])
                    depth = op_meta["function"](depth.to(torch.float32), interpolation=TF.InterpolationMode.NEAREST, **op_meta["kwargs"])
                    mask = op_meta["function"](mask, interpolation=TF.InterpolationMode.NEAREST, **op_meta["kwargs"])

                    camera.params[:, 0] = camera.params[:, 0] * op_meta["kwargs"]["scale"]
                    camera.params[:, 1] = camera.params[:, 1] * op_meta["kwargs"]["scale"]
                    camera.params[:, 2] = camera.params[:, 2] + op_meta["kwargs"]["translate"][0]
                    camera.params[:, 3] = camera.params[:, 3] + op_meta["kwargs"]["translate"][1]

                    camera.K[:, 0, 0] = camera.K[:, 0, 0] * op_meta["kwargs"]["scale"]
                    camera.K[:, 1, 1] = camera.K[:, 1, 1] * op_meta["kwargs"]["scale"]
                    camera.K[:, 0, 2] = camera.K[:, 0, 2] + op_meta["kwargs"]["translate"][0]
                    camera.K[:, 1, 2] = camera.K[:, 1, 2] + op_meta["kwargs"]["translate"][1]
                else:
                    image = op_meta["function"](image, **op_meta["kwargs"])


            image = image / 255.
            depth = depth.squeeze(0).numpy()
            mask = mask.squeeze(0).numpy().astype(bool)

        else:
            image = image / 255.
            extrinsic = torch.from_numpy(extrinsic).to(torch.float32)

        return image, depth, mask, camera, extrinsic


    def do_augmentation(self, target_size):

        new_width = target_size[0]
        new_height = target_size[1]

        augmentations_dict, augmentations_weights = self._augmentation_space(new_height, new_width)
        augmentations_probs = np.array(list(augmentations_weights.values()))
        self.augmentations_dict = augmentations_dict

        num_augmentations = np.random.choice(list(range(1, 5)), size=1, p=[0.3, 0.4, 0.2, 0.1])
        self.current_ops = np.random.choice(
                list(augmentations_weights.keys()),
                size=num_augmentations,
                replace=False,
                p=augmentations_probs / np.sum(augmentations_probs),
                )


    def _get_views(self, index, resolution, rng, is_flip=False):

        seq_name = self.sequence_list[index]

        metadata = self.data_store[seq_name]
        num_imgs = self.num_imgs[seq_name]

        ids = [rng.integers(0, num_imgs)]


        # sample
        prob_path = os.path.join(self.data_root, seq_name, "view_prob.npy")


        prob = np.load(prob_path)
        eps = 1e-12
        # TODO : magic number
        gamma = 10                   # higher -> more compact
        P_sharp = (prob + eps) ** gamma
        P_sharp /= P_sharp.sum(axis=1, keepdims=True) + eps
        targets = self.sample_targets_from_prob(P_sharp, idx=ids[0], k=self.frame_num-1)
        ids = np.concatenate([ids, targets])

        annos = [metadata[i] for i in ids]



        self.this_views_info = dict(
                scene=seq_name,
                idxs=ids
                )

        # augmentations
        if self.training:
            self.do_augmentation(resolution)

        views = []
        for anno in annos:
            filepath = anno["filepath"]

            image_path = osp.join(self.data_root, filepath)
            image = np.array(Image.open(image_path))
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


            # load depth
            depth_path = image_path.replace("/rgb/", "/depth/").replace("/rgb_", "/depth_").replace(".jpg", ".png")
            
            depth_map = cv2.imread(depth_path, -1) / 100.
            depth_map[~np.isfinite(depth_map)] = 0.
            depth_map = depth_map.astype(np.float32)

            mask = (depth_map > self.min_depth) & (depth_map < self.max_depth)
            depth_map[~mask] = 0.
            depth_map = depth_map.astype(np.float32)

            original_size = np.array(image.shape[:2])

            cam_pose_path = os.path.join(os.path.normpath(self.data_root) + "_anycam", filepath.replace("/rgb/", "/pose/").replace(".jpg", "_pose.txt"))
            camera_pose = np.loadtxt(cam_pose_path).astype(np.float32)
            camera_params_path = os.path.join(os.path.normpath(self.data_root) + "_anycam", filepath.replace("/rgb/", "/pose/").replace(".jpg", "_cam.txt"))
            cam_params = np.loadtxt(camera_params_path).astype(np.float32)
            cam_name = anno["cam_name"]

            rgb_image, depth_map, mask, camera, camera_pose = self.preprocess_inputs(
                    image,
                    depth_map,
                    mask, 
                    camera_pose,
                    cam_name,
                    cam_params,
                    target_size=resolution,
                    image_path=image_path,
                    is_flip=is_flip,
                    )

            views.append(dict(
                img=rgb_image,
                depthmap=depth_map,
                camera_pose=camera_pose,            # world saw cam
                camera=camera,
                dataset=self.dataset_label,
                filepath=filepath,
                valid_mask=mask,
                ))

        return views
        

    def sample_targets_from_prob(self, P, idx, k=8, exclude_self=True, replace=False, rng=None):
        """
        P: (N, N) probability map (each row is assumed to sum to 1)
        idx: query index (row)
        k: number of samples
        exclude_self: whether to exclude the query itself
        replace: whether to allow duplicates (automatically set to True if there are too few candidates)
        """
        probs = np.asarray(P[idx], dtype=np.float64).copy()
        N = probs.size

        # Exclude the query itself and renormalize
        if exclude_self and 0 <= idx < N:
            probs[idx] = 0.0

        s = probs.sum()
        if s <= 0:  # Fall back to a uniform distribution if all probabilities are zero
            probs[:] = 1.0
            if exclude_self and 0 <= idx < N:
                probs[idx] = 0.0
            s = probs.sum()
        probs /= s

        candidates = np.arange(N)
        valid = probs > 0
        if not replace and valid.sum() < k:
            replace = True  # Allow duplicates if there are too few valid candidates

        rng = np.random.default_rng() if rng is None else rng
        return rng.choice(candidates, size=k, replace=replace, p=probs)
        

    
if __name__ == "__main__":


    from hydra import initialize, compose
    import open3d as o3d

    from cam_utils.camera import CameraSampler
    from cam_utils.camera_augmenter import augment_camera, optional_augment_camera


    def create_camera_frame(R, t, size=1.0):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        frame.transform(T)
        return frame

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default")

    augs_dk = cfg.train_dataset.augs_dk
    frame_num = cfg.train_dataset.EDEN.frame_num

    dataset = vKITTI(
        data_root="/fs/gamma-projects/ANYCAM2/data/vkitti/",
        mode="train",
        # mode="test",
        min_num_frames=24,
        resolution=[518, 336],
        augs_dk=augs_dk,
        frame_num=frame_num
    )


    while True:

        for seq_i in range(len(dataset)):
            is_flip = random.random() < 0.5
            aa = dataset._get_views(seq_i, [224, 224], dataset._rng, is_flip)

            num_views = len(aa)
            height, width = aa[0]['depthmap'].shape[-2:]

            # cameras = torch.stack([aa[i]["camera"] for i in range(num_views)]) 
            cameras = torch.cat([aa[i]["camera"] for i in range(num_views)])  
            depths = torch.stack([torch.from_numpy(aa[i]["depthmap"][None]) for i in range(num_views)])

            c_points = cameras.reconstruct(depths).reshape(num_views, 3, -1)
            pose = torch.stack([aa[i]["camera_pose"] for i in range(num_views)])   # world saw cam

            w_points = pose @ torch.cat([c_points, torch.ones((num_views, 1, c_points.shape[-1]))], dim=1)  # (4, 3, N)
            w_points = w_points[:, :3, :]
            world_pts = w_points.permute(0, 2, 1)
            color_pts = np.stack([aa[i]["img"].permute(1, 2, 0) for i in range(len(aa))])


            frames = []
            for ii in range(len(aa)):
                frames.append(create_camera_frame(R=aa[ii]["camera_pose"][:3, :3].numpy(), t=aa[ii]["camera_pose"][:3, 3].numpy()))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_pts.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(color_pts.reshape(-1, 3))
            o3d.visualization.draw_geometries([pcd] + frames)
            ipdb.set_trace()


            # debug augment camera
            inputs = {}
            imgs = torch.stack([aa[i]["img"] for i in range(len(aa))])
            camera_sampler = CameraSampler(file_path="cam_utils/camera_sampler.json")

            valid_masks = torch.from_numpy(np.stack([aa[i]['valid_mask'] for i in range(len(aa))]))[:, None]  # [b, 1, h, w]
            dataset_names = ['vKITTI'] * len(aa)
            images, depths, cameras, masks = optional_augment_camera(imgs, depths, cameras, valid_masks, dataset_names, camera_sampler)

            mask_d = depths > 0.

            images = images.masked_fill(~masks.repeat(1, 3, 1, 1), 0.0).clip(0.0, 1.0)
            depths = depths.masked_fill(~masks, 0.0)

            c_points = cameras.reconstruct(depths).reshape(num_views, 3, -1)
            pose = torch.stack([aa[i]["camera_pose"] for i in range(num_views)])   # world saw cam

            w_points = pose @ torch.cat([c_points, torch.ones((num_views, 1, c_points.shape[-1]))], dim=1)  # (4, 3, N)
            w_points = w_points[:, :3, :]
            world_pts = w_points.permute(0, 2, 1)
            color_pts = images.reshape(num_views, 3, -1).permute(0, 2, 1)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_pts.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(color_pts.reshape(-1, 3))
            ipdb.set_trace()



            
            print(len(dataset))
