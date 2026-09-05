from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
import numpy as np
from datasets.base.base_dataset import sample_resolutions
import hydra

from wid3r.models.loss import Wid3RLoss

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__, __LOW_QUALITY_DATASETS__

from cam_utils.camera import CameraSampler
from cam_utils.camera_augmenter import optional_augment_camera
from math import tanh, cos

import ipdb

class Wid3RTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.train_loss)

        # for augmentation
        self.camera_sampler = CameraSampler(file_path="cam_utils/camera_sampler.json")

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            encoder_params = [param for param in model_.encoder.named_parameters()]

            # DK contribution
            anycam_header_params = []
            anycam_header_params.extend([("camera_model_token", model_.camera_model_token)])
            anycam_header_params.extend([(name, param) for name, param in model_.camera_model_token_adapter1.named_parameters()])
            anycam_header_params.extend([(name, param) for name, param in model_.camera_model_token_adapter2.named_parameters()])
            anycam_header_params.extend([(name, param) for name, param in model_.ray_module.named_parameters()])
            anycam_header_params.extend([(name, param) for name, param in model_.distance_module.named_parameters()])
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if not name.startswith("encoder.") and not '.encoder.' in name and not name.startswith("camera_model_token")
                and not name.startswith("camera_model_token_adapter1") and not name.startswith("camera_model_token_adapter2")
                and not name.startswith("ray_module") and not name.startswith("distance_module")
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Number of trainable camera model parameters:', sum(p.numel() for _, p in anycam_header_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                return [
                    {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
                    {"params": decay, "weight_decay": weight_decay, 'lr': lr},
                ]

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
            res.extend(handle_weight_decay(anycam_header_params, cfg_optimizer.weight_decay, cfg_optimizer.anycam_header_lr))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Wid3R Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)
            
    def forward_batch(self, batch, mode='train'):
        imgs = torch.stack([view['img'] for view in batch], dim=1)      # [b(3), v(19), 3, 224, 224]
        B, N, _, H, W = imgs.shape
        poses = torch.stack([view['camera_pose'] for view in batch], dim=1).reshape(B*N, 4, 4)          # [BN, 4, 4]   
        dataset_names = list(np.stack([view['dataset'] for view in batch], axis=1).reshape(-1))

        # augment camera
        with torch.cuda.amp.autocast(enabled=False):
            if 'camera' in batch[0]:
                imgs, depths, cameras, valid_masks = optional_augment_camera(batch, dataset_names, self.camera_sampler)
                imgs = imgs.reshape(B, N, 3, H, W)
                
            else:
                cameras = None

        ### LEGACY CODE for training
        if mode == "train":
            prob = 0.5 * (1.0 + cos(torch.pi * self.global_step / self.cfg.train.lr_scheduler.total_steps))
            where_use_gt_rays = torch.rand(B*N, 1, 1) < prob
            where_use_gt_rays = where_use_gt_rays.to(imgs.device)
        else:
            where_use_gt_rays = None
        pred = self.model(imgs, cameras=cameras, where_use_gt_rays=where_use_gt_rays)

        dataset_names = batch[0]['dataset']
        # assert all name in the list are included in the predefined dataset names
        assert all(name in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__ + __LOW_QUALITY_DATASETS__ for name in dataset_names)

        # scannetpp depthmap is depth map,
        # other depthmap is distance map
        pts3ds = []
        for i in range(B):
            camera = cameras[i*N:(i+1)*N]
            depth = depths[i*N:(i+1)*N]
            pose = poses[i*N:(i+1)*N]

            with torch.autocast(device_type=pose.device.type, enabled=False):
                if dataset_names[i] in ['ScanNetpp', 'vKITTI', 'MatrixCity']:
                    c_points = camera.reconstruct(depth).view(N, 3, -1)
                    c_points[torch.isnan(c_points)] = 0.
                    w_points = pose @ torch.cat([c_points, torch.ones((N, 1, c_points.shape[-1]), device=c_points.device)], dim=1)
                    w_points = w_points[:, :3, :].permute(0, 2, 1).reshape(N, H, W, 3)
                    del c_points
                else:
                    ray = camera.get_rays([N, H, W])
                    c_points = (ray * depth).view(N, 3, -1)
                    w_points = pose @ torch.cat([c_points, torch.ones((N, 1, c_points.shape[-1]), device=c_points.device)], dim=1)
                    w_points = w_points[:, :3, :].permute(0, 2, 1).reshape(N, H, W, 3)
                    del ray, c_points

            pts3ds.append(w_points.to(torch.float32))           # [N, H, W, 3]

        pts3ds = torch.stack(pts3ds, dim=0)    # [B, N, H, W, 3]
        for i in range(N):
            batch[i]['pts3d'] = pts3ds[:, i, :, :, :]

        del depths, poses, pts3ds
        if cameras is not None:
            del cameras

        """
        points:             # [3, 19, 224, 224, 3]
        local_points:       # [3, 19, 224, 224, 3]
        conf:               # None
        camera_poses:       # [3, 19, 4, 4]
        global_points:      # None
        """
        return [pred, batch]
    
    # def calculate_loss(self, output, batch, mode='train'):
    def calculate_loss(self, output, mode='train'):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

        return EasyDict(
            loss=loss,
            **details
        )


def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
