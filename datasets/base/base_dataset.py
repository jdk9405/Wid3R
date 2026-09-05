from datasets.base.easy_dataset import EasyDataset
from wid3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import numpy as np
import os
import PIL
import wid3r.utils.cropping as cropping
import torchvision.transforms as tvf
from omegaconf import OmegaConf
from .transforms import *
import pandas as pd
from .utils import *

from typing import Dict, Tuple
import torch
import torchvision.transforms.functional as TF

import random
import ipdb

class BaseDataset(EasyDataset):
    def __init__(
        self,
        seed=2024,
        resolution=None,            # (width, height) or list of (width, height) or list of int
        aug_crop=False,             # False or int, slightly scale the image a bit larger than the target resolution
        aug_focal=False,            # False or float in [0, 1]
        z_far=0,
        frame_num=2,
        transform=tvf.ToTensor(),
        cache_file=None,
        save_cache=False,
        mode='train',
        cache_name=None,
        max_refetch=3,
        random_sample_thres=0.1,
        shuffle=True,
        use_sparse_depth=False,
    ):
        super().__init__()
        self.frame_num = frame_num

        self.transform = transform

        self.shuffle = shuffle

        self.use_sparse_depth = use_sparse_depth

        self._rng = np.random.default_rng(seed)
        self._set_resolutions(resolution)

        self.aug_crop = aug_crop
        self.aug_focal = aug_focal

        self.z_far = z_far

        self.dataset_label = 'BaseDataset'

        self.save_cache = save_cache
        self.cache_loaded = False
        self.cache_name = cache_name
        if cache_file is not None:
            print(f'[BaseDataset] Loading cache from {cache_file}..')
            res = self.load_cache(cache_file)
            if res:
                self.cache_loaded = True
                print(f'[BaseDataset] Cache is loaded.')

        self.mode = mode
        self.max_refetch = max_refetch

        self.random_sample_thres = random_sample_thres  # default not to do that

    def convert_attributes(self):
        """
        Avoid memory leak caused by python list or python dict
        https://github.com/pytorch/pytorch/issues/13246
        """

        def _is_equivalent(original, converted):
            """
            Check if the converted data structure is equivalent to the original.
            """
            try:
                return original == converted
            except Exception:
                return False

        for attr_name in dir(self):
            if attr_name.startswith("__") or callable(getattr(self, attr_name)):
                continue
            
            attr_value = getattr(self, attr_name)
            
            if isinstance(attr_value, list):
                try:
                    converted_value = np.array(attr_value)
                    
                    if _is_equivalent(attr_value, converted_value.tolist()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)
            
            elif isinstance(attr_value, dict):
                try:
                    converted_value = pd.Series(attr_value)
                    if _is_equivalent(attr_value, converted_value.to_dict()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'
        if OmegaConf.is_config(resolutions):
            resolutions = OmegaConf.to_object(resolutions)

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            self._resolutions.append([width, height])

        self.num_resoluions = len(self._resolutions)

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            if len(idx) == 3:
                idx, ar_idx, frame_num = idx
                self.frame_num = frame_num
            else:
                idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        
        error = None
        for _ in range(10):
            try:
                is_flip = random.random() < 0.5
                views = self._get_views(idx, resolution, self._rng, is_flip)

                if self.shuffle:
                    self._rng.shuffle(views)

                # check data-type
                for v, view in enumerate(views):
                    height, width = view['img'].shape[-2:]
                    view['true_shape'] = np.int32((height, width))


            except Exception as e:
                views = None
                if hasattr(self, 'this_views_info'):
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} ({self.this_views_info}) for error {e}.", flush=True
                    )
                else:
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} for error {e}.", flush=True
                    )
                idx = np.random.randint(0, len(self))
                error = e
            
            if views is not None:
                error = None
                break

        if views is None:
            raise error
        
        return views
    
    def load_cache(self, cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            if isinstance(data, dict):
                for key, value in data.items():
                    setattr(self, key, value)
                return True
            else:
                print("Error: The npy file does not contain a dictionary.")
                return False
        except Exception as e:
            print(f"An error occurred while loading the cache: {e}")
            return False

    def _save_cache(self, keys, desc=None):
        if desc is None:
            save_path = f'data/dataset_cache/{self.dataset_label}_cache.npy'
        else:
            save_path = f'data/dataset_cache/{self.dataset_label}_{desc}_cache.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {}
        for key in keys:
            save_dict[key] = getattr(self, key)
        
        np.save(save_path, save_dict)

        print(f'Saved cache to {save_path}.', flush=True)



    """
    Augmentation
    """
    def _augmentation_space(self, height, width) -> Dict[str, Tuple[torch.Tensor, bool]]:

        return (
            {
                "Identity": {
                    "function": lambda x: x,
                    "kwargs": dict(),
                    "geometrical": False,
                    "weight": 1,
                },
                "Brightness": {
                    "function": TF.adjust_brightness,
                    "kwargs": dict(brightness_factor=10 ** np.random.uniform(-self.random_brightness, self.random_brightness)),
                    "geometrical": False,
                },
                "Contrast": {
                    "function": TF.adjust_contrast,
                    "kwargs": dict(contrast_factor=10 ** np.random.uniform(-self.random_contrast, self.random_contrast)),
                    "geometrical": False,
                },
                "Saturation": {
                    "function": TF.adjust_saturation,
                    "kwargs": dict(saturation_factor=10 ** np.random.uniform(-self.random_saturation, self.random_saturation)),
                    "geometrical": False,
                },
                "Gamma": {
                    "function": TF.adjust_gamma,
                    "kwargs": dict(gamma=np.random.uniform(1.0 - self.random_gamma, 1.0 + self.random_gamma)),
                    "geometrical": False,
                },
                "Hue": {
                    "function": TF.adjust_hue,
                    "kwargs": dict(hue_factor=np.random.uniform(-self.random_hue, self.random_hue)),
                    "geometrical": False,
                },
                "Sharpness": {
                    "function": TF.adjust_sharpness,
                    "kwargs": dict(sharpness_factor=10 ** np.random.uniform(-self.random_sharpness, self.random_sharpness)),
                    "geometrical": False,
                },
                "Posterize": {
                    "function": TF.posterize,
                    "kwargs": dict(bits=8 - np.random.randint(0, self.random_posterize)),
                    "geometrical": False,
                },
                "Solarize": {
                    "function": TF.solarize,
                    "kwargs": dict(threshold=int(255 * (1 - np.random.uniform(0, self.random_solarize)))),
                    "geometrical": False,
                },
                "Equalize": {
                    "function": TF.equalize,
                    "kwargs": dict(),
                    "geometrical": False,
                },
                "Autocontrast": {
                    "function": TF.autocontrast,
                    "kwargs": dict(),
                    "geometrical": False,
                },
                "Translation": {
                    "function": TF.affine,
                    "kwargs": dict(
                        angle=0,
                        scale=1,
                        translate=[
                            int(width * np.random.uniform(-self.random_translation, self.random_translation)),
                            int(height * np.random.uniform(-self.random_translation, self.random_translation)),
                        ],
                        shear=0,),
                    "geometrical": True,
                },
                "Scale": {
                    "function": TF.affine,
                    "kwargs": dict(
                        angle=0,
                        scale=np.random.uniform(1.0 - self.random_scale, 1.0 + self.random_scale),
                        translate=[0, 0],
                        shear=0,
                    ),
                    "geometrical": True,
                },
            },
            {
                "Identity": 1,
                "Brightness": self.brightness_p,
                "Contrast": self.contrast_p,
                "Saturation": self.saturation_p,
                "Gamma": self.gamma_p,
                "Hue": self.hue_p,
                "Sharpness": self.sharpness_p,
                "Posterize": self.posterize_p,
                "Solarize": self.solarize_p,
                "Equalize": self.equalize_p,
                "Autocontrast": self.autocontrast_p,
                "Translation": self.translation_p,
                "Scale": self.scale_p,
            },
        )
