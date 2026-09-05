import os
import os.path as osp
import glob
from typing import Optional
from omegaconf import DictConfig, ListConfig
import numpy as np
import torch

from cam_utils.camera import Spherical, Fisheye624, Pinhole


def get_all_sequences(dataset_cfg: DictConfig, sort_by_seq_name: bool = True):
    if isinstance(dataset_cfg.ls_all_seqs, str):
        # if ls_all_seqs is a string, it is the root path of sequences
        seq_list = [d for d in os.listdir(dataset_cfg.ls_all_seqs) if osp.isdir(osp.join(dataset_cfg.ls_all_seqs, d))]
    elif isinstance(dataset_cfg.ls_all_seqs, ListConfig):
        # if ls_all_seqs is a ListConfig, it is the ListConfig of sequence names
        seq_list = dataset_cfg.ls_all_seqs
    else:
        raise ValueError(f"Unknown ls_all_seqs type: {type(dataset_cfg.ls_all_seqs)}, ls_all_seqs is {dataset_cfg.ls_all_seqs}, which should be a string or a ListConfig")
    return sorted(seq_list) if sort_by_seq_name else seq_list

def list_imgs_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    subdir = dataset_cfg.img.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.img.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    return filelist

def list_imgs_cams_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None, width=518, height=336):
    subdir = dataset_cfg.img.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.img.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    # DK contribution
    num_files = len(filelist)
    cam_name = dataset_cfg.camera.type
    if cam_name == "Pinhole":
        cam_params = np.array([1, 1, width, height])
    elif cam_name == "Spherical":
        cam_params = np.array([1., 1., 1., 1., width, height, np.pi, np.pi / 2.])
    elif cam_name == "Fisheye":
        cam_params = np.zeros(16)
    else:
        raise AssertionError(f"Camera name {cam_name} is not supported.")
    cameras = torch.cat([eval(cam_name)(params=torch.from_numpy(cam_params)) for _ in range(num_files)])
    return filelist, cameras



def list_depths_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    subdir = dataset_cfg.depth.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.depth.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    return filelist