from functools import wraps
from time import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from scipy import interpolate

import builtins
import datetime
import os
import math
import torch.distributed as dist
import logging

from collections import defaultdict, deque


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



@torch.jit.script
def max_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).max(dim=-1).values


def last_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[-1]


def first_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[0]


@torch.jit.script
def softmax_stack(
    tensors: list[torch.Tensor], temperature: float = 1.0
) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return F.softmax(torch.stack(tensors, dim=-1) / temperature, dim=-1).sum(dim=-1)


@torch.jit.script
def mean_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).mean(dim=-1)


@torch.jit.script
def sum_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).sum(dim=-1)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def get_params(module, lr, wd):
    skip_list = {}
    skip_keywords = {}
    if hasattr(module, "no_weight_decay"):
        skip_list = module.no_weight_decay()
    if hasattr(module, "no_weight_decay_keywords"):
        skip_keywords = module.no_weight_decay_keywords()
    has_decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (name in skip_list)
            or any((kw in name for kw in skip_keywords))
            or len(param.shape) == 1
            or name.endswith(".gamma")
            or name.endswith(".beta")
            or name.endswith(".bias")
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    group1 = {
        "params": has_decay,
        "weight_decay": wd,
        "lr": lr,
        "weight_decay_init": wd,
        "weight_decay_base": wd,
        "lr_base": lr,
    }
    group2 = {
        "params": no_decay,
        "weight_decay": 0.0,
        "lr": lr,
        "weight_decay_init": 0.0,
        "weight_decay_base": 0.0,
        "weight_decay_final": 0.0,
        "lr_base": lr,
    }
    return [group1, group2], [lr, lr]


def get_num_layer_for_swin(var_name, num_max_layer, layers_per_stage):
    if var_name in ("cls_token", "mask_token", "pos_embed", "absolute_pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("layers"):
        if var_name.split(".")[2] == "blocks":
            stage_id = int(var_name.split(".")[1])
            layer_id = int(var_name.split(".")[3]) + sum(layers_per_stage[:stage_id])
            return layer_id + 1
        elif var_name.split(".")[2] == "downsample":
            stage_id = int(var_name.split(".")[1])
            layer_id = sum(layers_per_stage[: stage_id + 1])
            return layer_id
    else:
        return num_max_layer - 1


def get_params_layerdecayswin(module, lr, wd, ld):
    skip_list = {}
    skip_keywords = {}
    if hasattr(module, "no_weight_decay"):
        skip_list = module.no_weight_decay()
    if hasattr(module, "no_weight_decay_keywords"):
        skip_keywords = module.no_weight_decay_keywords()
    layers_per_stage = module.depths
    num_layers = sum(layers_per_stage) + 1
    lrs = []
    params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            print(f"{name} frozen")
            continue  # frozen weights
        layer_id = get_num_layer_for_swin(name, num_layers, layers_per_stage)
        lr_cur = lr * ld ** (num_layers - layer_id - 1)
        # if (name in skip_list) or any((kw in name for kw in skip_keywords)) or len(param.shape) == 1 or name.endswith(".bias"):
        if (name in skip_list) or any((kw in name for kw in skip_keywords)):
            wd_cur = 0.0
        else:
            wd_cur = wd
        params.append({"params": param, "weight_decay": wd_cur, "lr": lr_cur})
        lrs.append(lr_cur)
    return params, lrs


def log(t, eps: float = 1e-5):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def divisible_by(numer, denom):
    return (numer % denom) == 0


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)


def load_pretrained(state_dict, checkpoint):
    checkpoint_model = checkpoint["model"]
    if any([True if "encoder." in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint_model.items()
            if k.startswith("encoder.")
        }
        print("Detect pre-trained model, remove [encoder.] prefix.")
    else:
        print("Detect non-pre-trained model, pass without doing anything.")
    print(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
    checkpoint = load_checkpoint_swin(state_dict, checkpoint_model)


def load_checkpoint_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = (
                            relative_position_bias_table_pretrained[:, i]
                            .view(src_size, src_size)
                            .float()
                            .numpy()
                        )
                        f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx, dy))
                            .contiguous()
                            .view(-1, 1)
                            .to(relative_position_bias_table_pretrained.device)
                        )

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in checkpoint_model.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [
        k for k in checkpoint_model.keys() if "relative_coords_table" in k
    ]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if "cpb_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace("cpb_mlp", "rpe_mlp")] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    encoder_keys = [k for k in checkpoint_model.keys() if k.startswith("encoder.")]
    for k in encoder_keys:
        checkpoint_model[k.replace("encoder.", "")] = checkpoint_model.pop(k)

    return checkpoint_model


def add_padding_metas(out, image_metas):
    device = out.device
    # left, right, top, bottom
    paddings = [img_meta.get("paddings", [0] * 4) for img_meta in image_metas]
    paddings = torch.stack(paddings).to(device)
    outs = [F.pad(o, padding, value=0.0) for padding, o in zip(paddings, out)]
    return torch.stack(outs)


# left, right, top, bottom
def remove_padding(out, paddings):
    H, W = out.shape[-2:]
    outs = [
        o[..., padding[2] : H - padding[3], padding[0] : W - padding[1]]
        for padding, o in zip(paddings, out)
    ]
    return torch.stack(outs)


def remove_padding_metas(out, image_metas):
    B, C, H, W = out.shape
    device = out.device
    # left, right, top, bottom
    paddings = [
        torch.tensor(img_meta.get("paddings", [0] * 4)) for img_meta in image_metas
    ]
    return remove_padding(out, paddings)


def ssi_helper(tensor1, tensor2):
    stability_mat = 1e-4 * torch.eye(2, device=tensor1.device)
    tensor2_one = torch.stack([tensor2, torch.ones_like(tensor2)], dim=1)
    scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (
        tensor2_one.T @ tensor1.unsqueeze(1)
    )
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return scale, shift


def calculate_mean_values(names, values):
    # Create a defaultdict to store sum and count for each name
    name_values = {name: {} for name in names}

    # Iterate through the lists and accumulate values for each name
    for name, value in zip(names, values):
        name_values[name]["sum"] = name_values[name].get("sum", 0.0) + value
        name_values[name]["count"] = name_values[name].get("count", 0.0) + 1

    # Calculate mean values and create the output dictionary
    output_dict = {
        name: name_values[name]["sum"] / name_values[name]["count"]
        for name in name_values
    }

    return output_dict


def remove_leading_dim(infos):
    if isinstance(infos, dict):
        return {k: remove_leading_dim(v) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos.squeeze(0)
    else:
        return infos


def recursive_index(infos, index):
    if isinstance(infos, dict):
        return {k: recursive_index(v, index) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos[index]
    else:
        return infos


def to_cpu(infos):
    if isinstance(infos, dict):
        return {k: to_cpu(v) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos.detach()
    else:
        return infos


def masked_mean(
    data: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: list[int] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    dim = dim if dim is not None else list(range(data.dim()))
    if mask is None:
        return data.mean(dim=dim, keepdim=keepdim)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean.squeeze(dim) if not keepdim else mask_mean


class ProfileMethod:
    def __init__(self, model, func_name, track_statistics=True, verbose=False):
        self.model = model
        self.func_name = func_name
        self.verbose = verbose
        self.track_statistics = track_statistics
        self.timings = []

    def __enter__(self):
        # Start timing
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.end_time = time()

            elapsed_time = self.end_time - self.start_time

            self.timings.append(elapsed_time)
            if self.track_statistics and len(self.timings) > 25:

                # Compute statistics if tracking
                timings_array = np.array(self.timings)
                mean_time = np.mean(timings_array)
                std_time = np.std(timings_array)
                quantiles = np.percentile(timings_array, [0, 25, 50, 75, 100])
                print(
                    f"{self.model.__class__.__name__}.{self.func_name} took {elapsed_time:.4f} seconds"
                )
                print(f"Mean Time: {mean_time:.4f} seconds")
                print(f"Std Time: {std_time:.4f} seconds")
                print(
                    f"Quantiles: Min={quantiles[0]:.4f}, 25%={quantiles[1]:.4f}, Median={quantiles[2]:.4f}, 75%={quantiles[3]:.4f}, Max={quantiles[4]:.4f}"
                )

            else:
                print(
                    f"{self.model.__class__.__name__}.{self.func_name} took {elapsed_time:.4f} seconds"
                )


def profile_method(track_statistics=True, verbose=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with ProfileMethod(self, func.__name__, track_statistics, verbose):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ProfileFunction:
    def __init__(self, func_name, track_statistics=True, verbose=False):
        self.func_name = func_name
        self.verbose = verbose
        self.track_statistics = track_statistics
        self.timings = []

    def __enter__(self):
        # Start timing
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.end_time = time()

            elapsed_time = self.end_time - self.start_time

            self.timings.append(elapsed_time)
            if self.track_statistics and len(self.timings) > 25:

                # Compute statistics if tracking
                timings_array = np.array(self.timings)
                mean_time = np.mean(timings_array)
                std_time = np.std(timings_array)
                quantiles = np.percentile(timings_array, [0, 25, 50, 75, 100])
                print(f"{self.func_name} took {elapsed_time:.4f} seconds")
                print(f"Mean Time: {mean_time:.4f} seconds")
                print(f"Std Time: {std_time:.4f} seconds")
                print(
                    f"Quantiles: Min={quantiles[0]:.4f}, 25%={quantiles[1]:.4f}, Median={quantiles[2]:.4f}, 75%={quantiles[3]:.4f}, Max={quantiles[4]:.4f}"
                )

            else:
                print(f"{self.func_name} took {elapsed_time:.4f} seconds")


def profile_function(track_statistics=True, verbose=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with ProfileFunction(func.__name__, track_statistics, verbose):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def recursive_apply(inputs, func):
    if isinstance(inputs, list):
        return [recursive_apply(camera, func) for camera in inputs]
    else:
        return func(inputs)


def squeeze_list(nested_list, dim, current_dim=0):
    # If the current dimension is in the list of indices to squeeze
    if isinstance(nested_list, list) and len(nested_list) == 1 and current_dim == dim:
        return squeeze_list(nested_list[0], dim, current_dim + 1)
    elif isinstance(nested_list, list):
        return [squeeze_list(item, dim, current_dim + 1) for item in nested_list]
    else:
        return nested_list


def match_gt(tensor1, tensor2, padding1, padding2, mode: str = "bilinear"):
    """
    Transform each item in tensor1 batch to match tensor2's dimensions and padding.

    Args:
        tensor1 (torch.Tensor): The input tensor to transform, with shape (batch_size, channels, height, width).
        tensor2 (torch.Tensor): The target tensor to match, with shape (batch_size, channels, height, width).
        padding1 (tuple): Padding applied to tensor1 (pad_left, pad_right, pad_top, pad_bottom).
        padding2 (tuple): Desired padding to be applied to match tensor2 (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: The batch of transformed tensors matching tensor2's size and padding.
    """
    # Get batch size
    batch_size = len(tensor1)
    src_dtype = tensor1[0].dtype
    tgt_dtype = tensor2[0].dtype

    # List to store transformed tensors
    transformed_tensors = []

    for i in range(batch_size):
        item1 = tensor1[i]
        item2 = tensor2[i]

        h1, w1 = item1.shape[1], item1.shape[2]
        pad1_l, pad1_r, pad1_t, pad1_b = (
            padding1[i] if padding1 is not None else (0, 0, 0, 0)
        )
        pad2_l, pad2_r, pad2_t, pad2_b = (
            padding2[i] if padding2 is not None else (0, 0, 0, 0)
        )
        item1_unpadded = item1[:, pad1_t : h1 - pad1_b, pad1_l : w1 - pad1_r]

        h2, w2 = (
            item2.shape[1] - pad2_t - pad2_b,
            item2.shape[2] - pad2_l - pad2_r,
        )

        item1_resized = F.interpolate(
            item1_unpadded.unsqueeze(0).to(tgt_dtype), size=(h2, w2), mode=mode
        )
        item1_padded = F.pad(item1_resized, (pad2_l, pad2_r, pad2_t, pad2_b))
        transformed_tensors.append(item1_padded)

    transformed_batch = torch.cat(transformed_tensors)
    return transformed_batch.to(src_dtype)




"""
CroCo misc.
"""
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed_mode(args):
    nodist = args.nodist if hasattr(args,'nodist') else False 
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def get_parameter_groups(model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values 
    assert layer_decay==1.0 or 0.<layer_decay<1.
    if layer_decay<1.:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, 'dec_blocks') else 0
        num_layers = enc_depth+dec_depth
        layer_decay_values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        if layer_decay<1.:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, fname=None, best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None: fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    if best_so_far is not None: to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    args.start_epoch = 0
    best_so_far = None
    if args.resume is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("Resume checkpoint %s" % args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'best_so_far' in checkpoint:
            best_so_far = checkpoint['best_so_far']
            print(" & best_so_far={:g}".format(best_so_far))
        else:
            print("")
        print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
    return best_so_far


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_iter=None):
        i = 0
        if not header:
            header = ''
        start_time = time()
        end = time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ':' + str(len(str(len_iterable))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it,obj in enumerate(iterable):
            data_time.update(time() - end)
            yield obj
            iter_time.update(time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time()
            if max_iter and it >= max_iter:
                break
        total_time = time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len_iterable))

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x



def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"GPU SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
