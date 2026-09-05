import os
import torch
import numpy as np
import gradio as gr
import sys
import shutil
import tempfile
from datetime import datetime
import glob
import gc
import time
# import spaces         # only for web demo

from wid3r.utils.geometry import se3_inverse, homogenize_points, depth_edge
# from wid3r.models.wid3r import Wid3R
from wid3r.models.wid3r_training import Wid3R
from wid3r.utils.basic import load_images_as_tensor_dk

import trimesh
import matplotlib
from scipy.spatial.transform import Rotation

from cam_utils.camera import Spherical, Fisheye624, Pinhole
from gradio.themes import colors
import ipdb

os.environ["GRADIO_TEMP_DIR"] = "/fs/nexus-scratch/jdk9405/gradio_temp"

EX_360_IMAGES = sorted(
    glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ex_360", "*"))
)


"""
Gradio utils
"""

def set_mode(selected):
    return (
        selected,
        gr.update(variant="primary" if selected == "Fisheye" else "secondary"),
        gr.update(variant="primary" if selected == "360" else "secondary"),
    )

def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    show_cam=True,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        show_cam (bool): Include camera visualization (default: True)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_poses = predictions["camera_poses"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        # conf_threshold = np.percentile(conf, conf_thres)
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_poses)

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            # integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.)          # fixed camera size

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()            # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()           # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d

def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def copy_glb_to_local_tmp(src_glbfile):
    """Copy GLB to local /tmp so Gradio's share URL can serve it reliably."""
    tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False, dir='/tmp')
    tmp.close()
    shutil.copy2(src_glbfile, tmp.name)
    return tmp.name


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
# @spaces.GPU(duration=120)
def run_model(target_dir, model, cam_name) -> dict:
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    interval = 1
    imgs = load_images_as_tensor_dk(os.path.join(target_dir, "images"), interval=interval, TARGET_W=TARGET_W, TARGET_H=TARGET_H).to(device) # (N, 3, H, W)

    # DK contri
    num_imgs, _, height, width = imgs.shape
    if cam_name == "Pinhole":
        cam_params = np.array([1, 1, width, height])
        cameras = torch.cat([eval("Pinhole")(params=torch.from_numpy(cam_params)) for _ in range(num_imgs)])
        cameras = cameras.to(device)
    elif cam_name == "360":
        cam_params = np.array([1., 1., 1., 1., width, height, np.pi, np.pi / 2.])
        cameras = torch.cat([eval("Spherical")(params=torch.from_numpy(cam_params)) for _ in range(num_imgs)])
        cameras = cameras.to(device)
    elif cam_name == "Fisheye":
        cam_params = np.zeros(16)
        # debug = [610.9410078676575,610.9410078676575,690.2852877470175,715.1148341104505,0.4060356696288849,-0.489948419647729,0.1745652818132035,1.132983686620576,-1.701635218233742,0.6511555293441647,0.0006211469747214578,1.932200015697112e-05,-1.485525650871087e-05,0.0002601225712292815,-0.0006582109778598294,3.761395141407565e-05]
        # cam_params = np.array(debug)
        cameras = torch.cat([eval("Fisheye624")(params=torch.from_numpy(cam_params).float()) for _ in range(num_imgs)])
        cameras = cameras.to(device)
        # factor_x = 518 / 1408
        # factor_y = 336 / 1408
        # cameras.resize_v2(factor_x, factor_y)
    else:
        raise AssertionError(f"Camera name {cam_name} is not supported.")

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(imgs[None], cameras=cameras) # Add batch dimension
    predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)


    # Convert confidence to uncertainty
    max_clip = 1.
    predictions['conf'] = torch.sigmoid(1 - predictions['uncertain'].clip(max=max_clip))
    del predictions['local_points']

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            try:
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
            except:
                predictions[key] = predictions[key].cpu().numpy()
    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_images, interval=-1):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.abspath(f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []

    if input_images is not None:
        if interval is not None and interval > 0:
            input_images = input_images[::int(interval)]

        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)
    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_images, interval=-1):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_images, interval=interval)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def load_ex_360_example():
    return EX_360_IMAGES, None, EX_360_IMAGES, None, "Example loaded. Click 'Reconstruct' to begin 3D processing."


def prepare_reconstruction_target(target_dir, example_images, interval=-1):
    if target_dir and target_dir != "None" and os.path.isdir(target_dir):
        return target_dir
    if example_images:
        target_dir, _ = handle_uploads(example_images, interval=interval)
        return target_dir
    return target_dir


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
# commit below for local demo
# @spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    cam_name,
    conf_thres=3.0,
    frame_filter="All",
    show_cam=True,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model, cam_name)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return copy_glb_to_local_tmp(glbfile), log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, show_cam, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "images",
        "points",
        "conf",
        "camera_poses",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
        )
        glbscene.export(file_obj=glbfile)

    return copy_glb_to_local_tmp(glbfile), "Updating Visualization"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    CKPT_DIR = "pretrained_weights/wid3r.bin"
    TARGET_W, TARGET_H = 518, 336


    print("Initializing and loading Wid3R model...")

    model = Wid3R(pos_type="rope100", decoder_size="large", load_vggt=False, freeze_encoder=True, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None, use_camera_gt=False)
    checkpoint = torch.load(CKPT_DIR, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)

    model.eval()
    model = model.to(device)

    # theme = gr.themes.Ocean()
    theme = gr.themes.Ocean(
        primary_hue=colors.orange,
        secondary_hue=colors.orange,
    )
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        /* --- Google 字体导入 (科技感字体) --- */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&display=swap');

        /* --- 动画关键帧 --- */
        /* 背景动态星云效果 */
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* 标题和状态文字的霓虹灯光效 */
        @keyframes text-glow {
            0%, 100% {
                /* text-shadow: 0 0 10px #0ea5e9, 0 0 20px #0ea5e9, 0 0 30px #4f46e5, 0 0 40px #4f46e5; */
                text-shadow: 0 0 10px #F59E0B, 0 0 20px #F59E0B, 0 0 30px #CC7722, 0 0 40px #CC7722;
            }
            50% {
                /* text-shadow: 0 0 5px #0ea5e9, 0 0 10px #0ea5e9, 0 0 15px #4f46e5, 0 0 20px #4f46e5; */
                text-shadow: 0 0 5px #F59E0B, 0 0 10px #F59E0B, 0 0 15px #CC7722, 0 0 20px #CC7722;
            }
        }

        /* 卡片边框呼吸光晕 */
        /* @keyframes border-glow { */
        /*     0% { border-color: rgba(79, 70, 229, 0.5); box-shadow: 0 0 15px rgba(79, 70, 229, 0.3); } */
        /*     50% { border-color: rgba(14, 165, 233, 0.8); box-shadow: 0 0 25px rgba(14, 165, 233, 0.5); } */
        /*    100% { border-color: rgba(79, 70, 229, 0.5); box-shadow: 0 0 15px rgba(79, 70, 229, 0.3); } */
        /* } */
        @keyframes border-glow {
            0% { border-color: rgba(204, 119, 34, 0.5); box-shadow: 0 0 15px rgba(204, 119, 34, 0.3); }
            50% { border-color: rgba(245, 158, 11, 0.8); box-shadow: 0 0 25px rgba(245, 158, 11, 0.5); }
            100% { border-color: rgba(204, 119, 34, 0.5); box-shadow: 0 0 15px rgba(204, 119, 34, 0.3); }
        }

        /* --- 全局样式：宇宙黑暗主题 --- */
        .gradio-container {
            font-family: 'Rajdhani', sans-serif;
            /* background: linear-gradient(-45deg, #020617, #111827, #082f49, #4f46e5); */
            background: linear-gradient(-45deg, #1F1F1F, #2C2C2C, #3A3A3A, #1F1F1F);
            background-size: 400% 400%;
            animation: gradient-animation 20s ease infinite;
            color: #9ca3af;
        }

        /* --- 全局文字颜色修复 (解决Light Mode问题) --- */
        
        /* 1. 修复全局、标签和输入框内的文字颜色 */
        .gradio-container, .gr-label label, .gr-input, input, textarea, .gr-check-radio label {
            color: #d1d5db !important; /* 设置一个柔和的浅灰色 */
        }

        /* 2. 修复 Examples 表头 (这是您问题的核心) */
        thead th {
            color: white !important;
            background-color: #1f2937 !important; /* 同时给表头一个背景色，视觉效果更好 */
        }

        /* 3. 修复 Examples 表格内容文字 */
        tbody td {
            color: #d1d5db !important;
        }
        
        /* --- 状态信息 & 输出标题样式 (custom-log) ✨ --- */
        .custom-log * {
            font-family: 'Orbitron', sans-serif;
            font-size: 24px !important;
            font-weight: 700 !important;
            text-align: center !important;
            color: transparent !important;
            /* background-image: linear-gradient(120deg, #93c5fd, #6ee7b7, #fde047); */
            background-image: linear-gradient(120deg, #F59E0B, #CC7722, #B87333);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            background-clip: text;
            animation: gradient-animation 8s ease-in-out infinite, text-glow 3s ease-in-out infinite;
            padding: 10px 0;
            /* max-height: 80px; */
        }
        
        /* --- UI 卡片/分组样式 (玻璃拟态) 💎 --- */
        .gr-block.gr-group {
            background-color: rgba(17, 24, 39, 0.6);
            /* background-color: rgba(245, 158, 11, 1); */

            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(55, 65, 81, 0.5);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
            /* 应用边框呼吸光晕动画 */
            animation: border-glow 5s infinite alternate;
        }
        .gr-block.gr-group:hover {
            box-shadow: 0 0 25px rgba(14, 165, 233, 0.4);
            border-color: rgba(14, 165, 233, 0.6);
        }
        
        /* --- 酷炫按钮样式 🚀 --- */
        .gr-button {
            /* background: linear-gradient(to right, #4f46e5, #7c3aed, #0ea5e9) !important; */
            background: linear-gradient(to right, #F59E0B, #CC7722, #B87333) !important;
            background-size: 200% auto !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 10px !important;
            /* box-shadow: 0 4px 15px 0 rgba(79, 70, 229, 0.5) !important; */
            box-shadow: 0 4px 15px 0 rgba(245, 158, 11, 0.5) !important;
            transition: all 0.4s ease-in-out !important;
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .gr-button:hover {
            background-position: right center !important;
            /* box-shadow: 0 4px 20px 0 rgba(14, 165, 233, 0.6) !important; */
            box-shadow: 0 4px 20px 0 rgba(245, 158, 11, 0.6) !important;
            transform: translateY(-3px) scale(1.02);
        }
        .gr-button.primary {
            /* 主按钮增加呼吸光晕动画 */
            animation: border-glow 3s infinite alternate;
        }

        /* --- DK contri --- */
        /* --- cam mode --- */
        .mode-btn .gr-button-primary {
            background: linear-gradient(to right, #F59E0B, #F97316, #FDBA74) !important;
            color: #111827 !important;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.45) !important;
        }

        .mode-btn .gr-button-secondary {
            background: linear-gradient(to right, #4B5563, #374151, #111827) !important;
            color: #E5E7EB !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35) !important;
        }

        """,
    ) as demo:
        # Instead of gr.State, we use a hidden Textbox:
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        num_images = gr.Textbox(label="num_images", visible=False, value="None")
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
        example_images_state = gr.State([])

        gr.HTML(
        """
        <style>
                /* --- 介绍文字区专属样式 --- */
                /* .intro-content { font-size: 17px !important; line-height: 1.7; color: #C0C0C0 !important; } */
                .intro-content { font-size: 17px !important; line-height: 1.7; color: #F59E0B !important; }
                /* 额外为 p 标签添加规则，确保覆盖 */
                /* .intro-content p { color: #C0C0C0 !important; } */
                .intro-content p { color: #F59E0B !important; }
                
                .intro-content h1 {
                    font-family: 'Orbitron', sans-serif; font-size: 2.8em !important; font-weight: 900;
                    /* text-align: center; color: #C0C0C0 !important; animation: text-glow 4s ease-in-out infinite; margin-bottom: 0px; */
                    text-align: center; color: #F59E0B !important; margin-bottom: 0px;
                }
                .intro-content .pi-symbol {
                    display: inline-block; color: transparent;
                    /* background-image: linear-gradient(120deg, #38bdf8, #818cf8, #c084fc); */
                    background-image: linear-gradient(120deg, #F59E0B, #CC7722, #B87333);
                    -webkit-background-clip: text; background-clip: text;
                    /* text-shadow: 0 0 15px rgba(129, 140, 248, 0.5); */
                    text-shadow: 0 0 18px rgba(245,158,11,0.35);
                    animation: text-glow 4s ease-in-out infinite;
                }
                .intro-content .subtitle { text-align: center; font-size: 1.1em; margin-bottom: 2rem; }
                .intro-content a.themed-link {
                    /* color: #C0C0C0 !important; text-decoration: none; font-weight: 700; transition: all 0.3s ease; */
                    color: #F59E0B !important; text-decoration: none; font-weight: 700; transition: all 0.3s ease;
                }
                .intro-content a.themed-link:hover { color: #EAEAEA !important; text-shadow: 0 0 8px rgba(234, 234, 234, 0.7); }
                .intro-content h3 {
                    /* font-family: 'Orbitron', sans-serif; color: #C0C0C0 !important; text-transform: uppercase; */
                    font-family: 'Orbitron', sans-serif; color: #F59E0B !important; text-transform: uppercase;
                    letter-spacing: 2px; border-bottom: 1px solid #374151; padding-bottom: 8px; margin-top: 25px;
                }
                .intro-content ol { list-style: none; padding-left: 0; counter-reset: step-counter; }
                .intro-content ol li {
                    counter-increment: step-counter; margin-bottom: 15px; padding-left: 45px; position: relative;
                    /* color: #C0C0C0 !important; */
                    color: #F59E0B !important;
                }
                /* 自定义酷炫列表数字 */
                .intro-content ol li::before {
                    content: counter(step-counter); position: absolute; left: 0; top: 0;
                    width: 30px; height: 30px; background: linear-gradient(135deg, #1e3a8a, #4f46e5);
                    border-radius: 50%; color: white; font-weight: 700; font-family: 'Orbitron', sans-serif;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 0 10px rgba(79, 70, 229, 0.5);
                }
                /* .intro-content strong { color: #C0C0C0 !important; font-weight: 700; } */
                .intro-content strong { color: #F59E0B !important; font-weight: 700; }
                .intro-content .performance-note {
                    /* background-color: rgba(14, 165, 233, 0.1); border-left: 4px solid #0ea5e9; */
                    background-color: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B;
                    padding: 15px; border-radius: 8px; margin-top: 20px;
                }
                /* 确保提示框内的文字也生效 */
                /* .intro-content .performance-note p { color: #C0C0C0 !important; } */
                .intro-content .performance-note p { color: #F59E0B !important; }

                /* --- DK contri --- */
                /* --- cam mode --- */
                /*
                .mode-btn .gr-button {
                    border: 2px solid #CC7722 !important;
                    background: #F3D4A0 !important;
                    color: #7A4A0A !important;
                    font-weight: 600;
                    border-radius: 8px;
                    transition: 0.2s;
                }
                */

                /* 선택된(active) 버튼 스타일 */
                /* 
                .mode-btn-active .gr-button {
                    background: #CC7722 !important;
                    color: white !important;
                    border-color: #CC7722 !important;
                }
                */

                #ex-360-example {
                    position: relative;
                    cursor: pointer;
                }

                #ex-360-example-label {
                    position: absolute;
                    inset: 0;
                    z-index: 5;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    pointer-events: none;
                }

                #ex-360-example-label .example-label {
                    padding: 8px 14px;
                    border: 1px solid #F59E0B;
                    border-radius: 6px;
                    background: rgba(122, 74, 10, 0.86);
                    color: #FFFFFF;
                    font-family: "Orbitron", sans-serif;
                    font-size: 16px;
                    font-weight: 700;
                    letter-spacing: 0;
                    white-space: nowrap;
                }
        </style>
                
        <div class="intro-content">
            <h1>📸 <span class="pi-symbol">Wid3R</span>: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning</h1>
            <p class="subtitle">
                <a class="themed-link" href="https://github.com/jdk9405/Wid3R">🐙 GitHub Repository</a> |
                <a class="themed-link" href="https://jdk9405.github.io/Wid3R/">🚀 Project Page</a>
            </p>
            
            <p>Transform your image collections into detailed 3D models. The <strong class="pi-symbol">Wid3R</strong> model processes your visual data to generate a rich 3D point cloud and calculate the corresponding camera perspectives.</p>
            
            <h3>How to Use:</h3>
            <ol>
                <li><strong>Provide Your Images:</strong> Upload an image set. You can specify a sampling interval below; by default, every image is used. Your inputs will be displayed in the "Preview" gallery.</li>
                <li><strong>Generate the 3D Model:</strong> Press the "Reconstruct" button to initiate the process.</li>
                <li><strong>Explore and Refine Your Model:</strong> The generated 3D model will appear in the viewer on the right. Interact with it by rotating, panning, and zooming. You can also download the model as a GLB file. For further refinement, use the options below the viewer to adjust point confidence, filter by frame, or toggle camera visibility.</li>
            </ol>
            
            <div class="performance-note">
                <p><strong>Quick Note:</strong> The core processing by <strong class="pi-symbol">Wid3R</strong> is incredibly fast, typically finishing in under several seconds. However, rendering the final 3D point cloud can take longer, depending on the complexity of the scene and the capabilities of the rendering engine.</p>
            </div>
        </div>
        """
    )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Upload Images")
                    input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
                    interval = gr.Number(None, label="Image Interval", info="Sampling interval. By default, all images are used.")
                
                image_gallery = gr.Gallery(
                    label="Image Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=False,
                )
                gr.Markdown("### Example")
                with gr.Column(elem_id="ex-360-example"):
                    ex_360_image = gr.Image(
                        value=EX_360_IMAGES[0],
                        show_label=False,
                        height="180px",
                        interactive=False,
                        container=False,
                    )
                    gr.HTML(
                        '<span class="example-label">360 Example</span>',
                        elem_id="ex-360-example-label",
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 2. View Reconstruction")
                log_output = gr.Markdown("Please upload images and click Reconstruct.", elem_classes=["custom-log"])

                cam_mode = gr.State("360")
                with gr.Row(elem_classes=["mode-btn"]):
                    mode_btn_fisheye = gr.Button("Fisheye", scale=1, variant="secondary")
                    mode_btn_360 = gr.Button("360", scale=1, variant="primary")

                    mode_btn_fisheye.click(fn=lambda: set_mode("Fisheye"), inputs=None, outputs=[cam_mode, mode_btn_fisheye, mode_btn_360])
                    mode_btn_360.click(fn=lambda: set_mode("360"), outputs=[cam_mode, mode_btn_fisheye, mode_btn_360])

                reconstruction_output = gr.Model3D(height=480, zoom_speed=0.5, pan_speed=0.5, label="3D Output")
                
                with gr.Row(elem_classes=["mode-btn"]):
                    submit_btn = gr.Button("Reconstruct", scale=3, variant="primary")
                    clear_btn = gr.ClearButton(
                        scale=1
                    )
                
                with gr.Group():
                    gr.Markdown("### 3. Adjust Visualization")
                    with gr.Row():
                        conf_thres = gr.Slider(minimum=0, maximum=100, value=70, step=0.1, label="Confidence Threshold (%)")
                        show_cam = gr.Checkbox(label="Show Cameras", value=True)
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")

        # Set clear button targets
        clear_btn.add([input_images, reconstruction_output, log_output, target_dir_output, image_gallery, interval, example_images_state])

        ex_360_image.select(
            fn=load_ex_360_example,
            inputs=[],
            outputs=[example_images_state, target_dir_output, image_gallery, reconstruction_output, log_output],
        )

        # -------------------------------------------------------------------------
        # "Reconstruct" button logic:
        #  - Clear fields
        #  - Update log
        #  - gradio_demo(...) with the existing target_dir
        #  - Then set is_example = "False"
        # -------------------------------------------------------------------------
        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log, inputs=[], outputs=[log_output]
        ).then(
            fn=prepare_reconstruction_target,
            inputs=[target_dir_output, example_images_state, interval],
            outputs=[target_dir_output],
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                cam_mode,
                conf_thres,
                frame_filter,
                show_cam,
            ],
            outputs=[reconstruction_output, log_output, frame_filter],
        ).then(
            fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
        )

        # -------------------------------------------------------------------------
        # Real-time Visualization Updates
        # -------------------------------------------------------------------------
        conf_thres.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
        frame_filter.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
            ],
            [reconstruction_output, log_output],
        )
    
        show_cam.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
            ],
            [reconstruction_output, log_output],
        )

        # -------------------------------------------------------------------------
        # Auto-update gallery whenever user uploads or changes their files
        # -------------------------------------------------------------------------
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_images, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[example_images_state],
        )

    demo.queue(max_size=20).launch(show_error=True, share=True, allowed_paths=[os.getcwd(), '/tmp'])
