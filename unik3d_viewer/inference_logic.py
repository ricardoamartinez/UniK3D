import os
import numpy as np
import cv2
import torch
import time
import queue
import glob
import re
import traceback
import mss
import math
from collections import deque
import threading

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    print("ERROR: pytorch_wavelets not installed; install via 'pip install pytorch-wavelets'")
    DWTForward = None
    DWTInverse = None

from unik3d.models import UniK3D # Assuming unik3d is installed
from .file_io import save_glb, load_glb
from .config import DEFAULT_SETTINGS

# --- Inference Thread Helper Functions (copied from live_slam_viewer.py) ---

def _initialize_input_source(input_mode, input_filepath, data_queue, playback_state_ref):
    """Initializes the input source (camera, file, sequence)."""
    cap = None
    is_video = False
    is_image = False
    is_glb_sequence = False
    glb_files = []
    frame_source_name = "Live Camera"
    video_total_frames = 0
    video_fps = 30
    image_frame = None
    error_message = None

    if input_mode == "Live":
        print("Initializing camera...")
        data_queue.put(("status", "Initializing camera..."))
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            error_message = "Could not open camera."
        else:
            is_video = True
            frame_source_name = "Live Camera"
            video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            print("Camera initialized.")
            data_queue.put(("status", "Camera initialized."))
    elif input_mode == "File" and input_filepath and os.path.exists(input_filepath):
        frame_source_name = os.path.basename(input_filepath)
        if os.path.isdir(input_filepath):
            print(f"Scanning directory for GLB files: {input_filepath}")
            data_queue.put(("status", f"Scanning directory: {frame_source_name}..."))
            glb_files = sorted(glob.glob(os.path.join(input_filepath, "*.glb")),
                               key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)) if re.search(r'(\d+)', os.path.basename(x)) else -1)
            if not glb_files:
                error_message = f"No .glb files found in: {frame_source_name}"
            else:
                is_glb_sequence = True
                input_mode = "GLB Sequence" # Update mode explicitly
                video_total_frames = len(glb_files)
                video_fps = 30 # Assume 30 FPS
                playback_state_ref["total_frames"] = video_total_frames
                playback_state_ref["current_frame"] = 0
                print(f"GLB sequence loaded successfully ({video_total_frames} frames).")
                data_queue.put(("status", f"Loaded GLB sequence: {frame_source_name}"))
        else:
            print(f"Initializing video capture for file: {input_filepath}")
            data_queue.put(("status", f"Opening file: {frame_source_name}..."))
            cap = cv2.VideoCapture(input_filepath)
            if cap.isOpened():
                is_video = True
                video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
                playback_state_ref["total_frames"] = video_total_frames
                playback_state_ref["current_frame"] = 0
                print(f"Video file opened successfully ({video_total_frames} frames @ {video_fps:.2f} FPS).")
                data_queue.put(("status", f"Opened video: {frame_source_name}"))
            else:
                print("Failed to open as video, trying as image...")
                try:
                    image_frame = cv2.imread(input_filepath)
                    if image_frame is not None:
                        is_image = True
                        print("Image file loaded successfully.")
                        data_queue.put(("status", f"Loaded image: {frame_source_name}"))
                    else:
                        error_message = f"Cannot open file: {frame_source_name}"
                except Exception as e_img:
                    error_message = f"Error reading file: {frame_source_name} ({e_img})"
    elif input_mode == "Screen":
        print("Initializing screen capture...")
        data_queue.put(("status", "Initializing screen capture..."))
        is_video = True
        frame_source_name = "Screen Share"
        video_fps = 30
        playback_state_ref["total_frames"] = 0
        playback_state_ref["current_frame"] = 0
    else:
        error_message = "Invalid input source specified."

    if error_message:
        print(f"Error: {error_message}")
        data_queue.put(("error", error_message))
        return None, None, False, False, False, [], "Error", 0, 30, None, error_message

    return cap, image_frame, is_video, is_image, is_glb_sequence, glb_files, frame_source_name, video_total_frames, video_fps, input_mode, None


def _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index, playback_state_ref, last_frame_read_time_ref, video_fps, is_video, is_image, is_glb_sequence, image_frame, frame_count, data_queue, frame_source_name):
    """Reads the next frame/GLB based on playback state. Returns timing delta."""
    frame = None
    points_xyz_np = None
    colors_np = None
    ret = False
    frame_read_delta_t = 0.0
    current_time = time.time()
    last_frame_read_time = last_frame_read_time_ref[0]
    new_sequence_frame_index = sequence_frame_index
    end_of_stream = False
    read_successful = False

    is_playing = playback_state_ref.get("is_playing", True)
    playback_speed = playback_state_ref.get("speed", 1.0)
    loop_video = playback_state_ref.get("loop", True)
    restart_video = playback_state_ref.get("restart", False)

    if restart_video:
        if (is_video and cap) or is_glb_sequence:
            if cap: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            new_sequence_frame_index = 0
            playback_state_ref["current_frame"] = 0
            print("DEBUG: Input restarted.")
        playback_state_ref["restart"] = False

    read_next_frame = is_playing or (is_image and frame_count == 0)

    if not read_next_frame and (is_video or is_glb_sequence):
        time.sleep(0.05)
        return None, None, None, False, new_sequence_frame_index, frame_read_delta_t, False

    target_delta = (1.0 / video_fps) / playback_speed if video_fps > 0 and playback_speed > 0 else 0.1
    time_since_last_read = current_time - last_frame_read_time

    if time_since_last_read >= target_delta or (is_image and frame_count == 0):
        if is_video and cap:
            ret, frame = cap.read()
            if ret:
                new_sequence_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                playback_state_ref["current_frame"] = new_sequence_frame_index
                read_successful = True
            else:
                if loop_video:
                    print("Looping video.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    new_sequence_frame_index = 0
                    playback_state_ref["current_frame"] = 0
                    ret, frame = cap.read()
                    if ret:
                        read_successful = True
                    else:
                        end_of_stream = True
                else:
                    end_of_stream = True
        elif is_glb_sequence:
            if new_sequence_frame_index >= len(glb_files):
                if loop_video:
                    print("Looping GLB sequence.")
                    new_sequence_frame_index = 0
                    playback_state_ref["current_frame"] = 0
                else:
                    end_of_stream = True

            if not end_of_stream:
                current_glb_path = glb_files[new_sequence_frame_index]
                points_xyz_np, colors_np = load_glb(current_glb_path)
                if points_xyz_np is not None:
                    ret = True
                    playback_state_ref["current_frame"] = new_sequence_frame_index + 1
                    new_sequence_frame_index += 1
                    read_successful = True
                else:
                    print(f"Error loading GLB frame: {current_glb_path}")
                    end_of_stream = True
        elif is_image:
             if frame_count == 0:
                 frame = image_frame
                 if frame is not None:
                     ret = True
                     read_successful = True
                 else:
                     ret = False
             else:
                 end_of_stream = True

        if read_successful:
            frame_read_delta_t = current_time - last_frame_read_time
            last_frame_read_time_ref[0] = current_time

        if end_of_stream:
            print(f"End of {input_mode}: {frame_source_name}")
            data_queue.put(("status", f"Finished processing: {frame_source_name}"))
            return None, None, None, False, new_sequence_frame_index, frame_read_delta_t, True

        return frame, points_xyz_np, colors_np, ret, new_sequence_frame_index, frame_read_delta_t, False
    else:
        sleep_time = target_delta - time_since_last_read
        time.sleep(max(0.001, sleep_time))
        return None, None, None, False, new_sequence_frame_index, frame_read_delta_t, False


def _apply_sharpening(rgb_frame, edge_params_ref):
    """Applies sharpening to the RGB frame if enabled."""
    enable_sharpening = edge_params_ref.get("enable_sharpening", False)
    sharpness_amount = edge_params_ref.get("sharpness", 1.5)
    if enable_sharpening and sharpness_amount > 0 and rgb_frame is not None:
        try:
            blurred = cv2.GaussianBlur(rgb_frame, (0, 0), 3)
            alpha = 1.0 + (sharpness_amount - 1.0) * 0.5
            beta = - (sharpness_amount - 1.0) * 0.5
            sharpened = cv2.addWeighted(rgb_frame, alpha, blurred, beta, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        except Exception as e_sharp:
            print(f"Warning: Error during sharpening: {e_sharp}")
            return rgb_frame
    else:
        return rgb_frame


def _run_model_inference(model, frame, device):
    """Runs the UniK3D model inference."""
    if model is None or frame is None:
        return None
    try:
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
        frame_tensor = frame_tensor.unsqueeze(0)
        with torch.no_grad():
            predictions = model.infer(frame_tensor)
        return predictions
    except Exception as e_infer:
        print(f"Error during model inference: {e_infer}")
        traceback.print_exc()
        return None


def _process_inference_results(predictions, rgb_frame_processed, device, edge_params_ref, prev_depth_map, smoothed_mean_depth, smoothed_points_xyz, frame_h, frame_w, input_mode):
    """Processes model predictions to generate point clouds, colors, and debug maps."""
    points_xyz_np_processed = None
    colors_np_processed = None
    num_vertices = 0
    scaled_depth_map_for_queue = None
    edge_map_viz = None
    smoothing_map_viz = None
    new_prev_depth_map = prev_depth_map
    new_smoothed_mean_depth = smoothed_mean_depth
    new_smoothed_points_xyz = smoothed_points_xyz
    points_xyz = None
    newly_calculated_bias_map_for_queue = None
    main_screen_coeff_viz_for_queue = None

    # Get WPT parameters and current render mode
    render_mode = edge_params_ref.get("render_mode", DEFAULT_SETTINGS["render_mode"])
    wavelet_packet_type = edge_params_ref.get("wavelet_packet_type", DEFAULT_SETTINGS["wavelet_packet_type"])
    wavelet_packet_window_size = edge_params_ref.get("wavelet_packet_window_size", DEFAULT_SETTINGS["wavelet_packet_window_size"])
    
    # Get bias map and toggle state
    apply_bias = edge_params_ref.get("apply_depth_bias", False)
    bias_map = edge_params_ref.get("depth_bias_map", None)

    if predictions is None:
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz, newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue

    current_depth_map = None
    raw_model_depth_for_bias_capture = None

    if 'depth' in predictions:
        current_depth_map = predictions['depth'].squeeze().float()
        raw_model_depth_for_bias_capture = current_depth_map.clone()

        # Check for Bias Capture Trigger
        pending_bias_capture_type = edge_params_ref.get("trigger_bias_capture", None)
        if pending_bias_capture_type and raw_model_depth_for_bias_capture is not None:
            print(f"INFO: Inference thread processing '{pending_bias_capture_type}' bias capture request.")
            if pending_bias_capture_type == "mean_plane":
                try:
                    mean_val = torch.mean(raw_model_depth_for_bias_capture.float())
                    flat_ref_plane = torch.full_like(raw_model_depth_for_bias_capture, mean_val)
                    newly_calculated_bias_map_for_queue = raw_model_depth_for_bias_capture - flat_ref_plane
                    print(f"  Inference: Raw Depth for Bias - Min: {torch.min(raw_model_depth_for_bias_capture):.3f}, Max: {torch.max(raw_model_depth_for_bias_capture):.3f}, Mean: {mean_val:.3f}")
                    print(f"  Inference: Calculated Bias Map - Min: {torch.min(newly_calculated_bias_map_for_queue):.3f}, Max: {torch.max(newly_calculated_bias_map_for_queue):.3f}, Mean: {torch.mean(newly_calculated_bias_map_for_queue.float()):.3f}")
                except Exception as e_capture:
                    print(f"ERROR in inference thread during bias map calculation: {e_capture}")
                    traceback.print_exc()
                    newly_calculated_bias_map_for_queue = None
            elif pending_bias_capture_type == "smoothed_plane":
                try:
                    # Calculate smoothed reference plane using Gaussian blur
                    D_captured_np = raw_model_depth_for_bias_capture.cpu().numpy()
                    smooth_kernel_size = 51 
                    if smooth_kernel_size > D_captured_np.shape[0] or smooth_kernel_size > D_captured_np.shape[1]:
                        print(f"WARNING: Bias smooth kernel ({smooth_kernel_size}) > image ({D_captured_np.shape}). Clamping kernel size.")
                        smooth_kernel_size = min(D_captured_np.shape[0], D_captured_np.shape[1])
                        if smooth_kernel_size % 2 == 0: smooth_kernel_size -= 1 
                        smooth_kernel_size = max(1, smooth_kernel_size)
                    
                    D_smoothed_np = cv2.GaussianBlur(D_captured_np, (smooth_kernel_size, smooth_kernel_size), 0)
                    D_smoothed = torch.from_numpy(D_smoothed_np).to(device).type_as(raw_model_depth_for_bias_capture)
                    newly_calculated_bias_map_for_queue = raw_model_depth_for_bias_capture - D_smoothed
                    
                    print(f"  Inference: Raw Depth for Bias - Min: {torch.min(raw_model_depth_for_bias_capture):.3f}, Max: {torch.max(raw_model_depth_for_bias_capture):.3f}")
                    print(f"  Inference: Smoothed Depth - Min: {torch.min(D_smoothed):.3f}, Max: {torch.max(D_smoothed):.3f}")
                    print(f"  Inference: Calculated Bias Map - Min: {torch.min(newly_calculated_bias_map_for_queue):.3f}, Max: {torch.max(newly_calculated_bias_map_for_queue):.3f}")
                except Exception as e_capture:
                    print(f"ERROR in inference thread during smoothed bias map calculation: {e_capture}")
                    traceback.print_exc()
                    newly_calculated_bias_map_for_queue = None

        # Apply Bias Correction (if enabled and valid)
        if apply_bias and bias_map is not None and current_depth_map is not None:
            if bias_map.shape == current_depth_map.shape:
                if bias_map.device != current_depth_map.device:
                    bias_map = bias_map.to(current_depth_map.device)
                
                original_min = torch.min(current_depth_map).item()
                corrected_depth = current_depth_map - bias_map
                corrected_depth = torch.clamp(corrected_depth, min=0.01)
                corrected_min = torch.min(corrected_depth).item()
                print(f"DEBUG: Applied depth bias. Original Min: {original_min:.3f}, Corrected Min: {corrected_min:.3f}")
                current_depth_map = corrected_depth
            else:
                print(f"Warning: Skipping bias correction - bias map shape {bias_map.shape} != current depth shape {current_depth_map.shape}")

        # Apply WPT if render_mode is 3 (Wavelet/FFT)
        if render_mode == 3 and current_depth_map is not None and current_depth_map.numel() > 0:
            print("DEBUG: Applying WPT to current_depth_map for render_mode 3")
            try:
                d_in = current_depth_map.unsqueeze(0).unsqueeze(0)
                J = int(math.log2(max(1, wavelet_packet_window_size)))
                min_dim = min(d_in.shape[-2], d_in.shape[-1])
                max_J_possible = int(math.log2(min_dim)) if min_dim > 0 else 0
                if J > max_J_possible:
                    print(f"Warning: Requested J={J} for WPT is too large for depth map size ({d_in.shape[-2]}x{d_in.shape[-1]}). Clamping to J={max_J_possible}.")
                    J = max_J_possible

                if J > 0 and DWTForward is not None:
                    dwt_op = DWTForward(J=J, wave=wavelet_packet_type, mode='zero').to(device)
                    idwt_op = DWTInverse(wave=wavelet_packet_type, mode='zero').to(device)
                    Yl, Yh = dwt_op(d_in)
                    wpt_processed_depth = idwt_op((Yl, Yh)).squeeze(0).squeeze(0)
                    
                    # Compute per-pixel WPT energy for Gaussian splatting
                    lh = Yh[-1][0, 0, 0, :, :].abs().cpu().numpy()
                    hl = Yh[-1][0, 0, 1, :, :].abs().cpu().numpy()
                    hh = Yh[-1][0, 0, 2, :, :].abs().cpu().numpy()
                    energy = (lh**2 + hl**2 + hh**2)
                    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-7)
                    size_map = 2.0 + 8.0 * energy_norm
                    opacity_map = 0.2 + 0.8 * energy_norm
                    lh_n = (lh - lh.min()) / (lh.max() - lh.min() + 1e-7)
                    hl_n = (hl - hl.min()) / (hl.max() - hl.min() + 1e-7)
                    hh_n = (hh - hh.min()) / (hh.max() - hh.min() + 1e-7)
                    color_map = np.stack([lh_n, hl_n, hh_n], axis=-1)
                    
                    # Get 3D positions
                    H, W = current_depth_map.shape
                    input_fov_deg = edge_params_ref.get('input_camera_fov', 60.0)
                    fov_y_rad = math.radians(input_fov_deg)
                    f_y = H / (2 * math.tan(fov_y_rad / 2.0))
                    f_x = f_y
                    cx = W / 2.0
                    cy = H / 2.0
                    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    depth_values = current_depth_map.cpu().numpy()
                    X_cam = (ii + 0.5 - cx) * depth_values / f_x
                    Y_cam = (jj + 0.5 - cy) * depth_values / f_y
                    Z_cam = depth_values
                    points_xyz = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
                    
                    points_xyz_np_processed = points_xyz.reshape(-1, 3)
                    colors_np_processed = color_map.reshape(-1, 3)
                    sizes_np_processed = size_map.flatten()
                    opacities_np_processed = opacity_map.flatten()
                    num_vertices = points_xyz_np_processed.shape[0]
                    
                    coeff_viz_bgr_np = cv2.merge((hh_n, hl_n, lh_n))
                    coeff_viz_bgr_np = (coeff_viz_bgr_np * 255).astype(np.uint8)
                    main_screen_coeff_viz_for_queue = coeff_viz_bgr_np
                else:
                    print("Warning: Cannot apply WPT, J (levels) is 0 or less.")
            except ImportError:
                print("ERROR: pytorch_wavelets not available for WPT processing. Skipping.")
            except Exception as e_wpt:
                print(f"ERROR during WPT processing: {e_wpt}")
                traceback.print_exc()

        # Generate Coefficient Visualization if render_mode == 3
        if render_mode == 3 and current_depth_map is not None and current_depth_map.numel() > 0:
            print("DEBUG CoeffViz: Entered block for render_mode 3.")
            try:
                wavelet_type_viz = edge_params_ref.get("wavelet_packet_type", DEFAULT_SETTINGS["wavelet_packet_type"])
                J_viz = 1
                
                temp_h, temp_w = current_depth_map.shape
                print(f"DEBUG CoeffViz: current_depth_map shape: {temp_h}x{temp_w}")
                min_dim_viz = min(temp_h, temp_w)
                max_J_possible_viz = int(math.log2(min_dim_viz)) if min_dim_viz > 0 else 0
                actual_J_viz = min(J_viz, max_J_possible_viz)
                print(f"DEBUG CoeffViz: J_viz={J_viz}, actual_J_viz={actual_J_viz}")

                if actual_J_viz > 0 and DWTForward is not None:
                    dwt_op_viz = DWTForward(J=actual_J_viz, wave=wavelet_type_viz, mode='zero').to(device)
                    depth_tensor_for_dwt = current_depth_map.unsqueeze(0).unsqueeze(0)
                    _, Yh_viz = dwt_op_viz(depth_tensor_for_dwt)

                    lh_coeffs = Yh_viz[0][0, 0, 0, :, :]
                    hl_coeffs = Yh_viz[0][0, 0, 1, :, :]
                    hh_coeffs = Yh_viz[0][0, 0, 2, :, :]
                    print(f"DEBUG CoeffViz: Coeff shapes: LH={lh_coeffs.shape}, HL={hl_coeffs.shape}, HH={hh_coeffs.shape}")

                    target_size_viz = (frame_h, frame_w)
                    print(f"DEBUG CoeffViz: Target viz size: {target_size_viz}")

                    def process_coeff_band_for_viz(band_tensor_func):
                        resized_band = torch.nn.functional.interpolate(
                            band_tensor_func.unsqueeze(0).unsqueeze(0),
                            size=target_size_viz, mode='bilinear', align_corners=False
                        ).squeeze().cpu().numpy()
                        min_v, max_v = np.min(resized_band), np.max(resized_band)
                        norm_band = (resized_band - min_v) / (max_v - min_v + 1e-7) if (max_v - min_v) > 1e-7 else np.zeros_like(resized_band)
                        return (norm_band * 255).astype(np.uint8)

                    r_c = process_coeff_band_for_viz(torch.abs(lh_coeffs))
                    g_c = process_coeff_band_for_viz(torch.abs(hl_coeffs))
                    b_c = process_coeff_band_for_viz(torch.abs(hh_coeffs))
                    print(f"DEBUG CoeffViz: Processed band shapes: R={r_c.shape}, G={g_c.shape}, B={b_c.shape}")
                    
                    coeff_viz_bgr_np = cv2.merge((b_c, g_c, r_c))

                    depth_for_mod_np_viz = current_depth_map.cpu().numpy()
                    min_d_viz, max_d_viz = np.min(depth_for_mod_np_viz), np.max(depth_for_mod_np_viz)
                    norm_depth_mod_viz = (depth_for_mod_np_viz - min_d_viz) / (max_d_viz - min_d_viz + 1e-7) if (max_d_viz - min_d_viz) > 1e-7 else np.ones_like(depth_for_mod_np_viz)
                    
                    final_viz_bgr_np = coeff_viz_bgr_np * norm_depth_mod_viz[:, :, np.newaxis]
                    main_screen_coeff_viz_for_queue = np.clip(final_viz_bgr_np, 0, 255).astype(np.uint8)
                    print(f"DEBUG CoeffViz: Successfully generated main_screen_coeff_viz_for_queue with shape {main_screen_coeff_viz_for_queue.shape}")
                else:
                    print("DEBUG CoeffViz: actual_J_viz <= 0. Cannot generate coefficient viz.")
                    main_screen_coeff_viz_for_queue = None
            except Exception as e_coeff_viz:
                print(f"ERROR CoeffViz: Error generating coefficient visualization for main screen: {e_coeff_viz}")
                traceback.print_exc()
                main_screen_coeff_viz_for_queue = None
        else:
            if render_mode == 3:
                 print(f"DEBUG CoeffViz: Skipped main screen coeff viz. current_depth_map valid: {current_depth_map is not None and current_depth_map.numel() > 0}")
            main_screen_coeff_viz_for_queue = None

        # Process depth with planar or spherical projection
        if edge_params_ref.get('planar_projection', False):
            H_orig, W_orig = (480, 640)
            if current_depth_map is not None:
                 H_orig, W_orig = current_depth_map.shape

            if current_depth_map is not None:
                H, W = current_depth_map.shape
            else:
                H, W = H_orig, W_orig
            
            input_fov_deg = edge_params_ref.get('input_camera_fov', 60.0)
            if input_fov_deg is None:
                 print("WARNING: input_fov_deg was None in edge_params_ref! Defaulting to 60.0.")
                 input_fov_deg = 60.0

            fov_y_rad = torch.tensor(math.radians(input_fov_deg), device=device, dtype=torch.float32)
            f_y = H / (2 * torch.tan(fov_y_rad / 2.0))
            f_x = f_y
            cx = torch.tensor(W / 2.0, device=device, dtype=torch.float32)
            cy = torch.tensor(H / 2.0, device=device, dtype=torch.float32)
            jj = torch.arange(0, H, device=device, dtype=torch.float32)
            ii = torch.arange(0, W, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(jj, ii, indexing='ij')
            depth_values = current_depth_map
            X_cam = (grid_x + 0.5 - cx) * depth_values / f_x
            Y_cam = (grid_y + 0.5 - cy) * depth_values / f_y
            Z_cam = depth_values
            points_xyz = torch.stack([X_cam, Y_cam, Z_cam], dim=-1)

        elif 'rays' in predictions:
            rays = predictions['rays'].squeeze()
            if rays.ndim == 3 and rays.shape[0] == 3:
                rays = rays.permute(1, 2, 0)
            
            if rays is not None and rays.numel() > 0 and rays.ndim == 3 and rays.shape[-1] == 3:
                rays_norm_val = torch.norm(rays, p=2, dim=-1, keepdim=True)
                rays_norm_val[rays_norm_val < 1e-6] = 1e-6
                rays = rays / rays_norm_val

            # Force constant depth for spherical case
            forced_depth_value_spherical = 5.0
            H_sph, W_sph = (480, 640)
            if current_depth_map is not None:
                H_sph, W_sph = current_depth_map.shape
                print(f"DIAGNOSTIC: Spherical mode - Overriding model depth with constant {forced_depth_value_spherical}")
            elif rays is not None and rays.ndim == 3:
                H_sph, W_sph, _ = rays.shape
                print(f"DIAGNOSTIC: Spherical mode - Model provided no depth, creating constant depth {forced_depth_value_spherical} based on ray dimensions.")
            else:
                 print(f"DIAGNOSTIC: Spherical mode - Cannot determine shape for forced depth. Skipping override.")
            
            if H_sph > 0 and W_sph > 0:
                 current_depth_map = torch.full((H_sph, W_sph), forced_depth_value_spherical, device=device, dtype=torch.float32)

            if current_depth_map is not None and rays is not None and current_depth_map.shape == rays.shape[:2]:
                depth_to_multiply = current_depth_map.unsqueeze(-1)
                points_xyz = rays * depth_to_multiply
            else:
                print(f"Warning: Spherical rays shape {rays.shape if rays is not None else 'N/A'} and overridden/original depth map shape {current_depth_map.shape if current_depth_map is not None else 'N/A'} mismatch or depth missing.")
        else:
            print("Warning: Depth map present but no UniK3D rays found and planar projection is off. Cannot generate points from depth.")
    elif 'rays' in predictions:
        print("DIAGNOSTIC: Only rays found, no initial depth. Spherical mode will use forced constant depth.")
        rays = predictions['rays'].squeeze()
        if rays.ndim == 3 and rays.shape[0] == 3:
            rays = rays.permute(1, 2, 0)
        
        if rays is not None and rays.numel() > 0 and rays.ndim == 3 and rays.shape[-1] == 3:
            rays_norm_val = torch.norm(rays, p=2, dim=-1, keepdim=True)
            rays_norm_val[rays_norm_val < 1e-6] = 1e-6
            rays = rays / rays_norm_val

            forced_depth_value_spherical_no_init_depth = 5.0
            H_rays, W_rays, _ = rays.shape
            current_depth_map_generated = torch.full((H_rays, W_rays), forced_depth_value_spherical_no_init_depth, device=device, dtype=torch.float32)
            depth_to_multiply = current_depth_map_generated.unsqueeze(-1)
            points_xyz = rays * depth_to_multiply
            print(f"DIAGNOSTIC: Spherical - generated points with forced depth {forced_depth_value_spherical_no_init_depth}")
        else:
            print("Warning: Only rays found, but rays are invalid.")

    elif "points" in predictions:
        points_xyz = predictions["points"]
        if points_xyz.ndim == 3 and points_xyz.shape[0] == 1 and points_xyz.shape[-1] == 3:
            points_xyz = points_xyz.squeeze(0)
    else:
        print("Warning: No points or depth found in UniK3D predictions.")

    if points_xyz is None or points_xyz.numel() == 0:
        print("Warning: No valid points generated from predictions.")
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz, newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue

    # Edge Detection & Depth Processing
    combined_edge_map = None
    depth_gradient_map = None
    per_pixel_depth_motion_map = None
    final_alpha_map = None

    if current_depth_map is not None:
        try:
            depth_thresh1 = edge_params_ref.get("depth_edge_threshold1", 50.0)
            depth_thresh2 = edge_params_ref.get("depth_edge_threshold2", 150.0)
            rgb_thresh1 = edge_params_ref.get("rgb_edge_threshold1", 50.0)
            rgb_thresh2 = edge_params_ref.get("rgb_edge_threshold2", 150.0)

            depth_np_u8 = (torch.clamp(current_depth_map / 10.0, 0.0, 1.0) * 255).byte().cpu().numpy()
            depth_edge_map = cv2.Canny(depth_np_u8, depth_thresh1, depth_thresh2)

            grad_x = cv2.Sobel(depth_np_u8, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_np_u8, cv2.CV_64F, 0, 1, ksize=3)
            depth_gradient_map_np = cv2.magnitude(grad_x, grad_y)
            if np.max(depth_gradient_map_np) > 1e-6:
                depth_gradient_map_np = depth_gradient_map_np / np.max(depth_gradient_map_np)
            depth_gradient_map = torch.from_numpy(depth_gradient_map_np).float().to(device)

            gray_frame = cv2.cvtColor(rgb_frame_processed, cv2.COLOR_RGB2GRAY)
            rgb_edge_map = cv2.Canny(gray_frame, rgb_thresh1, rgb_thresh2)

            combined_edge_map = cv2.bitwise_or(depth_edge_map, rgb_edge_map)
            edge_map_viz = cv2.cvtColor(combined_edge_map, cv2.COLOR_GRAY2RGB)

            # Depth motion calculation
            if new_prev_depth_map is not None and current_depth_map.shape == new_prev_depth_map.shape:
                depth_diff = torch.abs(current_depth_map - new_prev_depth_map)
                depth_motion_threshold = 0.1
                per_pixel_depth_motion = torch.clamp(depth_diff / depth_motion_threshold, 0.0, 1.0)
                per_pixel_depth_motion_map = per_pixel_depth_motion

            new_prev_depth_map = current_depth_map.clone()
            scaled_depth_map_for_queue = current_depth_map

        except Exception as e_edge:
            print(f"Error during edge/depth processing: {e_edge}")
            combined_edge_map = None
            edge_map_viz = None
            depth_gradient_map = None
            per_pixel_depth_motion_map = None
            scaled_depth_map_for_queue = current_depth_map

    # Temporal Smoothing
    enable_point_smoothing = edge_params_ref.get("enable_point_smoothing", False)
    if enable_point_smoothing:
        if new_smoothed_points_xyz is None or new_smoothed_points_xyz.shape != points_xyz.shape:
            print("Initializing/Resetting smoothing state.")
            new_smoothed_points_xyz = points_xyz.clone()
            if points_xyz.ndim == 3: 
                final_alpha_map = torch.ones_like(points_xyz[:,:,0])
            elif points_xyz.ndim == 2: 
                final_alpha_map = torch.ones(points_xyz.shape[0], device=device)
        else:
            min_alpha_points = edge_params_ref.get("min_alpha_points", 0.0)
            max_alpha_points = edge_params_ref.get("max_alpha_points", 1.0)

            if per_pixel_depth_motion_map is not None and per_pixel_depth_motion_map.shape == points_xyz.shape[:2]:
                 motion_factor_points = per_pixel_depth_motion_map ** 2.0
                 base_alpha_map = min_alpha_points + (max_alpha_points - min_alpha_points) * motion_factor_points
            else:
                 base_alpha_map = torch.full_like(points_xyz[:,:,0], max_alpha_points) if points_xyz.ndim == 3 else torch.full((points_xyz.shape[0],), max_alpha_points, device=device)

            enable_edge_aware = edge_params_ref.get("enable_edge_aware_smoothing", False)
            if enable_edge_aware and combined_edge_map is not None and combined_edge_map.shape == base_alpha_map.shape:
                edge_mask_tensor = torch.from_numpy(combined_edge_map / 255.0).float().to(device)
                edge_influence = edge_params_ref.get("edge_smoothing_influence", 0.0)
                local_influence = edge_influence

                grad_influence_scale = edge_params_ref.get("gradient_influence_scale", 1.0)
                if depth_gradient_map is not None and depth_gradient_map.shape == base_alpha_map.shape:
                    local_influence = edge_influence * torch.clamp(depth_gradient_map * grad_influence_scale, 0.0, 1.0)

                final_alpha_map = torch.lerp(base_alpha_map, torch.ones_like(base_alpha_map), edge_mask_tensor * local_influence)
            else:
                final_alpha_map = base_alpha_map

            final_alpha_map_unsqueezed = final_alpha_map.unsqueeze(-1) if final_alpha_map.ndim == 2 else final_alpha_map.unsqueeze(-1)
            new_smoothed_points_xyz = final_alpha_map_unsqueezed * points_xyz + (1.0 - final_alpha_map_unsqueezed) * new_smoothed_points_xyz
    else:
        new_smoothed_points_xyz = points_xyz
        if points_xyz.ndim == 3: 
            final_alpha_map = torch.ones_like(points_xyz[:,:,0])
        elif points_xyz.ndim == 2: 
            final_alpha_map = torch.ones(points_xyz.shape[0], device=device)

    # Point Thickening (Operates on smoothed points, BEFORE coord flip)
    points_to_thicken = new_smoothed_points_xyz
    enable_thickening = edge_params_ref.get("enable_point_thickening", False)
    num_duplicates = edge_params_ref.get("thickening_duplicates", 0)
    variance = edge_params_ref.get("thickening_variance", 0.0)
    depth_bias = edge_params_ref.get("thickening_depth_bias", 0.0)

    if enable_thickening and num_duplicates > 0 and points_to_thicken.numel() > 0:
        try:
            if points_to_thicken.is_cuda:
                points_np = points_to_thicken.squeeze().cpu().numpy()
            else:
                points_np = points_to_thicken.squeeze().numpy()

            if points_np.ndim == 3:
                 N = points_np.shape[0] * points_np.shape[1]
                 points_np = points_np.reshape(N, 3)
            elif points_np.ndim == 2 and points_np.shape[1] == 3:
                 N = points_np.shape[0]
            else:
                 raise ValueError(f"Unexpected points shape for thickening: {points_np.shape}")

            if rgb_frame_processed is not None and rgb_frame_processed.ndim == 3:
                colors_np = (rgb_frame_processed.reshape(N, 3).astype(np.float32) / 255.0)
            else:
                colors_np = np.ones((N, 3), dtype=np.float32)

            norms = np.linalg.norm(points_np, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1e-6
            ray_directions = points_np / norms

            num_new_points = N * num_duplicates
            duplicate_points = np.zeros((num_new_points, 3), dtype=np.float32)
            duplicate_colors = np.zeros((num_new_points, 3), dtype=np.float32)

            random_offsets = np.random.normal(0.0, variance, size=(num_new_points, 3)).astype(np.float32)

            bias_magnitudes = np.random.uniform(0.0, depth_bias, size=(num_new_points, 1)).astype(np.float32)
            repeated_ray_dirs = np.repeat(ray_directions, num_duplicates, axis=0)
            biased_offsets = repeated_ray_dirs * bias_magnitudes

            repeated_original_points = np.repeat(points_np, num_duplicates, axis=0)
            duplicate_points = repeated_original_points + random_offsets + biased_offsets

            duplicate_colors = np.repeat(colors_np, num_duplicates, axis=0)

            points_xyz_np_thickened = np.vstack((points_np, duplicate_points))
            colors_np_thickened = np.vstack((colors_np, duplicate_colors))

        except Exception as e_thicken:
            print(f"Error during point thickening: {e_thicken}")
            traceback.print_exc()
            if new_smoothed_points_xyz.is_cuda:
                 points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().cpu().numpy()
            else:
                 points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().numpy()
            N_fallback = points_xyz_np_thickened.shape[0]
            if rgb_frame_processed is not None and rgb_frame_processed.ndim == 3 and rgb_frame_processed.size == N_fallback*3:
                 colors_np_thickened = (rgb_frame_processed.reshape(N_fallback, 3).astype(np.float32) / 255.0)
            else: 
                colors_np_thickened = np.ones((N_fallback, 3), dtype=np.float32)

    else:
        if new_smoothed_points_xyz.is_cuda:
             points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().cpu().numpy()
        else:
             points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().numpy()
        
        if points_xyz_np_thickened.ndim == 3:
            num_vertices_final = points_xyz_np_thickened.shape[0] * points_xyz_np_thickened.shape[1]
            points_xyz_np_processed = points_xyz_np_thickened.reshape(num_vertices_final, 3)
        elif points_xyz_np_thickened.ndim == 2:
            num_vertices_final = points_xyz_np_thickened.shape[0]
            points_xyz_np_processed = points_xyz_np_thickened
        else:
            print(f"Warning: Unexpected points shape after thickening/smoothing: {points_xyz_np_thickened.shape}")
            num_vertices_final = 0
            points_xyz_np_processed = np.array([], dtype=np.float32).reshape(0,3)
            colors_np_processed = np.array([], dtype=np.float32).reshape(0,3)

        # Define colors_np_thickened when thickening is disabled
        if rgb_frame_processed is not None and rgb_frame_processed.ndim == 3:
            colors_np_thickened = (rgb_frame_processed.reshape(num_vertices_final, 3).astype(np.float32) / 255.0)
        else:
            colors_np_thickened = np.ones((num_vertices_final, 3), dtype=np.float32)

        colors_np_processed = colors_np_thickened if num_vertices_final > 0 else np.array([], dtype=np.float32).reshape(0,3)
        num_vertices = num_vertices_final

    # Transform Coordinates for OpenGL/GLB standard (+X Right, +Y Up, -Z Forward)
    if points_xyz_np_processed.ndim >= 2 and points_xyz_np_processed.shape[-1] == 3:
        points_xyz_np_processed[..., 1] *= -1.0  # Flip Y (Down -> Up)
        points_xyz_np_processed[..., 2] *= -1.0  # Flip Z (Forward -> -Forward)

    return points_xyz_np_processed, colors_np_processed, num_vertices, \
           scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, \
           new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz, \
           newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue


def _handle_recording(points_xyz_np, colors_np, recording_state_ref, sequence_frame_index, recorded_frame_counter, is_video, is_glb_sequence, data_queue):
    """Handles saving the current frame to GLB if recording is active."""
    new_recorded_frame_counter = recorded_frame_counter
    is_recording = recording_state_ref.get("is_recording", False)
    output_dir = recording_state_ref.get("output_dir", "recording_output")

    if is_recording and points_xyz_np is not None:
        new_recorded_frame_counter += 1
        current_playback_index = sequence_frame_index -1 if (is_video or is_glb_sequence) else -1
        save_index = current_playback_index if current_playback_index >= 0 else new_recorded_frame_counter

        glb_filename = os.path.join(output_dir, f"frame_{save_index:05d}.glb")
        save_glb(glb_filename, points_xyz_np, colors_np)
        if new_recorded_frame_counter % 30 == 0:
            data_queue.put(("status", f"Recording frame {new_recorded_frame_counter}..."))

    return new_recorded_frame_counter


def _queue_results(data_queue, vertices_flat, colors_flat, num_vertices,
                   rgb_frame_orig, scaled_depth_map_for_queue, edge_map_viz,
                   smoothing_map_viz, t_capture, sequence_frame_index,
                   video_total_frames, current_recorded_count, frame_count,
                   frame_read_delta_t, depth_process_delta_t, latency_ms,
                   newly_calculated_bias_map_for_main_thread,
                   main_screen_coeff_viz_content):
    """Puts the processed data into the queue for the main thread."""
    # Always put data in queue - no blocking, no dropping
    data_queue.put((vertices_flat, colors_flat, num_vertices,
                   rgb_frame_orig,
                   scaled_depth_map_for_queue,
                   edge_map_viz,
                   smoothing_map_viz,
                   t_capture,
                   sequence_frame_index,
                   video_total_frames,
                   current_recorded_count,
                   frame_read_delta_t, depth_process_delta_t, latency_ms,
                   newly_calculated_bias_map_for_main_thread,
                   main_screen_coeff_viz_content))


# Main Inference Thread Function
def inference_thread_func(data_queue, exit_event,
                          model, device,
                          inference_interval,
                          scale_factor_ref,
                          edge_params_ref,
                          input_mode,
                          input_filepath,
                          playback_state,
                          recording_state,
                          live_processing_mode,
                          screen_capture_monitor_index,
                          target_inference_fps=10.0):
    """Main inference thread function."""
    
    print(f"Inference thread started. Mode: {input_mode}, File: {input_filepath if input_filepath else 'N/A'}, Monitor Idx: {screen_capture_monitor_index if input_mode=='Screen' else 'N/A'}")
    data_queue.put(("status", f"Inference thread started ({input_mode}) using pre-loaded model."))

    cap = None
    image_frame = None
    is_video = False
    is_image = False
    is_glb_sequence = False
    glb_files = []
    frame_source_name = "N/A"
    video_total_frames = 0
    video_fps = 30
    error_message = None

    try:
        if input_mode == "GLB Sequence":
            print("GLB Sequence mode: Skipping inference.")

        cap, image_frame, is_video, is_image, is_glb_sequence, glb_files, \
        frame_source_name, video_total_frames, video_fps, input_mode_returned, error_message = \
            _initialize_input_source(input_mode, input_filepath, data_queue, playback_state)

        if input_mode_returned != input_mode:
            print(f"Input mode auto-detected as: {input_mode_returned}")
            input_mode = input_mode_returned

        media_start_time = time.time()

        if error_message:
            return

        frame_count = 0
        sequence_frame_index = 0
        last_frame_read_time_ref = [time.time()]
        recorded_frame_counter = 0
        smoothed_points_xyz = None
        prev_depth_map = None
        smoothed_mean_depth = None
        prev_input_scale_factor_for_smoothing_reset = None
        last_depth_processed_time = time.time()
        
        # Add frame skipping control
        last_inference_time = time.time()
        target_inference_interval = 1.0 / target_inference_fps
        frames_skipped = 0
        
        # Add frame buffer for asynchronous capture
        frame_buffer = queue.Queue(maxsize=2)  # Small buffer
        capture_thread = None
        
        def camera_capture_worker():
            """Separate thread for camera capture to prevent blocking"""
            frames_captured = 0
            frames_dropped_in_capture = 0
            
            while not exit_event.is_set() and cap and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret:
                        frames_captured += 1
                        
                        # Always try to provide the freshest frame
                        # If buffer is full, drop the old frame and add the new one
                        try:
                            frame_buffer.put_nowait((frame, time.time()))
                        except queue.Full:
                            # Drop old frame and add new one
                            try:
                                frame_buffer.get_nowait()  # Remove old frame
                                frame_buffer.put_nowait((frame, time.time()))  # Add new frame
                                frames_dropped_in_capture += 1
                                
                                if frames_dropped_in_capture % 30 == 0:
                                    print(f"Camera capture: Dropped {frames_dropped_in_capture} old frames to keep buffer fresh")
                            except queue.Empty:
                                # Buffer became empty between checks, just add the frame
                                frame_buffer.put_nowait((frame, time.time()))
                    else:
                        time.sleep(0.001)  # Camera read failed, small delay
                except Exception as e:
                    print(f"Camera capture error: {e}")
                    time.sleep(0.01)
        
        # Start camera capture thread for live mode
        if input_mode == "Live" and is_video and cap:
            capture_thread = threading.Thread(target=camera_capture_worker, daemon=True)
            capture_thread.start()
            print(f"Started async camera capture thread for maximum inference speed")

        while not exit_event.is_set():
            frame = None
            points_xyz_np_loaded = None
            colors_np_loaded = None
            ret = False
            frame_read_delta_t = 0.0
            end_of_stream = False
            t_capture = time.time()
            newly_calculated_bias_map_for_queue = None # Reset per frame
            main_screen_coeff_viz_for_queue = None # Reset per frame

            # CRITICAL: For live mode, only process if frame is immediately available
            if input_mode == "Live" and is_video and cap:
                try:
                    frame, t_capture = frame_buffer.get_nowait()
                    ret = True
                    current_time = time.time()
                    frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                    last_frame_read_time_ref[0] = current_time
                    playback_state["current_frame"] = frame_count + 1
                except queue.Empty:
                    # No frame available, skip this iteration entirely
                    continue
            elif input_mode == "Screen" and is_video:
                try:
                    with mss.mss() as sct:
                        monitor_to_capture = sct.monitors[screen_capture_monitor_index] if screen_capture_monitor_index < len(sct.monitors) else sct.monitors[0]
                        sct_img = sct.grab(monitor_to_capture)
                    
                    img_arr = np.array(sct_img)
                    if img_arr.shape[2] == 4:
                        frame = img_arr[:, :, :3]
                    elif img_arr.shape[2] == 3:
                        frame = img_arr
                    else:
                        print(f"ERROR: Screen capture has unexpected shape {img_arr.shape}")
                        ret = False
                        frame = None

                    if frame is not None:
                        ret = True
                        current_time = time.time()
                        frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                        last_frame_read_time_ref[0] = current_time
                        sequence_frame_index = frame_count + 1
                        playback_state["current_frame"] = sequence_frame_index
                except Exception as e_screen:
                    print(f"Error capturing screen (Monitor {screen_capture_monitor_index}): {e_screen}")
                    try:
                        print(f"Attempting fallback capture...")
                        with mss.mss() as sct_fallback:
                            fallback_index = 1 if len(sct_fallback.monitors) > 1 else 0
                            print(f"Attempting fallback capture on monitor {fallback_index}...")
                            fallback_monitor = sct_fallback.monitors[fallback_index] if fallback_index < len(sct_fallback.monitors) else sct_fallback.monitors[0]
                            sct_img = sct_fallback.grab(fallback_monitor)
                        
                        img_arr = np.array(sct_img)
                        if img_arr.shape[2] == 4:
                            frame = img_arr[:, :, :3]
                        elif img_arr.shape[2] == 3:
                            frame = img_arr
                        else:
                            print(f"ERROR: Fallback screen capture has unexpected shape {img_arr.shape}")
                            ret = False
                            frame = None

                        if frame is not None:
                            ret = True
                            current_time = time.time()
                            frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                            last_frame_read_time_ref[0] = current_time
                            sequence_frame_index = frame_count + 1
                            playback_state["current_frame"] = sequence_frame_index
                            print(f"Fallback capture successful.")
                    except Exception as e_fallback:
                            print(f"Fallback screen capture failed: {e_fallback}")
                            ret = False
                            time.sleep(0.1)
                            frame = None
                            if not ret:
                                continue

            elif input_mode in ["File", "GLB Sequence"] and not is_image:
                if playback_state.get("restart", False):
                    media_start_time = time.time()
                    playback_state["restart"] = False

                elapsed = time.time() - media_start_time
                frame_idx = int(elapsed * video_fps)
                total_frames = video_total_frames
                loop_video = playback_state.get("loop", True)
                
                if total_frames > 0:
                    if frame_idx >= total_frames:
                        if loop_video:
                            frame_idx = frame_idx % total_frames
                        else:
                            end_of_stream = True

                    if not end_of_stream:
                        if is_video and cap:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                        elif is_glb_sequence:
                            if frame_idx < len(glb_files):
                                points_xyz_np_loaded, colors_np_loaded = load_glb(glb_files[frame_idx])
                                ret = points_xyz_np_loaded is not None
                            else:
                                end_of_stream = True

                        sequence_frame_index = frame_idx + 1
                        playback_state["current_frame"] = sequence_frame_index
                        current_time = time.time()
                        frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                        last_frame_read_time_ref[0] = current_time
            else:
                frame, points_xyz_np_loaded, colors_np_loaded, ret, \
                sequence_frame_index, frame_read_delta_t, end_of_stream = \
                    _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index,
                                       playback_state, last_frame_read_time_ref, video_fps,
                                       is_video, is_image, is_glb_sequence, image_frame,
                                       frame_count, data_queue, frame_source_name)

            if frame_count > 1 and sequence_frame_index <= 1 and input_mode != "Live":
                 if recorded_frame_counter > 0:
                    print("DEBUG: Resetting recorded frame counter due to loop/restart.")
                    recorded_frame_counter = 0

            if end_of_stream:
                break

            if not ret:
                continue

            frame_count += 1
            
            last_inference_time = time.time()
            print(f"Processing frame {frame_count} (Seq Idx: {sequence_frame_index})")

            rgb_frame_orig = None
            points_xyz_np_processed = None
            colors_np_processed = None
            num_vertices = 0
            scaled_depth_map_for_queue = None
            edge_map_viz = None
            smoothing_map_viz = None
            depth_process_delta_t = 0.0

            if is_glb_sequence:
                # --- Process Loaded GLB Data ---
                if points_xyz_np_loaded is not None:
                    points_xyz_np_processed = points_xyz_np_loaded
                    colors_np_processed = colors_np_loaded
                    num_vertices = points_xyz_np_processed.shape[0]
                    # Transform loaded GLB data to OpenGL standard (+X Right, +Y Up, -Z Forward)
                    if points_xyz_np_processed.ndim >= 2 and points_xyz_np_processed.shape[-1] == 3:
                         points_xyz_np_processed[..., 1] *= -1.0 # Flip Y
                         points_xyz_np_processed[..., 2] *= -1.0 # Flip Z
                    # Create dummy frame for debug view consistency
                    frame_h, frame_w = (100, 100) if num_vertices == 0 else (int(np.sqrt(num_vertices)), int(np.sqrt(num_vertices))) # Guess dims
                    rgb_frame_orig = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

                    # Mark depth processed time for GLB
                    current_time = time.time()
                    depth_process_delta_t = current_time - last_depth_processed_time
                    last_depth_processed_time = current_time
                else: # GLB load failed for this frame index
                    print(f"Warning: Failed to load GLB frame at index {sequence_frame_index-1 if sequence_frame_index > 0 else 0}. Skipping.")
                    # Don't 'continue' here, let it try the next frame index in the next loop iteration
                    # Reset processing results for this failed frame
                    points_xyz_np_processed = None
                    colors_np_processed = None
                    num_vertices = 0
                    # Still need to update depth processing time to avoid large spikes in FPS calc
                    current_time = time.time()
                    depth_process_delta_t = current_time - last_depth_processed_time
                    last_depth_processed_time = current_time

            else: # Live, Video File, Image File, Screen Share (Needs Inference)
                current_scale = scale_factor_ref[0]
                if current_scale > 0.1:
                    new_width = int(frame.shape[1] * current_scale)
                    new_height = int(frame.shape[0] * current_scale)
                    interpolation = cv2.INTER_AREA if current_scale < 1.0 else cv2.INTER_LINEAR
                    scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
                else:
                    scaled_frame = frame

                rgb_frame_orig = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                rgb_frame_processed = _apply_sharpening(rgb_frame_orig, edge_params_ref)
                frame_h, frame_w, _ = rgb_frame_processed.shape

                predictions = _run_model_inference(model, rgb_frame_processed, device)
                data_queue.put(("status", f"Processing results for frame {frame_count}..."))
                points_xyz_np_processed, colors_np_processed, num_vertices, \
                scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, \
                prev_depth_map, smoothed_mean_depth, smoothed_points_xyz, \
                newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue = \
                    _process_inference_results(predictions, rgb_frame_processed, device,
                                               edge_params_ref, prev_depth_map,
                                               smoothed_mean_depth, smoothed_points_xyz,
                                               frame_h, frame_w, input_mode)
                current_time = time.time()
                depth_process_delta_t = current_time - last_depth_processed_time
                last_depth_processed_time = current_time

            recorded_frame_counter = _handle_recording(points_xyz_np_processed, colors_np_processed,
                                                       recording_state, sequence_frame_index,
                                                       recorded_frame_counter, is_video, is_glb_sequence,
                                                       data_queue)

            latency_ms = (time.time() - t_capture) * 1000.0 if t_capture else 0.0
            vertices_flat = points_xyz_np_processed.flatten() if points_xyz_np_processed is not None and points_xyz_np_processed.size > 0 else None
            colors_flat = colors_np_processed.flatten() if colors_np_processed is not None and colors_np_processed.size > 0 else None

            # Always try to queue results - the _queue_results function handles dropping
            _queue_results(data_queue, vertices_flat, colors_flat, num_vertices,
                           rgb_frame_orig, scaled_depth_map_for_queue, edge_map_viz,
                           smoothing_map_viz, t_capture, sequence_frame_index,
                           video_total_frames, recorded_frame_counter, frame_count,
                           frame_read_delta_t, depth_process_delta_t, latency_ms,
                           newly_calculated_bias_map_for_queue,
                           main_screen_coeff_viz_for_queue)
            print(f"DEBUG: Queuing {num_vertices if vertices_flat is not None else 0} vertices (frame {frame_count}).")

            # No artificial delays - run at maximum speed

    except Exception as e_thread:
        print(f"Error in inference thread: {e_thread}")
        traceback.print_exc()
        data_queue.put(("error", str(e_thread)))
        data_queue.put(("status", "Inference thread error!"))
    finally:
        if cap and is_video:
            cap.release()
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
        print("Inference thread finished.")
        data_queue.put(("status", "Inference thread finished."))
