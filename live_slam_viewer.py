import pyglet
import pyglet.gl as gl
import numpy as np
import os
import argparse
import math
import threading
import time
import queue # For thread-safe communication
import cv2 # For camera capture
import torch # For UniK3D model and tensors
import json # For saving/loading settings

# Assuming unik3d is installed and importable
from unik3d.models import UniK3D

from pyglet.math import Mat4, Vec3
from pyglet.window import key # Import key for KeyStateHandler
import traceback # Import traceback
import pyglet.text # Import for Label
import imgui
from imgui.integrations.pyglet import create_renderer

# Default settings dictionary
DEFAULT_SETTINGS = {
    "render_mode": 2, # 0=Square, 1=Circle, 2=Gaussian
    "falloff_factor": 5.0,
    "saturation": 1.5,
    "brightness": 1.0,
    "contrast": 1.1,
    "sharpness": 1.1,
    "point_size_boost": 2.5,
    "input_scale_factor": 0.9,
    "enable_point_smoothing": True,
    "min_alpha_points": 0.0,
    "max_alpha_points": 1.0,
    "enable_edge_aware_smoothing": True,
    "depth_edge_threshold1": 50.0,
    "depth_edge_threshold2": 150.0,
    "rgb_edge_threshold1": 50.0,
    "rgb_edge_threshold2": 150.0,
    "edge_smoothing_influence": 0.7,
    "gradient_influence_scale": 1.0, # How much depth gradient scales edge influence
    "show_camera_feed": False,
    "show_depth_map": False,
    "show_edge_map": False,
    "show_smoothing_map": False,
}

# Simple shaders (Using 'vertices' instead of 'position')
vertex_source = """#version 150 core
    in vec3 vertices;
    in vec3 colors; // Expects normalized floats (0.0-1.0)

    out vec3 vertex_colors;

    uniform mat4 projection;
    uniform mat4 view;
    uniform float inputScaleFactor; // Controlled via ImGui
    uniform float pointSizeBoost;   // Controlled via ImGui

    void main() {
        gl_Position = projection * view * vec4(vertices, 1.0);
        vertex_colors = colors; // Pass color data through

        // --- Point Size based ONLY on distance from Origin ---
        float originDist = length(vertices);
        float baseScalingFactor = 1.0;
        float effectiveScaleFactor = max(0.1, inputScaleFactor);
        float pointSizePixels = (baseScalingFactor * originDist) / effectiveScaleFactor;
        float clampedSize = max(1.0, min(30.0, pointSizePixels));
        gl_PointSize = clampedSize * pointSizeBoost;
        // --- End Point Size Calculation ---
    }
"""

# Modified Fragment Shader with Controls
fragment_source = """#version 150 core
    in vec3 geom_colors; // Input from Geometry Shader
    in vec2 texCoord;    // Input texture coordinate from Geometry Shader
    out vec4 final_color;

    uniform int renderMode; // 0=Square, 1=Circle, 2=Gaussian
    uniform float falloffFactor; // For Gaussian
    uniform float saturation;
    uniform float brightness;
    uniform float contrast;
    uniform float sharpness; // Simple contrast boost for sharpening

    // Function to convert RGB to HSV
    vec3 rgb2hsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    // Function to convert HSV to RGB
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    void main() {
        // --- Image Processing ---
        vec3 processed_color = geom_colors;

        // Saturation, Brightness, Contrast (applied in HSV)
        vec3 hsv = rgb2hsv(processed_color);
        hsv.y = clamp(hsv.y * saturation, 0.0, 1.0); // Saturation
        hsv.z = clamp(hsv.z * brightness, 0.0, 1.0); // Brightness
        hsv.z = clamp(0.5 + (hsv.z - 0.5) * contrast, 0.0, 1.0); // Contrast
        processed_color = hsv2rgb(hsv);

        // --- Simple Sharpening via Contrast Boost ---
        vec3 hsv_sharp = rgb2hsv(processed_color);
        hsv_sharp.z = clamp(0.5 + (hsv_sharp.z - 0.5) * sharpness, 0.0, 1.0);
        processed_color = hsv2rgb(hsv_sharp);
        // --- End Sharpening ---


        // --- Shape & Alpha ---
        if (renderMode == 0) { // Square (Opaque)
            final_color = vec4(processed_color, 1.0);
        } else { // Circle or Gaussian
            vec2 coord = texCoord - vec2(0.5);
            float dist_sq = dot(coord, coord);

            if (dist_sq > 0.25) { // Discard if outside circle
                discard;
            }

            if (renderMode == 1) { // Circle (Opaque)
                final_color = vec4(processed_color, 1.0);
            } else { // Gaussian (renderMode == 2)
                // Calculate Gaussian alpha
                float alpha = exp(-4.0 * falloffFactor * dist_sq);
                // Premultiply color by alpha
                vec3 premultipliedRgb = processed_color * alpha;
                final_color = vec4(premultipliedRgb, alpha); // Output premultiplied RGB and alpha
            }
        }
    }
"""

# Geometry shader remains the same as before (outputting texCoord)
geometry_source = """#version 150 core
    layout (points) in;
    layout (triangle_strip, max_vertices = 4) out;

    in vec3 vertex_colors[]; // Receive from vertex shader
    out vec3 geom_colors;    // Pass color to fragment shader
    out vec2 texCoord;       // Pass texture coordinate to fragment shader

    uniform vec2 viewportSize; // To convert pixel size to clip space

    void main() {
        vec4 centerPosition = gl_in[0].gl_Position;
        float pointSize = gl_in[0].gl_PointSize; // Get size calculated in vertex shader

        // Calculate half-size in clip space coordinates
        float halfSizeX = pointSize / viewportSize.x;
        float halfSizeY = pointSize / viewportSize.y;

        // Emit 4 vertices for the quad
        gl_Position = centerPosition + vec4(-halfSizeX, -halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(0.0, 0.0); // Bottom-left
        EmitVertex();

        gl_Position = centerPosition + vec4( halfSizeX, -halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(1.0, 0.0); // Bottom-right
        EmitVertex();

        gl_Position = centerPosition + vec4(-halfSizeX,  halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(0.0, 1.0); // Top-left
        EmitVertex();

        gl_Position = centerPosition + vec4( halfSizeX,  halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(1.0, 1.0); // Top-right
        EmitVertex();

        EndPrimitive();
    }
"""

# --- Simple Texture Shader (For FBO rendering - if needed later) ---
# Keep these definitions in case FBO is revisited, but they aren't used now
texture_vertex_source = """#version 150 core
    in vec2 position;
    in vec2 texCoord_in;
    out vec2 texCoord;

    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        texCoord = texCoord_in;
    }
"""

texture_fragment_source = """#version 150 core
    in vec2 texCoord;
    out vec4 final_color;

    uniform sampler2D fboTexture;

    void main() {
        final_color = texture(fboTexture, texCoord);
    }
"""

# --- Inference Thread Function ---
def inference_thread_func(data_queue, exit_event, model_name, inference_interval,
                          scale_factor_ref, edge_params_ref):
    """Loads model, captures camera, runs inference, puts results in queue."""
    print("Inference thread started.")
    data_queue.put(("status", "Inference thread started..."))
    try:
        # --- Load Model ---
        print(f"Loading UniK3D model: {model_name}...")
        data_queue.put(("status", f"Loading model: {model_name}..."))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = UniK3D.from_pretrained(f"lpiccinelli/{model_name}")
        model = model.to(device)
        model.eval()
        print("Model loaded.")
        data_queue.put(("status", "Model loaded."))

        # --- Initialize Camera ---
        print("Initializing camera...")
        data_queue.put(("status", "Initializing camera..."))
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            data_queue.put(("error", "Could not open camera."))
            return
        print("Camera initialized.")
        data_queue.put(("status", "Camera initialized."))

        frame_count = 0
        last_inference_time = time.time()

        # --- Temporal Smoothing & Motion Init ---
        smoothed_points_xyz = None
        depth_motion_threshold = 0.1
        prev_depth_map = None
        smoothed_mean_depth = None
        prev_scale_factor = None
        # -----------------------------

        while not exit_event.is_set():
            t_capture = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                time.sleep(0.1)
                continue

            # --- Dynamically Scale Frame ---
            current_scale = scale_factor_ref[0]
            if current_scale > 0.1:
                new_width = int(frame.shape[1] * current_scale)
                new_height = int(frame.shape[0] * current_scale)
                interpolation = cv2.INTER_AREA if current_scale < 1.0 else cv2.INTER_LINEAR
                scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
            else:
                scaled_frame = frame

            frame_count += 1

            if frame_count % inference_interval == 0:
                current_scale = scale_factor_ref[0]
                if prev_scale_factor is not None and abs(current_scale - prev_scale_factor) > 1e-3:
                    print(f"Scale factor changed ({prev_scale_factor:.2f} -> {current_scale:.2f}). Resetting smoothing state.")
                    smoothed_points_xyz = None
                    prev_depth_map = None
                    smoothed_mean_depth = None
                prev_scale_factor = current_scale

                current_time = time.time()
                print(f"Running inference for frame {frame_count} (Time since last: {current_time - last_inference_time:.2f}s)")
                last_inference_time = current_time

                try:
                    data_queue.put(("status", f"Preprocessing frame {frame_count}..."))
                    rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                    frame_h, frame_w, _ = rgb_frame.shape
                    frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().to(device)
                    frame_tensor = frame_tensor.unsqueeze(0)

                    data_queue.put(("status", f"Running inference on frame {frame_count}..."))
                    with torch.no_grad():
                        predictions = model.infer(frame_tensor)

                    points_xyz = None
                    scaled_depth_map_for_queue = None
                    edge_map_viz = None
                    smoothing_map_viz = None

                    if 'rays' in predictions and 'depth' in predictions:
                        current_depth_map = predictions['depth'].squeeze()

                        # --- Edge Detection on Depth & RGB ---
                        combined_edge_map = None
                        depth_gradient_map = None
                        if current_depth_map is not None:
                            try:
                                depth_thresh1 = edge_params_ref["depth_threshold1"]
                                depth_thresh2 = edge_params_ref["depth_threshold2"]
                                rgb_thresh1 = edge_params_ref["rgb_threshold1"]
                                rgb_thresh2 = edge_params_ref["rgb_threshold2"]

                                depth_np_u8 = (torch.clamp(current_depth_map / 10.0, 0.0, 1.0) * 255).byte().cpu().numpy()

                                # Depth Edges (Canny)
                                depth_edge_map = cv2.Canny(depth_np_u8, depth_thresh1, depth_thresh2)

                                # Depth Gradient (Sobel)
                                grad_x = cv2.Sobel(depth_np_u8, cv2.CV_64F, 1, 0, ksize=3)
                                grad_y = cv2.Sobel(depth_np_u8, cv2.CV_64F, 0, 1, ksize=3)
                                depth_gradient_map = cv2.magnitude(grad_x, grad_y)
                                # Normalize gradient map 0-1 for influence calculation
                                if np.max(depth_gradient_map) > 1e-6:
                                    depth_gradient_map = depth_gradient_map / np.max(depth_gradient_map)

                                # RGB Edges (Canny)
                                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                                rgb_edge_map = cv2.Canny(gray_frame, rgb_thresh1, rgb_thresh2)

                                # Combine Edges
                                combined_edge_map = cv2.bitwise_or(depth_edge_map, rgb_edge_map)

                                # Create RGB visualization for queue
                                edge_map_viz = cv2.cvtColor(combined_edge_map, cv2.COLOR_GRAY2RGB)
                            except Exception as e_edge:
                                print(f"Error during edge detection: {e_edge}")
                                combined_edge_map = None
                                edge_map_viz = None
                                depth_gradient_map = None
                        # ---------------------------------

                        per_pixel_depth_motion_map = None
                        per_pixel_depth_motion = None
                        if prev_depth_map is not None and current_depth_map.shape == prev_depth_map.shape:
                            depth_diff = torch.abs(current_depth_map - prev_depth_map)
                            per_pixel_depth_motion = torch.clamp(depth_diff / depth_motion_threshold, 0.0, 1.0)
                            normalized_depth = torch.clamp(current_depth_map / 10.0, 0.0, 1.0)
                            distance_modulated_motion = per_pixel_depth_motion * (1.0 - normalized_depth)
                            per_pixel_depth_motion_map = distance_modulated_motion

                        prev_depth_map = current_depth_map.clone()

                        global_motion_unmodulated = 0.0
                        if per_pixel_depth_motion is not None:
                            global_motion_unmodulated = per_pixel_depth_motion.mean().item()

                        scaled_depth_map = current_depth_map
                        current_mean_depth = current_depth_map.mean().item()
                        min_alpha_scale = edge_params_ref.get("min_alpha_scale", 0.0) # Default if not set
                        max_alpha_scale = edge_params_ref.get("max_alpha_scale", 1.0) # Default if not set
                        motion_factor_scale = global_motion_unmodulated ** 2.0
                        adaptive_alpha_scale = min_alpha_scale + (max_alpha_scale - min_alpha_scale) * motion_factor_scale

                        if smoothed_mean_depth is None:
                            smoothed_mean_depth = current_mean_depth
                        else:
                            smoothed_mean_depth = adaptive_alpha_scale * current_mean_depth + (1.0 - adaptive_alpha_scale) * smoothed_mean_depth

                        if current_mean_depth > 1e-6:
                            scale_factor = smoothed_mean_depth / current_mean_depth
                            scaled_depth_map = current_depth_map * scale_factor

                        scaled_depth_map_for_queue = scaled_depth_map

                        rays = predictions['rays'].squeeze()
                        if rays.shape[0] == 3 and rays.ndim == 3:
                            rays = rays.permute(1, 2, 0)
                        depth_to_multiply = scaled_depth_map.unsqueeze(-1)
                        points_xyz = rays * depth_to_multiply
                        data_queue.put(("status", f"Calculated points for frame {frame_count}..."))

                    elif "points" in predictions:
                        data_queue.put(("status", f"Using direct points for frame {frame_count}..."))
                        points_xyz = predictions["points"]
                    else:
                        data_queue.put(("warning", f"No points/depth found for frame {frame_count}"))

                    if points_xyz is not None:
                        if smoothed_points_xyz is None:
                            smoothed_points_xyz = points_xyz.clone()
                        else:
                            if smoothed_points_xyz.shape == points_xyz.shape:
                                # --- Edge-Aware Smoothing ---
                                enable_point_smoothing = edge_params_ref["enable_point_smoothing"]
                                enable_edge_aware = edge_params_ref["enable_edge_aware"]
                                edge_influence = edge_params_ref["influence"]
                                min_alpha_points = edge_params_ref["min_alpha_points"]
                                max_alpha_points = edge_params_ref["max_alpha_points"]
                                grad_influence_scale = edge_params_ref["gradient_influence_scale"]

                                final_alpha_map = None # Initialize

                                if enable_point_smoothing and per_pixel_depth_motion_map is not None:
                                    if points_xyz.ndim == 3 and points_xyz.shape[:2] == per_pixel_depth_motion_map.shape:
                                        motion_factor_points = 0.0
                                        if per_pixel_depth_motion is not None:
                                             motion_factor_points = per_pixel_depth_motion ** 2.0

                                        base_alpha_map = min_alpha_points + (max_alpha_points - min_alpha_points) * motion_factor_points

                                        # Modulate alpha by combined edge map and gradient
                                        if enable_edge_aware and combined_edge_map is not None and combined_edge_map.shape == base_alpha_map.shape:
                                            edge_mask_tensor = torch.from_numpy(combined_edge_map / 255.0).float().to(device) # Normalize 0-1

                                            # Calculate local influence based on gradient
                                            local_influence = edge_influence # Default to global influence
                                            if depth_gradient_map is not None and depth_gradient_map.shape == base_alpha_map.shape:
                                                gradient_tensor = torch.from_numpy(depth_gradient_map).float().to(device)
                                                # Scale global influence by normalized gradient magnitude
                                                local_influence = edge_influence * torch.clamp(gradient_tensor * grad_influence_scale, 0.0, 1.0)

                                            # Increase alpha (reduce smoothing) near edges, modulated by gradient
                                            final_alpha_map = torch.lerp(base_alpha_map, torch.ones_like(base_alpha_map), edge_mask_tensor * local_influence)
                                        else:
                                            final_alpha_map = base_alpha_map # Use base alpha if disabled or edge map invalid

                                        final_alpha_map_unsqueezed = final_alpha_map.unsqueeze(-1)
                                        smoothed_points_xyz = final_alpha_map_unsqueezed * points_xyz + (1.0 - final_alpha_map_unsqueezed) * smoothed_points_xyz
                                    else:
                                        smoothed_points_xyz = points_xyz
                                        final_alpha_map = torch.ones_like(points_xyz[:,:,0]) if points_xyz.ndim == 3 else torch.ones(points_xyz.shape[0], device=device)
                                else:
                                    smoothed_points_xyz = points_xyz
                                    # Create alpha map visualization even if smoothing is off
                                    if points_xyz.ndim == 3:
                                        final_alpha_map = torch.ones_like(points_xyz[:,:,0])
                                    elif points_xyz.ndim == 2:
                                         final_alpha_map = torch.ones(points_xyz.shape[0], device=device)

                                # --- End Edge-Aware Smoothing ---

                                # --- Create Smoothing Map Visualization ---
                                if final_alpha_map is not None:
                                    try:
                                        # Ensure final_alpha_map has 2 dimensions for visualization
                                        if final_alpha_map.ndim == 1: # Handle case where points_xyz was (N,3)
                                            # Cannot directly visualize 1D alpha map as 2D image easily
                                            # Create a placeholder or skip visualization
                                            h, w = frame_h, frame_w # Use frame dimensions as fallback
                                            smoothing_map_viz = np.zeros((h, w, 3), dtype=np.uint8)
                                            print("Warning: Cannot visualize 1D smoothing alpha map.")
                                        else:
                                            smoothing_map_vis_np = (final_alpha_map.cpu().numpy() * 255).astype(np.uint8)
                                            smoothing_map_viz = cv2.cvtColor(smoothing_map_vis_np, cv2.COLOR_GRAY2RGB)
                                    except Exception as e_smooth_viz:
                                        print(f"Error creating smoothing map viz: {e_smooth_viz}")
                                        smoothing_map_viz = None
                                else:
                                     # Create black image if no alpha map generated
                                     h, w = frame_h, frame_w
                                     smoothing_map_viz = np.zeros((h, w, 3), dtype=np.uint8)


                            else:
                                print("Warning: Points tensor shape changed, resetting smoothing.")
                                smoothed_points_xyz = points_xyz.clone()
                        points_xyz_to_process = smoothed_points_xyz
                    else:
                        points_xyz_to_process = None

                    if points_xyz_to_process is None or points_xyz_to_process.numel() == 0:
                        continue

                    points_xyz_np = points_xyz_to_process.squeeze().cpu().numpy()

                    num_vertices = 0
                    try:
                        if points_xyz_np.ndim == 3 and points_xyz_np.shape[0] == 3:
                            points_xyz_np = np.transpose(points_xyz_np, (1, 2, 0))
                            num_vertices = points_xyz_np.shape[0] * points_xyz_np.shape[1]
                            points_xyz_np = points_xyz_np.reshape(num_vertices, 3)
                        elif points_xyz_np.ndim == 2 and points_xyz_np.shape[1] == 3:
                             num_vertices = points_xyz_np.shape[0]
                        elif points_xyz_np.ndim == 3 and points_xyz_np.shape[2] == 3:
                             num_vertices = points_xyz_np.shape[0] * points_xyz_np.shape[1]
                             points_xyz_np = points_xyz_np.reshape(num_vertices, 3)
                        else:
                            print(f"Warning: Unexpected points_xyz_np shape after processing: {points_xyz_np.shape}")
                    except Exception as e_reshape:
                            print(f"Error reshaping points_xyz_np: {e_reshape}")
                            num_vertices = 0

                    colors_np = None
                    if num_vertices > 0:
                        try:
                            # Ensure rgb_frame has dimensions before reshape
                            if rgb_frame is not None and rgb_frame.ndim == 3:
                                colors_np = rgb_frame.reshape(frame_h * frame_w, 3)
                                subsample_rate = 1
                                if subsample_rate > 1:
                                    points_xyz_np = points_xyz_np[::subsample_rate]
                                    colors_np = colors_np[::subsample_rate]
                                num_vertices = points_xyz_np.shape[0]

                                if colors_np.dtype == np.uint8:
                                    colors_np = colors_np.astype(np.float32) / 255.0
                                elif colors_np.dtype == np.float32:
                                    colors_np = np.clip(colors_np, 0.0, 1.0)
                                else:
                                    print(f"Warning: Unexpected color dtype {colors_np.dtype}, using white.")
                                    colors_np = None
                            else:
                                print("Warning: rgb_frame invalid for color extraction.")
                                colors_np = None

                        except Exception as e_color_subsample:
                            print(f"Error processing/subsampling colors: {e_color_subsample}")
                            colors_np = None
                            if num_vertices > 0:
                                points_xyz_np = points_xyz_np[::subsample_rate]
                                num_vertices = points_xyz_np.shape[0]

                    if num_vertices > 0:
                        vertices_flat = points_xyz_np.flatten()
                        if colors_np is not None:
                            colors_flat = colors_np.flatten()
                        else:
                            colors_np_white = np.ones((num_vertices, 3), dtype=np.float32)
                            colors_flat = colors_np_white.flatten()

                        # Put all data into queue
                        if not data_queue.full():
                                data_queue.put((vertices_flat, colors_flat, num_vertices,
                                                rgb_frame, # Pass the raw frame (numpy uint8)
                                                scaled_depth_map_for_queue, # Pass the depth tensor
                                                edge_map_viz, # Pass edge map visualization
                                                smoothing_map_viz, # Pass smoothing map visualization
                                                t_capture))
                        else:
                            print(f"Warning: Viewer queue full, dropping frame {frame_count}.")
                            data_queue.put(("status", f"Viewer queue full, dropping frame {frame_count}"))
                    else:
                        pass

                except Exception as e_infer:
                    print(f"Error during inference processing for frame {frame_count}: {e_infer}")
                    traceback.print_exc()

            time.sleep(0.005)

    except Exception as e_thread:
        print(f"Error in inference thread: {e_thread}")
        traceback.print_exc()
        data_queue.put(("error", str(e_thread)))
        data_queue.put(("status", "Inference thread error!"))
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Inference thread finished.")
        data_queue.put(("status", "Inference thread finished."))


# --- Viewer Class ---
class LiveViewerWindow(pyglet.window.Window):
    def __init__(self, model_name, inference_interval=1,
                 disable_point_smoothing=False,
                 *args, **kwargs):
        # Store args before calling super
        self._model_name = model_name
        self._inference_interval = inference_interval
        # Store the initial smoothing preference (used when starting thread)
        self._initial_disable_point_smoothing = disable_point_smoothing

        super().__init__(*args, **kwargs)

        self.vertex_list = None
        self.frame_count_display = 0
        self.current_point_count = 0
        self.last_update_time = time.time()
        self.point_cloud_fps = 0.0
        self.last_capture_timestamp = None

        # --- Control State Variables (Initialized in load_settings) ---
        self.render_mode = None
        self.falloff_factor = None
        self.saturation = None
        self.brightness = None
        self.contrast = None
        self.sharpness = None
        self.point_size_boost = None
        self.input_scale_factor = None
        self.enable_point_smoothing = None
        self.min_alpha_points = None
        self.max_alpha_points = None
        self.enable_edge_aware_smoothing = None
        self.depth_edge_threshold1 = None
        self.depth_edge_threshold2 = None
        self.rgb_edge_threshold1 = None
        self.rgb_edge_threshold2 = None
        self.edge_smoothing_influence = None
        self.gradient_influence_scale = None # New control
        self.show_camera_feed = None
        self.show_depth_map = None
        self.show_edge_map = None
        self.show_smoothing_map = None
        self.scale_factor_ref = None # Initialized in load_settings
        self.edge_params_ref = {} # Dictionary to pass edge params to thread

        # --- Debug View State ---
        self.latest_rgb_frame = None
        self.latest_depth_map_viz = None
        self.latest_edge_map = None
        self.latest_smoothing_map = None
        self.camera_texture = None
        self.depth_texture = None
        self.edge_texture = None
        self.smoothing_texture = None
        self.debug_textures_initialized = False

        # Load initial settings (this initializes control variables and refs)
        self.load_settings()

        # --- Status Display ---
        self.ui_batch = pyglet.graphics.Batch()
        self.status_message = "Initializing..."
        # Pyglet labels removed

        # --- Camera Setup ---
        self.camera_position = Vec3(0, 0, 5)
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.world_up_vector = Vec3(0, 1, 0)
        self._aspect_ratio = float(self.width) / self.height if self.height > 0 else 1.0
        self.move_speed = 2.0
        self.fast_move_speed = 6.0
        self.prev_camera_position = Vec3(self.camera_position.x, self.camera_position.y, self.camera_position.z)
        self.prev_camera_rotation_x = self.camera_rotation_x
        self.prev_camera_rotation_y = self.camera_rotation_y
        self.camera_motion_confidence = 0.0

        # --- Input Handlers ---
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.mouse_down = False

        # --- Geometry Shader Source ---
        # (Keep geometry_source definition as is)
        geometry_source = """#version 150 core
            layout (points) in;
            layout (triangle_strip, max_vertices = 4) out;

            in vec3 vertex_colors[]; // Receive from vertex shader (array because input is point)
            out vec3 geom_colors;    // Pass color to fragment shader
            out vec2 texCoord;       // Pass texture coordinate to fragment shader

            uniform vec2 viewportSize; // To convert pixel size to clip space

            void main() {
                vec4 centerPosition = gl_in[0].gl_Position;
                float pointSize = gl_in[0].gl_PointSize; // Get size calculated in vertex shader

                // Calculate half-size in clip space coordinates
                float halfSizeX = pointSize / viewportSize.x;
                float halfSizeY = pointSize / viewportSize.y;

                // Emit 4 vertices for the quad
                gl_Position = centerPosition + vec4(-halfSizeX, -halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                texCoord = vec2(0.0, 0.0); // Bottom-left
                EmitVertex();

                gl_Position = centerPosition + vec4( halfSizeX, -halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                texCoord = vec2(1.0, 0.0); // Bottom-right
                EmitVertex();

                gl_Position = centerPosition + vec4(-halfSizeX,  halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                texCoord = vec2(0.0, 1.0); // Top-left
                EmitVertex();

                gl_Position = centerPosition + vec4( halfSizeX,  halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                texCoord = vec2(1.0, 1.0); // Top-right
                EmitVertex();

                EndPrimitive();
            }
        """

        # Shader only (No Batch)
        try:
            print("DEBUG: Creating shaders...")
            vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
            frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
            geom_shader = pyglet.graphics.shader.Shader(geometry_source, 'geometry')
            self.shader_program = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader, geom_shader)
            print(f"DEBUG: Shader program created.")
        except pyglet.graphics.shader.ShaderException as e:
            print(f"FATAL: Shader compilation error: {e}")
            pyglet.app.exit()
            return
        except Exception as e:
             print(f"FATAL: Error during pyglet setup: {e}")
             pyglet.app.exit()
             return

        # Schedule the main update function
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)
        pyglet.clock.schedule_interval(self.update_camera, 1.0 / 60.0)

        # --- Threading Setup ---
        self._data_queue = queue.Queue(maxsize=5)
        self._exit_event = threading.Event()
        self.inference_thread = None

        # Start the inference thread
        # self.scale_factor_ref and self.edge_params_ref initialized in load_settings
        self.start_inference_thread(self._model_name, self._inference_interval) # Removed smoothing arg

        # Set up OpenGL state
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA) # Default blend func

        # --- ImGui Setup ---
        try:
            print("DEBUG: Initializing ImGui...")
            imgui.create_context()
            self.imgui_renderer = create_renderer(self)
            print("DEBUG: ImGui initialized.")
        except Exception as e_imgui:
            print(f"FATAL: Error initializing ImGui: {e_imgui}")
            pyglet.app.exit()
            return
        # --- End ImGui Setup ---

        # --- Debug Texture Setup ---
        self.create_debug_textures()


    def create_debug_textures(self):
        """Creates or re-creates textures for debug views."""
        # Delete existing textures if they exist
        if hasattr(self, 'camera_texture') and self.camera_texture:
            try: gl.glDeleteTextures(1, self.camera_texture)
            except: pass
        if hasattr(self, 'depth_texture') and self.depth_texture:
            try: gl.glDeleteTextures(1, self.depth_texture)
            except: pass
        if hasattr(self, 'edge_texture') and self.edge_texture:
            try: gl.glDeleteTextures(1, self.edge_texture)
            except: pass
        if hasattr(self, 'smoothing_texture') and self.smoothing_texture:
            try: gl.glDeleteTextures(1, self.smoothing_texture)
            except: pass

        # Create textures (using window size initially, will resize in update)
        width = max(1, self.width)
        height = max(1, self.height)

        self.camera_texture = gl.GLuint()
        gl.glGenTextures(1, self.camera_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self.depth_texture = gl.GLuint()
        gl.glGenTextures(1, self.depth_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self.edge_texture = gl.GLuint()
        gl.glGenTextures(1, self.edge_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.edge_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self.smoothing_texture = gl.GLuint()
        gl.glGenTextures(1, self.smoothing_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.smoothing_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)


        gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # Unbind
        self.debug_textures_initialized = True
        print("DEBUG: Debug textures created/recreated.")


    def start_inference_thread(self, model_name, inference_interval): # Removed disable_point_smoothing_initial
        # Ensure refs are initialized before starting thread
        if not hasattr(self, 'scale_factor_ref') or self.scale_factor_ref is None:
             self.scale_factor_ref = [self.input_scale_factor]
        if not hasattr(self, 'edge_params_ref') or not self.edge_params_ref:
             self.update_edge_params_ref() # Initialize edge params ref

        if self.inference_thread is None or not self.inference_thread.is_alive():
            self._exit_event.clear()
            self.inference_thread = threading.Thread(
                target=inference_thread_func,
                args=(self._data_queue, self._exit_event, model_name, inference_interval,
                      # enable_point_smoothing_arg removed
                      self.scale_factor_ref,
                      self.edge_params_ref), # Pass edge params dict
                daemon=True
            )
            self.inference_thread.start()
            print("DEBUG: Inference thread started via start_inference_thread.")

    def update_edge_params_ref(self):
        """Updates the dictionary passed to the inference thread."""
        self.edge_params_ref["enable_point_smoothing"] = self.enable_point_smoothing
        self.edge_params_ref["min_alpha_points"] = self.min_alpha_points
        self.edge_params_ref["max_alpha_points"] = self.max_alpha_points
        self.edge_params_ref["enable_edge_aware"] = self.enable_edge_aware_smoothing
        self.edge_params_ref["depth_threshold1"] = int(self.depth_edge_threshold1)
        self.edge_params_ref["depth_threshold2"] = int(self.depth_edge_threshold2)
        self.edge_params_ref["rgb_threshold1"] = int(self.rgb_edge_threshold1)
        self.edge_params_ref["rgb_threshold2"] = int(self.rgb_edge_threshold2)
        self.edge_params_ref["influence"] = self.edge_smoothing_influence
        self.edge_params_ref["gradient_influence_scale"] = self.gradient_influence_scale


    def update(self, dt):
        """Scheduled function to process data from the inference thread."""
        try:
            while True:
                latest_data = self._data_queue.get_nowait()

                if isinstance(latest_data, tuple) and isinstance(latest_data[0], str):
                    # Handle status/error/warning messages
                    if latest_data[0] == "status": self.status_message = latest_data[1]
                    elif latest_data[0] == "error": self.status_message = f"ERROR: {latest_data[1]}"; print(f"ERROR: {latest_data[1]}")
                    elif latest_data[0] == "warning": self.status_message = f"WARN: {latest_data[1]}"; print(f"WARN: {latest_data[1]}")
                    continue
                else:
                    # --- Process actual vertex and image data ---
                    try:
                        # Unpack new data format including edge and smoothing maps
                        vertices_data, colors_data, num_vertices_actual, \
                        rgb_frame_np, depth_map_tensor, edge_map_viz_np, \
                        smoothing_map_viz_np, t_capture = latest_data

                        self.last_capture_timestamp = t_capture

                        # --- Update Debug Views ---
                        self.latest_rgb_frame = rgb_frame_np
                        self.latest_edge_map = edge_map_viz_np
                        self.latest_smoothing_map = smoothing_map_viz_np # Store smoothing map viz

                        if depth_map_tensor is not None:
                            try:
                                depth_np = depth_map_tensor.cpu().numpy()
                                depth_normalized = np.clip(depth_np / 10.0, 0.0, 1.0)
                                depth_vis = (depth_normalized * 255).astype(np.uint8)
                                depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                                self.latest_depth_map_viz = np.ascontiguousarray(depth_vis_colored)
                            except Exception as e_depth_viz:
                                print(f"Error processing depth map for viz: {e_depth_viz}")
                                self.latest_depth_map_viz = None
                        else:
                            self.latest_depth_map_viz = None

                        # Update textures if they exist and data is available
                        if self.debug_textures_initialized:
                            if self.latest_rgb_frame is not None and self.camera_texture is not None:
                                try:
                                    h, w, _ = self.latest_rgb_frame.shape
                                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture)
                                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.latest_rgb_frame.ctypes.data)
                                except Exception as e_tex_cam:
                                    print(f"Error updating camera texture: {e_tex_cam}")


                            if self.latest_depth_map_viz is not None and self.depth_texture is not None:
                                try:
                                    h, w, _ = self.latest_depth_map_viz.shape
                                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
                                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self.latest_depth_map_viz.ctypes.data) # OpenCV uses BGR
                                except Exception as e_tex_depth:
                                     print(f"Error updating depth texture: {e_tex_depth}")

                            if self.latest_edge_map is not None and self.edge_texture is not None:
                                try:
                                    h, w, _ = self.latest_edge_map.shape
                                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.edge_texture)
                                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.latest_edge_map.ctypes.data) # Edge map viz is RGB
                                except Exception as e_tex_edge:
                                     print(f"Error updating edge texture: {e_tex_edge}")

                            if self.latest_smoothing_map is not None and self.smoothing_texture is not None:
                                try:
                                    h, w, _ = self.latest_smoothing_map.shape
                                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.smoothing_texture)
                                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.latest_smoothing_map.ctypes.data) # Smoothing map viz is RGB
                                except Exception as e_tex_smooth:
                                     print(f"Error updating smoothing texture: {e_tex_smooth}")


                            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # Unbind


                        # --- Calculate Point Cloud FPS ---
                        current_time = time.time()
                        time_delta = current_time - self.last_update_time
                        if time_delta > 1e-6:
                            self.point_cloud_fps = 1.0 / time_delta
                        self.last_update_time = current_time
                        # -----------------------------

                        self.current_point_count = num_vertices_actual

                        if vertices_data is not None and colors_data is not None and num_vertices_actual > 0:
                            if self.vertex_list:
                                try: self.vertex_list.delete()
                                except Exception: pass # Ignore error if already deleted
                                self.vertex_list = None

                            try:
                                self.vertex_list = self.shader_program.vertex_list(
                                    num_vertices_actual,
                                    gl.GL_POINTS,
                                    vertices=('f', vertices_data),
                                    colors=('f', colors_data)
                                )
                                self.frame_count_display += 1
                            except Exception as e_create:
                                 print(f"Error creating vertex list: {e_create}")
                                 traceback.print_exc()
                        else:
                            self.frame_count_display += 1

                    except Exception as e_unpack:
                        print(f"Error unpacking or processing data: {e_unpack}")
                        traceback.print_exc()

        except queue.Empty:
            pass # No new data is fine


    def update_camera(self, dt):
        """Scheduled function to handle camera movement based on key states."""
        io = imgui.get_io()
        if io.want_capture_keyboard: return

        speed = self.fast_move_speed if self.keys[key.LSHIFT] or self.keys[key.RSHIFT] else self.move_speed
        move_amount = speed * dt
        rot_y = -math.radians(self.camera_rotation_y)
        rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(math.sin(rot_y) * math.cos(rot_x), -math.sin(rot_x), -math.cos(rot_y) * math.cos(rot_x)).normalize()
        right = self.world_up_vector.cross(forward).normalize()
        up = self.world_up_vector

        if self.keys[key.W]: self.camera_position += forward * move_amount
        if self.keys[key.S]: self.camera_position -= forward * move_amount
        if self.keys[key.A]: self.camera_position += right * move_amount # Swapped A/D
        if self.keys[key.D]: self.camera_position -= right * move_amount # Swapped A/D
        if self.keys[key.E]: self.camera_position += up * move_amount
        if self.keys[key.Q]: self.camera_position -= up * move_amount

    def get_camera_matrices(self):
        rot_y = -math.radians(self.camera_rotation_y)
        rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(math.sin(rot_y) * math.cos(rot_x), -math.sin(rot_x), -math.cos(rot_y) * math.cos(rot_x)).normalize()
        target = self.camera_position + forward
        view = Mat4.look_at(self.camera_position, target, self.world_up_vector)
        projection = Mat4.perspective_projection(self._aspect_ratio, z_near=0.1, z_far=1000.0, fov=60.0) # Use renamed variable
        return projection, view

    def reset_settings(self):
        """Resets settings to default values."""
        print("DEBUG: Resetting settings to default.")
        for key, value in DEFAULT_SETTINGS.items():
            setattr(self, key, value)
        # Always create/update the scale factor reference list
        self.scale_factor_ref = [self.input_scale_factor]
        # Update edge params ref dictionary
        self.update_edge_params_ref()


    def save_settings(self, filename="viewer_settings.json"):
        """Saves current settings to a JSON file."""
        settings_to_save = {key: getattr(self, key) for key in DEFAULT_SETTINGS}
        try:
            with open(filename, 'w') as f:
                json.dump(settings_to_save, f, indent=4)
            print(f"DEBUG: Settings saved to {filename}")
            self.status_message = f"Settings saved to {filename}"
        except Exception as e:
            print(f"Error saving settings: {e}")
            self.status_message = "Error saving settings."

    def load_settings(self, filename="viewer_settings.json"):
        """Loads settings from a JSON file or uses defaults."""
        # Initialize attributes from defaults first
        for key, value in DEFAULT_SETTINGS.items():
            setattr(self, key, value)

        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_settings = json.load(f)
                # Update attributes from loaded file, keeping defaults if key missing
                for key in DEFAULT_SETTINGS:
                    setattr(self, key, loaded_settings.get(key, getattr(self, key)))
                print(f"DEBUG: Settings loaded from {filename}")
            else:
                print(f"DEBUG: Settings file {filename} not found, using defaults.")
                # Defaults are already set
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            # Ensure defaults are set if loading fails
            for key, value in DEFAULT_SETTINGS.items():
                setattr(self, key, value)

        # Ensure scale factor and edge params reference dicts are updated/created *after* loading/defaults
        self.scale_factor_ref = [self.input_scale_factor]
        self.update_edge_params_ref() # Initialize edge params ref


    def on_draw(self):
        # Clear the main window buffer
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # --- Draw 3D Splats ---
        if self.vertex_list:
            projection, current_view = self.get_camera_matrices()
            self.shader_program.use()
            self.shader_program['projection'] = projection
            self.shader_program['view'] = current_view
            self.shader_program['viewportSize'] = (float(self.width), float(self.height))
            # Pass control uniforms to shader
            self.shader_program['inputScaleFactor'] = self.input_scale_factor
            self.shader_program['pointSizeBoost'] = self.point_size_boost
            self.shader_program['renderMode'] = self.render_mode
            self.shader_program['falloffFactor'] = self.falloff_factor
            self.shader_program['saturation'] = self.saturation
            self.shader_program['brightness'] = self.brightness
            self.shader_program['contrast'] = self.contrast
            self.shader_program['sharpness'] = self.sharpness

            # Set GL state based on render mode
            if self.render_mode == 0: # Square (Opaque)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)
            elif self.render_mode == 1: # Circle (Opaque)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)
            elif self.render_mode == 2: # Gaussian (Premultiplied Alpha Blend)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_FALSE) # Disable depth write for transparency
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA) # Premultiplied alpha blend func

            # Draw the splats
            try:
                self.vertex_list.draw(gl.GL_POINTS)
            except Exception as e:
                print(f"Error during vertex_list.draw: {e}")
            finally:
                # Restore default-ish state after drawing splats
                gl.glDepthMask(gl.GL_TRUE)
                # Keep blend enabled for ImGui/UI, but set to standard alpha
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                self.shader_program.stop()
        # --- End Draw 3D Splats ---

        # --- ImGui Frame ---
        imgui.new_frame()

        # --- Debug Views ---
        if self.show_camera_feed and self.camera_texture and self.latest_rgb_frame is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.begin("Camera Feed")
            imgui.image(self.camera_texture, self.latest_rgb_frame.shape[1], self.latest_rgb_frame.shape[0])
            imgui.end()

        if self.show_depth_map and self.depth_texture and self.latest_depth_map_viz is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.begin("Depth Map")
            imgui.image(self.depth_texture, self.latest_depth_map_viz.shape[1], self.latest_depth_map_viz.shape[0])
            imgui.end()

        if self.show_edge_map and self.edge_texture and self.latest_edge_map is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.begin("Edge Map")
            imgui.image(self.edge_texture, self.latest_edge_map.shape[1], self.latest_edge_map.shape[0])
            imgui.end()

        if self.show_smoothing_map and self.smoothing_texture and self.latest_smoothing_map is not None:
             imgui.set_next_window_size(320, 240, imgui.ONCE)
             imgui.begin("Smoothing Alpha Map")
             imgui.image(self.smoothing_texture, self.latest_smoothing_map.shape[1], self.latest_smoothing_map.shape[0])
             imgui.end()
        # --- End Debug Views ---


        # --- Controls Panel ---
        imgui.set_next_window_position(self.width - 310, 10, imgui.ONCE) # Position bottom-rightish once
        imgui.set_next_window_size(300, 650, imgui.ONCE) # Increased height
        imgui.begin("Controls", True) # True = closable window

        # --- Presets ---
        if imgui.button("Load Settings"): self.load_settings()
        imgui.same_line()
        if imgui.button("Save Settings"): self.save_settings()
        imgui.same_line()
        if imgui.button("Reset All"): self.reset_settings()
        imgui.separator()

        # --- Rendering Section ---
        if imgui.collapsing_header("Rendering", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text("Render Mode:")
            if imgui.radio_button("Square##RenderMode", self.render_mode == 0): self.render_mode = 0
            imgui.same_line()
            if imgui.radio_button("Circle##RenderMode", self.render_mode == 1): self.render_mode = 1
            imgui.same_line()
            if imgui.radio_button("Gaussian##RenderMode", self.render_mode == 2): self.render_mode = 2

            # Gaussian Params (only relevant if Gaussian mode)
            if self.render_mode == 2:
                imgui.indent()
                changed_falloff, self.falloff_factor = imgui.slider_float("Falloff", self.falloff_factor, 0.1, 20.0)
                if imgui.button("Reset##Falloff"): self.falloff_factor = DEFAULT_SETTINGS["falloff_factor"]
                imgui.unindent()

        # --- View Section ---
        if imgui.collapsing_header("View", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            changed_boost, self.point_size_boost = imgui.slider_float("Point Size Boost", self.point_size_boost, 0.1, 10.0)
            if imgui.button("Reset##PointSize"): self.point_size_boost = DEFAULT_SETTINGS["point_size_boost"]

            changed_scale, self.input_scale_factor = imgui.slider_float("Input Scale", self.input_scale_factor, 0.1, 1.0)
            if changed_scale: self.scale_factor_ref[0] = self.input_scale_factor # Update shared ref for thread
            if imgui.button("Reset##InputScale"):
                self.input_scale_factor = DEFAULT_SETTINGS["input_scale_factor"]
                self.scale_factor_ref[0] = self.input_scale_factor

        # --- Image Processing Section ---
        if imgui.collapsing_header("Image Processing", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            changed_sat, self.saturation = imgui.slider_float("Saturation", self.saturation, 0.0, 3.0)
            if imgui.button("Reset##Saturation"): self.saturation = DEFAULT_SETTINGS["saturation"]

            changed_brt, self.brightness = imgui.slider_float("Brightness", self.brightness, 0.0, 2.0)
            if imgui.button("Reset##Brightness"): self.brightness = DEFAULT_SETTINGS["brightness"]

            changed_con, self.contrast = imgui.slider_float("Contrast", self.contrast, 0.1, 3.0)
            if imgui.button("Reset##Contrast"): self.contrast = DEFAULT_SETTINGS["contrast"]

            changed_shp, self.sharpness = imgui.slider_float("Sharpness", self.sharpness, 0.1, 3.0)
            if imgui.button("Reset##Sharpness"): self.sharpness = DEFAULT_SETTINGS["sharpness"]

        # --- Smoothing Section ---
        if imgui.collapsing_header("Smoothing", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
             changed_smooth, self.enable_point_smoothing = imgui.checkbox("Enable Point Smoothing", self.enable_point_smoothing)
             if changed_smooth: self.update_edge_params_ref() # Update ref dict

             imgui.indent()
             changed_min_alpha, self.min_alpha_points = imgui.slider_float("Min Alpha", self.min_alpha_points, 0.0, 1.0)
             if changed_min_alpha: self.update_edge_params_ref()
             changed_max_alpha, self.max_alpha_points = imgui.slider_float("Max Alpha", self.max_alpha_points, 0.0, 1.0)
             if changed_max_alpha: self.update_edge_params_ref()
             if imgui.button("Reset##SmoothAlpha"):
                 self.min_alpha_points = DEFAULT_SETTINGS["min_alpha_points"]
                 self.max_alpha_points = DEFAULT_SETTINGS["max_alpha_points"]
                 self.update_edge_params_ref()
             imgui.unindent()

             imgui.separator()
             changed_edge_smooth, self.enable_edge_aware_smoothing = imgui.checkbox("Enable Edge-Aware Smoothing", self.enable_edge_aware_smoothing)
             if changed_edge_smooth: self.update_edge_params_ref()

             if self.enable_edge_aware_smoothing:
                 imgui.indent()
                 changed_d_thresh1, self.depth_edge_threshold1 = imgui.slider_float("Depth Thresh 1", self.depth_edge_threshold1, 1.0, 255.0)
                 if changed_d_thresh1: self.update_edge_params_ref()
                 changed_d_thresh2, self.depth_edge_threshold2 = imgui.slider_float("Depth Thresh 2", self.depth_edge_threshold2, 1.0, 255.0)
                 if changed_d_thresh2: self.update_edge_params_ref()

                 changed_rgb_thresh1, self.rgb_edge_threshold1 = imgui.slider_float("RGB Thresh 1", self.rgb_edge_threshold1, 1.0, 255.0)
                 if changed_rgb_thresh1: self.update_edge_params_ref()
                 changed_rgb_thresh2, self.rgb_edge_threshold2 = imgui.slider_float("RGB Thresh 2", self.rgb_edge_threshold2, 1.0, 255.0)
                 if changed_rgb_thresh2: self.update_edge_params_ref()

                 changed_edge_inf, self.edge_smoothing_influence = imgui.slider_float("Edge Influence", self.edge_smoothing_influence, 0.0, 1.0)
                 if changed_edge_inf: self.update_edge_params_ref()

                 changed_grad_inf, self.gradient_influence_scale = imgui.slider_float("Gradient Scale", self.gradient_influence_scale, 0.0, 5.0)
                 if changed_grad_inf: self.update_edge_params_ref()


                 if imgui.button("Reset##EdgeParams"):
                     self.depth_edge_threshold1 = DEFAULT_SETTINGS["depth_edge_threshold1"]
                     self.depth_edge_threshold2 = DEFAULT_SETTINGS["depth_edge_threshold2"]
                     self.rgb_edge_threshold1 = DEFAULT_SETTINGS["rgb_edge_threshold1"]
                     self.rgb_edge_threshold2 = DEFAULT_SETTINGS["rgb_edge_threshold2"]
                     self.edge_smoothing_influence = DEFAULT_SETTINGS["edge_smoothing_influence"]
                     self.gradient_influence_scale = DEFAULT_SETTINGS["gradient_influence_scale"]
                     self.update_edge_params_ref()
                 imgui.unindent()


        # --- Debug Views Section ---
        if imgui.collapsing_header("Debug Views")[0]:
            _, self.show_camera_feed = imgui.checkbox("Show Camera Feed", self.show_camera_feed)
            _, self.show_depth_map = imgui.checkbox("Show Depth Map", self.show_depth_map)
            _, self.show_edge_map = imgui.checkbox("Show Edge Map", self.show_edge_map)
            _, self.show_smoothing_map = imgui.checkbox("Show Smoothing Map", self.show_smoothing_map)


        # --- Info Section ---
        if imgui.collapsing_header("Info")[0]:
            imgui.text(f"Points: {self.current_point_count}")
            imgui.text(f"FPS: {self.point_cloud_fps:.1f}")
            imgui.text_wrapped(f"Status: {self.status_message}")

        imgui.end()
        # --- End Controls Panel ---

        # Render ImGui
        # Ensure correct GL state for ImGui rendering (standard alpha blend)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST) # ImGui draws in 2D

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())
        # --- End ImGui Frame ---

        # Restore GL state needed for next 3D frame
        gl.glEnable(gl.GL_DEPTH_TEST)


    # --- Input Handlers ---
    def on_resize(self, width, height):
        gl.glViewport(0, 0, max(1, width), max(1, height))
        self._aspect_ratio = float(width) / height if height > 0 else 1.0 # Update renamed variable
        # Recreate debug textures on resize
        self.create_debug_textures()

    def on_mouse_press(self, x, y, button, modifiers):
        io = imgui.get_io()
        if io.want_capture_mouse: return
        if button == pyglet.window.mouse.LEFT: self.set_exclusive_mouse(True); self.mouse_down = True

    def on_mouse_release(self, x, y, button, modifiers):
        io = imgui.get_io()
        if io.want_capture_mouse: return
        if button == pyglet.window.mouse.LEFT: self.set_exclusive_mouse(False); self.mouse_down = False

    def on_mouse_motion(self, x, y, dx, dy):
        io = imgui.get_io()
        if io.want_capture_mouse: return
        if self.mouse_down:
            sensitivity = 0.1; self.camera_rotation_y -= dx * sensitivity; self.camera_rotation_y %= 360
            self.camera_rotation_x += dy * sensitivity; self.camera_rotation_x = max(min(self.camera_rotation_x, 89.9), -89.9)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        io = imgui.get_io()
        if io.want_capture_mouse: return
        zoom_speed = 0.5; rot_y = -math.radians(self.camera_rotation_y); rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(math.sin(rot_y) * math.cos(rot_x), -math.sin(rot_x), -math.cos(rot_y) * math.cos(rot_x)).normalize()
        self.camera_position += forward * scroll_y * zoom_speed

    def on_key_press(self, symbol, modifiers):
        io = imgui.get_io()
        if io.want_capture_keyboard: return
        if symbol == pyglet.window.key.ESCAPE:
            self.set_exclusive_mouse(False)
            self.close()
        # Keybindings for scale/boost removed

    def on_close(self):
        print("Window closing, stopping inference thread...")
        self._exit_event.set()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.vertex_list:
            try: self.vertex_list.delete()
            except Exception as e_del: print(f"Error deleting vertex list on close: {e_del}")
        if self.shader_program: self.shader_program.delete()
        # Debug texture cleanup
        if hasattr(self, 'camera_texture') and self.camera_texture:
             try: gl.glDeleteTextures(1, self.camera_texture)
             except: pass
        if hasattr(self, 'depth_texture') and self.depth_texture:
             try: gl.glDeleteTextures(1, self.depth_texture)
             except: pass
        if hasattr(self, 'edge_texture') and self.edge_texture:
             try: gl.glDeleteTextures(1, self.edge_texture)
             except: pass
        if hasattr(self, 'smoothing_texture') and self.smoothing_texture:
             try: gl.glDeleteTextures(1, self.smoothing_texture)
             except: pass
        # --- ImGui Cleanup ---
        if hasattr(self, 'imgui_renderer') and self.imgui_renderer:
            self.imgui_renderer.shutdown()
        # --- End ImGui Cleanup ---
        super().on_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live SLAM viewer using UniK3D.")
    parser.add_argument("--model", type=str, default="unik3d-vitl", help="Name of the UniK3D model to use (e.g., unik3d-vits, unik3d-vitb, unik3d-vitl)")
    parser.add_argument("--interval", type=int, default=1, help="Run inference every N frames (default: 1).")
    # Argument --disable-point-smoothing is now effectively ignored, control is via settings/ImGui
    # parser.add_argument("--disable-point-smoothing", action="store_true", help="Disable temporal smoothing on 3D points (default: enabled).")
    args = parser.parse_args()

    # Pass smoothing flags to window constructor (using default GL config)
    window = LiveViewerWindow(model_name=args.model, inference_interval=args.interval,
                              disable_point_smoothing=False, # Pass False, initial state loaded from settings
                              width=1024, height=768, caption='Live UniK3D SLAM Viewer', resizable=True)
    try:
        pyglet.app.run()
    except Exception as e:
        print("\n--- Uncaught Exception ---")
        traceback.print_exc()
        print("------------------------")
    finally:
        # Ensure ImGui context is destroyed if app exits unexpectedly
        if imgui.get_current_context():
            print("DEBUG: Destroying ImGui context in finally block.")
            if hasattr(window, 'imgui_renderer') and window.imgui_renderer:
                 try:
                     window.imgui_renderer.shutdown()
                     print("DEBUG: ImGui renderer shutdown.")
                 except Exception as e_shutdown:
                     print(f"Error shutting down ImGui renderer: {e_shutdown}")
            try:
                imgui.destroy_context()
                print("DEBUG: ImGui context destroyed.")
            except Exception as e_destroy:
                print(f"Error destroying ImGui context: {e_destroy}")
        print("Application finished.")