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

# Assuming unik3d is installed and importable
from unik3d.models import UniK3D

from pyglet.math import Mat4, Vec3
from pyglet.window import key # Import key for KeyStateHandler
import traceback # Import traceback
import pyglet.text # Import for Label

# Simple shaders (Using 'vertices' instead of 'position')
# Expects vec3 floats for both vertices and colors
vertex_source = """#version 150 core
    in vec3 vertices;
    in vec3 colors; // Expects normalized floats (0.0-1.0)

    out vec3 vertex_colors;

    uniform mat4 projection;
    uniform mat4 view;
    uniform float inputScaleFactor; // Added uniform for dynamic input scale
    uniform float pointSizeBoost;   // Added uniform for dynamic point size boost
    // Removed viewportHeight uniform

    void main() {
        gl_Position = projection * view * vec4(vertices, 1.0);
        vertex_colors = colors; // Pass color data through

        // --- Point Size based ONLY on distance from Origin ---
        // 1. Calculate distance from origin (capturing camera)
        float originDist = length(vertices);

        // 2. Calculate pixel size directly proportional to origin distance.
        //    Adjust scalingFactor (e.g., 3.0) and min/max pixels (e.g., 1.0, 30.0) as needed.
        float baseScalingFactor = 1.0; // Reduced overall point size multiplier
        // Adjust point size based on distance AND inversely by input scale factor
        // Ensure inputScaleFactor is not zero to avoid division by zero
        float effectiveScaleFactor = max(0.1, inputScaleFactor); // Prevent division by zero or negative scale
        float pointSizePixels = (baseScalingFactor * originDist) / effectiveScaleFactor;

        // 3. Set final point size (clamped)
        // Clamp the calculated size first
        float clampedSize = max(1.0, min(30.0, pointSizePixels));
        // Increase the clamped size by the dynamic boost factor
        gl_PointSize = clampedSize * pointSizeBoost;
        // --- End Point Size Calculation ---
    }
"""

fragment_source = """#version 150 core
    in vec3 geom_colors; // Input from Geometry Shader
    out vec4 final_color;

    // Function to convert RGB to HSV
    // Source: Adapted from various online sources (e.g., StackOverflow, blogs)
    vec3 rgb2hsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10; // Epsilon to avoid division by zero
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    // Function to convert HSV to RGB
    // Source: Adapted from various online sources
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    void main() {
        // Convert incoming RGB color to HSV
        vec3 hsv = rgb2hsv(geom_colors);

        // Increase saturation (multiply by a factor > 1, e.g., 1.5)
        // Clamp saturation to the valid range [0.0, 1.0]
        float saturationFactor = 1.5;
        hsv.y = clamp(hsv.y * saturationFactor, 0.0, 1.0);

        // Convert modified HSV back to RGB
        vec3 saturatedRgb = hsv2rgb(hsv);

        // --- Approximate Sharpening via Contrast Boost ---
        // Convert saturated RGB back to HSV
        vec3 hsv_sharp = rgb2hsv(saturatedRgb);
        // Define contrast factor (e.g., 1.1 means 10% boost)
        float contrastFactor = 1.1;
        // Push Value away from 0.5
        hsv_sharp.z = clamp(0.5 + (hsv_sharp.z - 0.5) * contrastFactor, 0.0, 1.0);
        // Convert final HSV back to RGB
        vec3 finalRgb = hsv2rgb(hsv_sharp);
        // --- End Sharpening ---

        // Output the final sharpened color
        final_color = vec4(finalRgb, 1.0);
    }
"""

# --- Inference Thread Function ---
# (Keep inference thread function as is)
def inference_thread_func(data_queue, exit_event, model_name, inference_interval,
                          enable_point_smoothing_arg, scale_factor_ref): # Added scale_factor_ref
    """Loads model, captures camera, runs inference (with dynamic scaling), puts results in queue."""
    print("Inference thread started.")
    data_queue.put(("status", "Inference thread started...")) # Status update
    try:
        # --- Load Model ---
        print(f"Loading UniK3D model: {model_name}...")
        data_queue.put(("status", f"Loading model: {model_name}...")) # Status update
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = UniK3D.from_pretrained(f"lpiccinelli/{model_name}")
        model = model.to(device)
        model.eval() # Set to evaluation mode
        print("Model loaded.")
        data_queue.put(("status", "Model loaded.")) # Status update

        # --- Initialize Camera ---
        print("Initializing camera...")
        data_queue.put(("status", "Initializing camera...")) # Status update
        # Reverted CAP_DSHOW change
        cap = cv2.VideoCapture(0) # Use default camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            data_queue.put(("error", "Could not open camera."))
            print("DEBUG inference: cap.isOpened() failed, returning from thread.") # Debug print
            return
        print("Camera initialized.")
        data_queue.put(("status", "Camera initialized.")) # Status update

        # --- Start of main loop logic (Corrected Indentation) ---
        frame_count = 0
        last_inference_time = time.time()

        # --- Temporal Smoothing & Motion Init ---
        smoothed_points_xyz = None # For smoothing final points
        # Flags to enable/disable smoothing (can be controlled by args)
        enable_point_smoothing = enable_point_smoothing_arg # Use arg passed to thread
        # Adaptive alpha for points (only used if enable_point_smoothing is True)
        min_alpha_points = 0.0  # Freeze points when still
        max_alpha_points = 1.0  # No smoothing when moving
        depth_motion_threshold = 0.1 # Restore threshold, will use non-linear mapping
        prev_depth_map = None # Store previous depth map for difference calculation
        # Global scale smoothing parameters
        smoothed_mean_depth = None
        min_alpha_scale = 0.0 # Freeze scale when still
        max_alpha_scale = 1.0 # Instant scale update when moving
        prev_scale_factor = None # Track previous scale factor for smoothing reset
        # -----------------------------

        # Removed debug print
        while not exit_event.is_set():
            # Removed debug prints
            t_capture = time.time() # Timestamp frame capture
            ret, frame = cap.read()
            # Removed debug print
            if not ret:
                print("Error: Failed to capture frame.")
                time.sleep(0.1)
                continue

            # --- Dynamically Scale Frame ---
            current_scale = scale_factor_ref[0] # Read current scale from shared reference
            if current_scale > 0.1: # Only resize if scale is reasonably positive
                new_width = int(frame.shape[1] * current_scale)
                new_height = int(frame.shape[0] * current_scale)
                # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging for better quality
                interpolation = cv2.INTER_AREA if current_scale < 1.0 else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
            # --- End Scale Frame ---
            # (Motion calculation moved to after depth prediction)
            # Removed debug print

            frame_count += 1
            # Removed debug print

            # --- Run Inference periodically ---
            # Removed debug print
            if frame_count % inference_interval == 0:
                # --- Check for Scale Change and Reset Smoothing ---
                current_scale = scale_factor_ref[0] # Read current scale factor
                if prev_scale_factor is not None and abs(current_scale - prev_scale_factor) > 1e-3: # Check if scale changed significantly
                    print(f"Scale factor changed ({prev_scale_factor:.2f} -> {current_scale:.2f}). Resetting smoothing state.")
                    smoothed_points_xyz = None
                    prev_depth_map = None
                    # Also reset smoothed mean depth to avoid sudden jumps
                    smoothed_mean_depth = None
                prev_scale_factor = current_scale # Update previous scale factor for next check
                # --- End Smoothing Reset Logic ---

                # Removed debug print
                current_time = time.time()
                # Keep the user-facing print below
                print(f"Running inference for frame {frame_count} (Time since last: {current_time - last_inference_time:.2f}s)")
                last_inference_time = current_time

                # Removed debug print
                try:
                    # --- Preprocess Frame ---
                    data_queue.put(("status", f"Preprocessing frame {frame_count}...")) # Status update
                    # Convert BGR (OpenCV) to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_h = rgb_frame.shape[0] # Get frame height
                    # Convert to torch tensor (HWC -> CHW) and move to device
                    # Assuming UniK3D handles normalization internally based on README example
                    frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().to(device)
                    # Add batch dimension
                    frame_tensor = frame_tensor.unsqueeze(0)

                    # --- Inference ---
                    data_queue.put(("status", f"Running inference on frame {frame_count}...")) # Status update
                    with torch.no_grad():
                        predictions = model.infer(frame_tensor) # No camera intrinsics for now
                        # Removed debug print for prediction keys
                        # Removed key logging print(f"DEBUG: Prediction keys: {predictions.keys()}")

                    # --- Get Points (Depth calculation removed) ---
                    if 'rays' in predictions and 'depth' in predictions:
                        # We still need depth and rays to calculate initial points_xyz
                        # but we won't use the depth difference for motion anymore.
                        current_depth_map = predictions['depth'].squeeze() # (H, W)

                        # --- Calculate Per-Pixel Depth Motion Map ---
                        per_pixel_depth_motion_map = None # Initialize for this frame
                        if prev_depth_map is not None:
                            if current_depth_map.shape == prev_depth_map.shape:
                                depth_diff = torch.abs(current_depth_map - prev_depth_map)
                                # Scale difference to 0-1 motion value per pixel
                                per_pixel_depth_motion = torch.clamp(depth_diff / depth_motion_threshold, 0.0, 1.0) # Normalized 0-1

                                # --- Modulate motion by distance ---
                                # Normalize depth (0 near, 1 far, clamped at 10 units)
                                normalized_depth = torch.clamp(current_depth_map / 10.0, 0.0, 1.0)
                                # Reduce motion sensitivity for farther points
                                distance_modulated_motion = per_pixel_depth_motion * (1.0 - normalized_depth)
                                per_pixel_depth_motion_map = distance_modulated_motion # Use modulated motion map
                                # --- End distance modulation ---
                            # else: map remains None if shape changes

                        # --- End Per-Pixel Depth Motion Map --- (per_pixel_depth_motion_map now holds distance_modulated_motion)

                        # Store prev_depth_map *after* using current_depth_map for modulation calc
                        prev_depth_map = current_depth_map.clone()

                        # --- Apply Global Scale Smoothing (Adaptive) ---
                        # Calculate global motion based on the *original* per-pixel motion map
                        # before distance modulation for a more stable global metric.
                        global_motion_unmodulated = 0.0
                        if per_pixel_depth_motion is not None: # Use original motion before modulation
                            global_motion_unmodulated = per_pixel_depth_motion.mean().item()

                        scaled_depth_map = current_depth_map # Default to current if no smoothing needed
                        current_mean_depth = current_depth_map.mean().item()
                        # Calculate adaptive alpha for scale using non-linear mapping (exponent 1.5) on unmodulated global motion
                        motion_factor_scale = global_motion_unmodulated ** 2.0 # Increased exponent for sharper drop-off
                        adaptive_alpha_scale = min_alpha_scale + (max_alpha_scale - min_alpha_scale) * motion_factor_scale

                        if smoothed_mean_depth is None:
                            smoothed_mean_depth = current_mean_depth
                        else:
                            # Apply EMA to the mean depth
                            smoothed_mean_depth = adaptive_alpha_scale * current_mean_depth + (1.0 - adaptive_alpha_scale) * smoothed_mean_depth

                        # Calculate and apply scale correction factor
                        if current_mean_depth > 1e-6: # Avoid division by zero
                            scale_factor = smoothed_mean_depth / current_mean_depth
                            scaled_depth_map = current_depth_map * scale_factor
                        # --- End Global Scale Smoothing ---

                        # --- Calculate Points from Scaled Depth ---
                        rays = predictions['rays'].squeeze()
                        if rays.shape[0] == 3 and rays.ndim == 3: # Permute if (C, H, W)
                            rays = rays.permute(1, 2, 0) # (H, W, 3)
                        # Use the globally scaled depth map
                        depth_to_multiply = scaled_depth_map.unsqueeze(-1) # (H, W, 1)
                        points_xyz = rays * depth_to_multiply # (H, W, 3)
                        # --- End Point Calculation ---
                        data_queue.put(("status", f"Calculated points for frame {frame_count}..."))
                    elif "points" in predictions:
                        # Fallback if depth or rays are missing but points are present
                        data_queue.put(("status", f"Using direct points for frame {frame_count}..."))
                        points_xyz = predictions["points"] # Use original points
                    else:
                        # If neither depth/rays nor points are available
                        points_xyz = None
                        data_queue.put(("warning", f"No points/depth found for frame {frame_count}"))

                    # --- Original Point Extraction (Now part of fallback) ---
                    # points_xyz = predictions["points"] # Shape might be (B, N, 3) or similar

                    # --- Apply EMA Smoothing to Points ---
                    if points_xyz is not None:
                        if smoothed_points_xyz is None:
                            smoothed_points_xyz = points_xyz.clone()
                        else:
                            # Ensure shapes match before EMA
                            if smoothed_points_xyz.shape == points_xyz.shape:
                                if enable_point_smoothing and per_pixel_depth_motion_map is not None:
                                    # Check if points_xyz has compatible shape (H, W, 3)
                                    if points_xyz.ndim == 3 and points_xyz.shape[:2] == per_pixel_depth_motion_map.shape:
                                        # --- Calculate Per-Point Adaptive Alpha Map with distance-dependent min_alpha and non-linear mapping ---
                                        # Calculate distance-dependent minimum alpha (0.1 near, 0.0 far)
                                        # Removed distance-dependent minimum alpha calculation
                                        # Use original per_pixel_depth_motion for mapping, apply exponent 2.0
                                        motion_factor_points = per_pixel_depth_motion ** 2.0 # Increased exponent for sharper drop-off
                                        # Interpolate between global min (0.0) and global max (1.0)
                                        adaptive_alpha_points_map = min_alpha_points + (max_alpha_points - min_alpha_points) * motion_factor_points
                                        # Unsqueeze for broadcasting: (H, W) -> (H, W, 1)
                                        adaptive_alpha_points_map = adaptive_alpha_points_map.unsqueeze(-1)
                                        # --- Apply EMA with Per-Point Adaptive Alpha ---
                                        smoothed_points_xyz = adaptive_alpha_points_map * points_xyz + (1.0 - adaptive_alpha_points_map) * smoothed_points_xyz
                                    else:
                                        # Fallback if shapes don't match: use current points
                                        smoothed_points_xyz = points_xyz
                                else:
                                    # If smoothing disabled or no motion map, use current points
                                    smoothed_points_xyz = points_xyz
                            else:
                                print("Warning: Points tensor shape changed, resetting smoothing.")
                                smoothed_points_xyz = points_xyz.clone()
                        # Use the smoothed points for further processing
                        points_xyz_to_process = smoothed_points_xyz
                    else:
                        points_xyz_to_process = None # Handle case where points were None initially
                    # --- End Point Smoothing ---

                    # Removed debug print
                    if points_xyz_to_process is None or points_xyz_to_process.numel() == 0: # Check the processed points
                        # Removed debug print
                        # Put None to signal empty frame? Or just skip? Let's skip for now.
                        continue
                    # Removed debug print

                    # Ensure it's on CPU and NumPy for pyglet
                    # Removed shape debug prints
                    # Use the (potentially smoothed) points_xyz_to_process
                    points_xyz_np = points_xyz_to_process.squeeze().cpu().numpy() # Remove batch dim if present
                    # Removed debug print

                    # --- Reshape based on observed (C, H, W) format ---
                    num_vertices = 0
                    try: # Wrap the whole reshape logic in try/except
                        if points_xyz_np.ndim == 3 and points_xyz_np.shape[0] == 3: # Check if shape is (3, H, W)
                                # Transpose from (C, H, W) to (H, W, C)
                                points_xyz_np = np.transpose(points_xyz_np, (1, 2, 0))
                                # Reshape to (N, 3) where N = H * W
                                num_vertices = points_xyz_np.shape[0] * points_xyz_np.shape[1]
                                points_xyz_np = points_xyz_np.reshape(num_vertices, 3)
                        elif points_xyz_np.ndim == 2 and points_xyz_np.shape[1] == 3: # Handle expected (N, 3) case
                             num_vertices = points_xyz_np.shape[0]
                        elif points_xyz_np.ndim == 3 and points_xyz_np.shape[2] == 3: # Handle (H, W, 3) case
                             num_vertices = points_xyz_np.shape[0] * points_xyz_np.shape[1]
                             points_xyz_np = points_xyz_np.reshape(num_vertices, 3) # Reshape to (N, 3)
                        else:
                            print(f"Warning: Unexpected points_xyz_np shape after processing: {points_xyz_np.shape}")
                        # Removed debug print
                    except Exception as e_reshape:
                            print(f"Error reshaping points_xyz_np: {e_reshape}")
                            num_vertices = 0 # Prevent further processing if reshape fails
                    # Removed duplicated elif/else block below
                    # ----------------------------------------------------
                    # Removed num_vertices debug print

                    # --- Get Colors from Input Frame & Subsample ---
                    colors_np = None
                    if num_vertices > 0:
                        try:
                            # Reshape rgb_frame (H, W, 3) to (N, 3) matching points
                            # Assuming rgb_frame is already (H, W, 3) after cvtColor
                            original_h, original_w, _ = rgb_frame.shape
                            colors_np = rgb_frame.reshape(original_h * original_w, 3)

                            # Subsample points AND colors
                            subsample_rate = 1 # Render all points (was 4)
                            if subsample_rate > 1: # Only subsample if rate > 1
                                points_xyz_np = points_xyz_np[::subsample_rate]
                                colors_np = colors_np[::subsample_rate] # Apply same subsampling to colors

                            # Recalculate num_vertices after potential subsampling
                            num_vertices = points_xyz_np.shape[0]

                            # Normalize colors (uint8 0-255 -> float32 0.0-1.0)
                            if colors_np.dtype == np.uint8:
                                colors_np = colors_np.astype(np.float32) / 255.0
                            elif colors_np.dtype == np.float32: # Just in case it's already float
                                colors_np = np.clip(colors_np, 0.0, 1.0)
                            else: # Fallback if unexpected dtype
                                print(f"Warning: Unexpected color dtype {colors_np.dtype}, using white.")
                                colors_np = None

                        except Exception as e_color_subsample:
                            print(f"Error processing/subsampling colors: {e_color_subsample}")
                            colors_np = None # Fallback
                            # Ensure points are still subsampled even if color fails
                            if num_vertices > 0: # Check if num_vertices was calculated before error
                                points_xyz_np = points_xyz_np[::subsample_rate] # Apply point subsampling
                                num_vertices = points_xyz_np.shape[0] # Recalculate

                    # --- End Color/Subsample ---

                    # --- Queue data if vertices were processed successfully --- (Corrected Indentation)
                    if num_vertices > 0:
                        # Ensure points_xyz_np is now (N, 3) before flattening
                        vertices_flat = points_xyz_np.flatten()

                        # Create colors (either from frame or fallback white)
                        if colors_np is not None:
                            colors_flat = colors_np.flatten()
                        else: # Fallback to white if color processing failed
                            colors_np_white = np.ones((num_vertices, 3), dtype=np.float32)
                            colors_flat = colors_np_white.flatten()

                        # Put data into queue for the viewer
                        # Removed debug print
                        if not data_queue.full():
                                # No longer passing motion confidence
                                data_queue.put((vertices_flat, colors_flat, num_vertices, frame_h, t_capture))
                                # Removed debug print
                        else:
                            print(f"Warning: Viewer queue full, dropping frame {frame_count}.") # Modified warning
                            data_queue.put(("status", f"Viewer queue full, dropping frame {frame_count}")) # Status update
                    else: # This 'else' corresponds to 'if num_vertices > 0:'
                        # Removed debug print
                        pass # Keep the else block structure


                except Exception as e_infer: # Corrected indentation to match 'try' on line 96
                    print(f"Error during inference processing for frame {frame_count}: {e_infer}") # Modified print
                    traceback.print_exc() # Uncommented to show full error details

            # Small sleep to prevent busy-looping if camera is fast
            time.sleep(0.005)

    except Exception as e_thread:
        print(f"Error in inference thread: {e_thread}")
        traceback.print_exc()
        data_queue.put(("error", str(e_thread)))
        data_queue.put(("status", "Inference thread error!")) # Status update
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Inference thread finished.")
        data_queue.put(("status", "Inference thread finished.")) # Status update


# --- Viewer Class ---
class LiveViewerWindow(pyglet.window.Window):
    def __init__(self, model_name, inference_interval=1, # Changed default interval
                 disable_point_smoothing=False, # Removed depth smoothing arg
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertex_list = None # Initialize as None
        self.frame_count_display = 0 # For UI
        self.current_point_count = 0 # For UI
        self.current_scale_factor = 0.9 # Initial scale factor (Default 0.9x)
        self.point_size_boost = 2.5 # Initial point size boost factor (Default 2.5x)
        # self.last_depth_diff = 0.0 # Removed depth diff storage
        # --- FPS / Latency Tracking ---
        self.last_update_time = time.time()
        self.point_cloud_fps = 0.0
        self.last_capture_timestamp = None
        # -----------------------------

        # --- Status Display ---
        self.ui_batch = pyglet.graphics.Batch() # Create a batch for UI elements
        self.status_message = "Initializing..."
        # Status Label (moved up slightly)
        self.status_label = pyglet.text.Label(
            self.status_message,
            font_name='Arial',
            font_size=12,
            x=10, y=30, # Position slightly higher bottom-left
            anchor_x='left', anchor_y='bottom',
            color=(255, 255, 255, 200), # White with some transparency
            batch=self.ui_batch # Assign label to the batch
        )
        # Point Count Label (bottom-left)
        self.point_count_label = pyglet.text.Label(
            f"Points: {self.current_point_count}",
            font_name='Arial',
            font_size=12,
            x=10, y=10, # Position at bottom-left
            anchor_x='left', anchor_y='bottom',
            color=(255, 255, 255, 200),
            batch=self.ui_batch
        )
        # Scale Factor Label (top-right)
        self.scale_factor_label = pyglet.text.Label(
            f"Scale: {self.current_scale_factor:.2f}x",
            font_name='Arial',
            font_size=12,
            x=self.width - 10, y=self.height - 10, # Position near top-right
            anchor_x='right', anchor_y='top',
            color=(255, 255, 255, 200),
            batch=self.ui_batch
        )
        # Point Size Boost Label (below scale factor)
        self.point_boost_label = pyglet.text.Label(
            f"Pt Boost: {self.point_size_boost:.2f}x",
            font_name='Arial',
            font_size=12,
            x=self.width - 10, y=self.height - 30, # Position below scale label
            anchor_x='right', anchor_y='top',
            color=(255, 255, 255, 200),
            batch=self.ui_batch
        )
        # Removed FPSDisplay initialization
        # --------------------

        # --- Threading related ---
        self.data_queue = queue.Queue(maxsize=2) # Only buffer 1-2 results
        self.inference_thread = None
        self._exit_event = threading.Event()

        # --- Camera State ---
        self.camera_position = Vec3(0.0, 1.0, 5.0) # Move camera back and slightly up
        self.camera_rotation_x = -15.0 # Look slightly down
        self.camera_rotation_y = 180.0 # Start looking forward (Z-)
        self.world_up_vector = Vec3(0, 1, 0)
        self.move_speed = 2.0 # Slower speed for potentially smaller scenes
        self.fast_move_speed = 6.0
        # Track previous state for motion detection
        # Manually copy Vec3 components as .copy() doesn't exist
        self.prev_camera_position = Vec3(self.camera_position.x, self.camera_position.y, self.camera_position.z)
        self.prev_camera_rotation_x = self.camera_rotation_x
        self.prev_camera_rotation_y = self.camera_rotation_y
        self.camera_motion_confidence = 0.0 # Initialize motion confidence
        # --------------------

        # --- Input Handlers ---
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.mouse_down = False
        # --------------------

        # --- Geometry Shader Source ---
        geometry_source = """#version 150 core
            layout (points) in;
            layout (triangle_strip, max_vertices = 4) out;

            in vec3 vertex_colors[]; // Receive from vertex shader (array because input is point)
            out vec3 geom_colors;    // Pass to fragment shader

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
                EmitVertex();

                gl_Position = centerPosition + vec4( halfSizeX, -halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                EmitVertex();

                gl_Position = centerPosition + vec4(-halfSizeX,  halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                EmitVertex();

                gl_Position = centerPosition + vec4( halfSizeX,  halfSizeY, 0.0, 0.0);
                geom_colors = vertex_colors[0];
                EmitVertex();

                EndPrimitive();
            }
        """
        # --------------------------

        # Shader only (No Batch)
        try:
            print("DEBUG: Creating shaders...")
            vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
            frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
            geom_shader = pyglet.graphics.shader.Shader(geometry_source, 'geometry') # Create geometry shader
            # Include geometry shader in program
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
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0) # Update viewer frequently
        pyglet.clock.schedule_interval(self.update_camera, 1.0 / 60.0) # Update camera

        # Start the inference thread, passing point smoothing flag
        self.start_inference_thread(model_name, inference_interval,
                                    disable_point_smoothing)

        # Set up OpenGL state
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    def start_inference_thread(self, model_name, inference_interval,
                               disable_point_smoothing): # Removed depth smoothing arg
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self._exit_event.clear()
            # Pass point smoothing flag to the thread function
            # Note: We pass the *enable* flag, so negate the disable flag from args
            # Create a mutable reference (list) for the scale factor and store it
            self.scale_factor_ref = [self.current_scale_factor]
            thread_args = (self.data_queue, self._exit_event, model_name, inference_interval,
                           not disable_point_smoothing, self.scale_factor_ref) # Add scale factor ref to args
            self.inference_thread = threading.Thread(
                target=inference_thread_func,
                args=thread_args,
                daemon=True)
            self.inference_thread.start()

    def update(self, dt):
        """Checks queue for new point cloud data and recreates VertexList."""
        try:
            # Get the latest data, discard older ones if queue built up
            latest_data = None
            while not self.data_queue.empty():
                latest_data = self.data_queue.get_nowait()

            # Removed debug print for latest_data type

            if latest_data:
                # --- Check for message type ---
                if isinstance(latest_data[0], str):
                    # Handle status or error messages
                    if latest_data[0] == "error":
                        print(f"Received error from inference thread: {latest_data[1]}")
                        self.status_message = f"Error: {latest_data[1]}"
                        self.status_label.text = self.status_message
                        # Potentially clear vertex list on error?
                        # if self.vertex_list:
                        #     self.vertex_list.delete()
                        #     self.vertex_list = None
                    elif latest_data[0] == "status":
                        self.status_message = latest_data[1]
                        self.status_label.text = self.status_message
                    # After handling string message, do nothing else this update cycle
                else:
                    # --- Process actual vertex data ---
                    try:
                        # Unpack data (motion confidence removed)
                        vertices_data, colors_data, _, frame_h, t_capture = latest_data
                        self.current_image_height = frame_h # Store image height
                        # self.camera_motion_confidence = motion_confidence # Removed
                        self.last_capture_timestamp = t_capture # Store capture time

                        # --- Calculate Point Cloud FPS ---
                        current_time = time.time()
                        time_delta = current_time - self.last_update_time
                        if time_delta > 1e-6: # Avoid division by zero
                            self.point_cloud_fps = 1.0 / time_delta
                        self.last_update_time = current_time
                        # -----------------------------

                        # Calculate actual number of vertices based on received data length
                        actual_num_vertices = 0
                        if vertices_data is not None:
                            # vertices_data is flattened, length is num_vertices * 3
                            actual_num_vertices = len(vertices_data) // 3
                            # Update point count state and label
                            self.current_point_count = actual_num_vertices
                            self.point_count_label.text = f"Points: {self.current_point_count}"
                        # Removed Debug Print for Point Range

                        if vertices_data is not None and colors_data is not None and actual_num_vertices > 0:
                            # Delete previous vertex list if it exists
                            if self.vertex_list:
                                try:
                                    self.vertex_list.delete()
                                except Exception as e_del:
                                    print(f"Error deleting previous vertex list: {e_del}")
                                self.vertex_list = None

                            # Create new vertex list using ACTUAL vertex count
                            try:
                                self.vertex_list = self.shader_program.vertex_list(
                                    actual_num_vertices, # Use actual count based on data length
                                    gl.GL_POINTS,
                                    vertices=('f', vertices_data),
                                    colors=('f', colors_data)
                                )
                                self.frame_count_display += 1 # Increment display counter
                                # Removed final debug print
                            except Exception as e_create:
                                 print(f"Error creating vertex list: {e_create}")
                                 traceback.print_exc()
                        else:
                            # Handle empty frame or invalid data
                            # Increment display counter even if data is bad/empty?
                            self.frame_count_display += 1

                    except Exception as e_unpack:
                        print(f"Error unpacking or processing vertex data: {e_unpack}")
                        traceback.print_exc()
                        # Optionally clear vertex list here too
                        # if self.vertex_list:
                        #     self.vertex_list.delete()
                        #     self.vertex_list = None

        except queue.Empty:
            pass # No new data is fine


    def update_camera(self, dt):
        """Scheduled function to handle camera movement based on key states."""

        # Removed viewer camera motion confidence calculation

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

        # Removed update for previous viewer camera state

    def get_camera_matrices(self):
        rot_y = -math.radians(self.camera_rotation_y)
        rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(math.sin(rot_y) * math.cos(rot_x), -math.sin(rot_x), -math.cos(rot_y) * math.cos(rot_x)).normalize()
        target = self.camera_position + forward
        view = Mat4.look_at(self.camera_position, target, self.world_up_vector)
        projection = Mat4.perspective_projection(self.aspect_ratio, z_near=0.1, z_far=1000.0, fov=60.0)
        return projection, view

    def on_draw(self):
        self.clear()
        projection, current_view = self.get_camera_matrices()
        self.shader_program.use()
        self.shader_program['projection'] = projection
        self.shader_program['view'] = current_view
        # Removed passing cameraPosition uniform
        # Removed passing viewportHeight uniform
        # Pass viewport size for geometry shader calculations
        self.shader_program['viewportSize'] = (float(self.width), float(self.height))
        # Pass the current input scale factor to the shader
        self.shader_program['inputScaleFactor'] = float(self.current_scale_factor)
        # Pass the current point size boost factor to the shader
        self.shader_program['pointSizeBoost'] = float(self.point_size_boost)
        self.shader_program['inputScaleFactor'] = float(self.current_scale_factor)
        # --- Draw VertexList directly ---
        # print("DEBUG: on_draw called") # Optional: Uncomment if needed to confirm on_draw frequency
        if self.vertex_list:
            print(f"DEBUG: Attempting to draw vertex_list (exists: True, count: {self.vertex_list.count})") # Check if list exists and has points
            try:
                self.vertex_list.draw(gl.GL_POINTS)
                # print("DEBUG: vertex_list.draw() called successfully") # Optional: Uncomment if needed
            except Exception as e:
                print(f"Error during vertex_list.draw: {e}")
        else:
            print("DEBUG: Skipping draw (vertex_list is None)") # Check if list is None
        # --------------------------------

        self.shader_program.stop()

        # --- Draw Status Label & FPS ---
        # Disable depth test for 2D UI rendering
        gl.glDisable(gl.GL_DEPTH_TEST)

        # Calculate latency
        latency_ms = (time.time() - self.last_capture_timestamp) * 1000 if self.last_capture_timestamp else 0

        # Update label text (Motion Confidence removed)
        status_text = f"PointCloud FPS: {self.point_cloud_fps:.1f} | Latency: {latency_ms:.0f}ms | {self.status_message}"
        self.status_label.text = status_text

        # Draw the UI batch (which contains the label)
        self.ui_batch.draw()

        # Removed FPS display draw call

        # Re-enable depth test for next frame's 3D rendering
        gl.glEnable(gl.GL_DEPTH_TEST)
        # -------------------------


    # --- Input Handlers (Simplified for brevity, same as before) ---
    def on_resize(self, width, height):
        gl.glViewport(0, 0, max(1, width), max(1, height))
        # Removed FPS display update
    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT: self.set_exclusive_mouse(True); self.mouse_down = True
    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT: self.set_exclusive_mouse(False); self.mouse_down = False
    def on_mouse_motion(self, x, y, dx, dy):
        if self.mouse_down:
            sensitivity = 0.1; self.camera_rotation_y -= dx * sensitivity; self.camera_rotation_y %= 360
            self.camera_rotation_x += dy * sensitivity; self.camera_rotation_x = max(min(self.camera_rotation_x, 89.9), -89.9)
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        zoom_speed = 0.5; rot_y = -math.radians(self.camera_rotation_y); rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(math.sin(rot_y) * math.cos(rot_x), -math.sin(rot_x), -math.cos(rot_y) * math.cos(rot_x)).normalize()
        self.camera_position += forward * scroll_y * zoom_speed
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.set_exclusive_mouse(False)
            self.close()
        elif symbol == pyglet.window.key.UP:
            # Increase scale factor
            self.current_scale_factor += 0.1
            if hasattr(self, 'scale_factor_ref'): # Check if ref exists (thread started)
                self.scale_factor_ref[0] = self.current_scale_factor
            self.scale_factor_label.text = f"Scale: {self.current_scale_factor:.2f}x"
            print(f"Scale factor increased to: {self.current_scale_factor:.2f}")
        elif symbol == pyglet.window.key.DOWN:
            # Decrease scale factor, with a minimum limit
            self.current_scale_factor = max(0.1, self.current_scale_factor - 0.1) # Min scale 0.1
            if hasattr(self, 'scale_factor_ref'): # Check if ref exists
                self.scale_factor_ref[0] = self.current_scale_factor
            self.scale_factor_label.text = f"Scale: {self.current_scale_factor:.2f}x"
            print(f"Scale factor decreased to: {self.current_scale_factor:.2f}")
        elif symbol == pyglet.window.key.RIGHT:
            # Increase point size boost
            self.point_size_boost += 0.1
            self.point_boost_label.text = f"Pt Boost: {self.point_size_boost:.2f}x"
            print(f"Point boost increased to: {self.point_size_boost:.2f}")
        elif symbol == pyglet.window.key.LEFT:
            # Decrease point size boost, with a minimum limit
            self.point_size_boost = max(0.1, self.point_size_boost - 0.1) # Min boost 0.1
            self.point_boost_label.text = f"Pt Boost: {self.point_size_boost:.2f}x"
            print(f"Point boost decreased to: {self.point_size_boost:.2f}")
        # Add other key presses if needed (like pause, loop toggle - less relevant for live view)

    def on_close(self):
        print("Window closing, stopping inference thread...")
        self._exit_event.set()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.vertex_list:
            try: self.vertex_list.delete()
            except Exception as e_del: print(f"Error deleting vertex list on close: {e_del}")
        if self.shader_program: self.shader_program.delete()
        super().on_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live SLAM viewer using UniK3D.")
    parser.add_argument("--model", type=str, default="unik3d-vitl", help="Name of the UniK3D model to use (e.g., unik3d-vits, unik3d-vitb, unik3d-vitl)")
    parser.add_argument("--interval", type=int, default=1, help="Run inference every N frames (default: 1).")
    # Removed --enable-depth-smoothing argument
    parser.add_argument("--disable-point-smoothing", action="store_true", help="Disable temporal smoothing on 3D points (default: enabled).") # Clarified default
    args = parser.parse_args()

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    # Pass smoothing flags from args to window constructor
    window = LiveViewerWindow(model_name=args.model, inference_interval=args.interval,
                              # Removed depth smoothing argument
                              disable_point_smoothing=args.disable_point_smoothing,
                              width=1024, height=768, caption='Live UniK3D SLAM Viewer', resizable=True)
    try:
        pyglet.app.run()
    except Exception as e:
        print("\n--- Uncaught Exception ---")
        traceback.print_exc()
        print("------------------------")
    finally:
        print("Exiting application.")