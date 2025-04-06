import pyglet
import pyglet.gl as gl
import numpy as np
import os
import argparse
import math
import threading
import mss
import numpy as np
import glob
import re
import tkinter as tk
from tkinter import filedialog
import win32gui # Added for window enumeration
import win32con # Added for window enumeration constants
import time
import queue # For thread-safe communication
import cv2 # For camera capture
import torch # For UniK3D model and tensors
import json # For saving/loading settings
import tkinter as tk
from tkinter import filedialog
import glob # For finding files
import re # For sorting files numerically
import trimesh # For GLB saving/loading

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
    "input_mode": "Live", # "Live", "File", "GLB Sequence"
    "input_filepath": "", # Can be file or directory path
    "render_mode": 2, # 0=Square, 1=Circle, 2=Gaussian
    "falloff_factor": 5.0,
    "saturation": 1.5,
    "brightness": 1.0,
    "contrast": 1.1,
    "sharpness": 1.1,
    "enable_sharpening": False,
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
    "gradient_influence_scale": 1.0,
    "playback_speed": 1.0, # For video/GLB sequence files
    "loop_video": True, # For video/GLB sequence files
    "is_recording": False, # Recording state
    "recording_output_dir": "recording_output", # Default output dir
    "show_camera_feed": False,
    "show_depth_map": False,
    "show_edge_map": False,
    "show_smoothing_map": False,
    # Overlay Toggles
    "show_fps_overlay": False,
    "show_points_overlay": False,
    "show_input_fps_overlay": False,
    "show_depth_fps_overlay": False,
    "show_latency_overlay": False,
    "live_processing_mode": "Real-time", # "Real-time" or "Buffered"
}

# --- GLB Saving Helper ---
def save_glb(filepath, points, colors=None):
    """Saves a point cloud to a GLB file using trimesh."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        num_points = points.shape[0]

        # Trimesh expects colors as RGBA uint8
        point_colors = None
        if colors is not None and colors.shape[0] == num_points:
            # Convert float 0-1 to uint8 0-255
            colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
            # Add alpha channel (fully opaque)
            point_colors = np.hstack((colors_uint8, np.full((num_points, 1), 255, dtype=np.uint8)))
        else:
             if colors is not None: # Warn if colors provided but invalid
                 print(f"Warning: Color data shape mismatch or invalid for {filepath}. Saving without colors.")

        # Create a trimesh PointCloud object
        # Save coordinates as provided (assuming they are already transformed)
        cloud = trimesh.points.PointCloud(vertices=points, colors=point_colors)

        # Export to GLB
        cloud.export(filepath, file_type='glb')
        # print(f"Saved {filepath}") # Optional
    except Exception as e:
        print(f"Error saving GLB file {filepath}: {e}")
        traceback.print_exc()


# --- GLB Loading Helper ---
def load_glb(filepath):
    """Loads points and colors from a GLB file using trimesh."""
    try:
        # Load the GLB file
        mesh = trimesh.load(filepath, file_type='glb', process=False) # process=False to keep original data

        if isinstance(mesh, trimesh.points.PointCloud):
            points = np.array(mesh.vertices, dtype=np.float32)
            # No flips here, return raw data
            colors = None
            if hasattr(mesh, 'colors') and mesh.colors is not None and len(mesh.colors) == len(points):
                # Convert RGBA uint8 to RGB float 0-1
                colors = np.array(mesh.colors[:, :3], dtype=np.float32) / 255.0
            return points, colors
        elif isinstance(mesh, trimesh.Trimesh):
             # If it loaded as a mesh, just use its vertices
             print(f"Warning: Loaded GLB {filepath} as Trimesh, using vertices only.")
             points = np.array(mesh.vertices, dtype=np.float32)
             # No flips here, return raw data
             # Try to get vertex colors if they exist
             colors = None
             if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(points):
                 colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
             return points, colors
        else:
            print(f"Warning: Loaded GLB {filepath} is not a PointCloud or Trimesh.")
            return None, None

    except Exception as e:
        print(f"Error loading GLB file {filepath}: {e}")
        return None, None


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

# --- Window Enumeration Helper (Windows Only) ---
def _get_window_list():
    """Enumerates visible windows with titles."""
    windows = []
    def callback(hwnd, _):
        # Check if window is visible and has a title
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            # Optional: Filter out certain window types if needed
            # style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            # if style & win32con.WS_DISABLED == 0: # Example: exclude disabled windows
            windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    try:
        win32gui.EnumWindows(callback, None)
    except Exception as e:
        print(f"Error enumerating windows: {e}") # Catch potential errors
    # Sort by title for easier selection
    return sorted(windows, key=lambda x: x[1])


# --- Inference Thread Helper Functions ---

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

    # --- Initialize based on input_mode ---
    if input_mode == "Live Camera": # Use exact name from UI
        print("Initializing camera...")
        data_queue.put(("status", "Initializing camera..."))
        cap = cv2.VideoCapture(0) # TODO: Allow selecting camera index
        if cap.isOpened():
            is_video = True # Treat live feed as video
            frame_source_name = "Live Camera"
            video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            print("Camera initialized.")
            data_queue.put(("status", "Camera ready."))
        else:
            error_message = "Cannot open camera."
            cap = None # Ensure cap is None on error

    elif input_mode == "File (Video/Image)": # Use exact name from UI
        if not input_filepath or not os.path.isfile(input_filepath):
             error_message = f"Invalid or missing file path: {input_filepath}"
        else:
            frame_source_name = os.path.basename(input_filepath)
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
                cap = None # Ensure cap is None
                print("Failed to open as video, trying as image...")
                try:
                    image_frame = cv2.imread(input_filepath)
                    if image_frame is not None:
                        is_image = True
                        video_total_frames = 1 # Single frame
                        video_fps = 1 # N/A for image
                        playback_state_ref["total_frames"] = video_total_frames
                        playback_state_ref["current_frame"] = 0
                        print("Image file loaded successfully.")
                        data_queue.put(("status", f"Loaded image: {frame_source_name}"))
                    else:
                        error_message = f"Cannot open file as video or image: {frame_source_name}"
                except Exception as e_img:
                    error_message = f"Error reading file: {frame_source_name} ({e_img})"

    elif input_mode == "GLB Sequence": # Use exact name from UI
        if not input_filepath or not os.path.isdir(input_filepath):
             error_message = f"Invalid or missing directory path: {input_filepath}"
        else:
            frame_source_name = os.path.basename(input_filepath)
            print(f"Scanning directory for GLB files: {input_filepath}")
            data_queue.put(("status", f"Scanning directory: {frame_source_name}..."))
            glb_files = sorted(glob.glob(os.path.join(input_filepath, "*.glb")),
                               key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)) if re.search(r'(\d+)', os.path.basename(x)) else -1)
            if not glb_files:
                error_message = f"No .glb files found in: {frame_source_name}"
            else:
                is_glb_sequence = True
                # Don't reassign input_mode here
                video_total_frames = len(glb_files)
                video_fps = 30 # Assume 30 FPS
                playback_state_ref["total_frames"] = video_total_frames
                playback_state_ref["current_frame"] = 0
                print(f"GLB sequence loaded successfully ({video_total_frames} frames).")
                data_queue.put(("status", f"Loaded GLB sequence: {frame_source_name}"))

    elif input_mode == "Screen Capture": # Use exact name from UI
        # No specific initialization needed here, capture happens per-frame
        is_video = True # Treat screen capture like a video feed
        frame_source_name = "Screen Capture"
        video_fps = 30 # Assume target FPS
        print("Screen Capture mode initialized.")
        data_queue.put(("status", "Screen Capture ready."))

    else:
        # This case handles None or any other invalid string
        error_message = f"Invalid input mode selected for initialization: {input_mode}"

    if error_message:
        print(f"Error: {error_message}")
        data_queue.put(("error", error_message))
        # Return None for critical objects if error occurred
        return None, None, False, False, False, [], "Error", 0, 30, None, error_message

    return cap, image_frame, is_video, is_image, is_glb_sequence, glb_files, frame_source_name, video_total_frames, video_fps, input_mode, None


def _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index, playback_state_ref, last_frame_read_time_ref, video_fps, is_video, is_image, is_glb_sequence, image_frame, frame_count, data_queue, frame_source_name, selected_monitor_index, selected_window_hwnd): # Added screen capture args
    """Reads the next frame/GLB based on playback state. Returns timing delta."""
    frame = None
    points_xyz_np = None
    colors_np = None
    ret = False
    frame_read_delta_t = 0.0 # Time since last successful read
    current_time = time.time()
    last_frame_read_time = last_frame_read_time_ref[0] # Get time from mutable ref
    new_sequence_frame_index = sequence_frame_index
    end_of_stream = False
    read_successful = False # Flag to track if we got new data this cycle

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
        playback_state_ref["restart"] = False # Consume restart flag

    read_next_frame = is_playing or (is_image and frame_count == 0) # Read if playing or first image frame

    if not read_next_frame and (is_video or is_glb_sequence):
        time.sleep(0.05) # Sleep if paused
        return None, None, None, False, new_sequence_frame_index, frame_read_delta_t, False # Indicate no frame read

    target_delta = (1.0 / video_fps) / playback_speed if video_fps > 0 and playback_speed > 0 else 0.1
    time_since_last_read = current_time - last_frame_read_time # Use local copy for calculation

    # Check if it's time to read the next frame or if it's the first image frame
    if time_since_last_read >= target_delta or (is_image and frame_count == 0):
        # --- Attempt to read/load based on input mode ---
        # --- Read/Load based on input_mode ---
        if input_mode == "Live Camera" and cap:
            ret, frame = cap.read()
            if ret:
                read_successful = True
                new_sequence_frame_index = 0 # No sequence index for live camera
                playback_state_ref["current_frame"] = frame_count # Use frame_count for live?
            else:
                print("Warning: Failed to read frame from live camera.")
                ret = False # Indicate failure
        elif input_mode == "File (Video/Image)" and cap:
            ret, frame = cap.read()
            if ret:
                new_sequence_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                playback_state_ref["current_frame"] = new_sequence_frame_index
                read_successful = True
            else: # End of video
                if loop_video:
                    print("Looping video.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    new_sequence_frame_index = 0
                    playback_state_ref["current_frame"] = 0
                    ret, frame = cap.read() # Read first frame after loop
                    if ret:
                        read_successful = True # Read succeeded after loop
                    else:
                        end_of_stream = True # Failed to read after loop
                else:
                    end_of_stream = True # Not looping, end of stream
        elif input_mode == "GLB Sequence":
            if new_sequence_frame_index >= len(glb_files): # End of sequence
                if loop_video:
                    print("Looping GLB sequence.")
                    new_sequence_frame_index = 0
                    playback_state_ref["current_frame"] = 0
                else:
                    end_of_stream = True

            if not end_of_stream:
                current_glb_path = glb_files[new_sequence_frame_index]
                points_xyz_np, colors_np = load_glb(current_glb_path) # Assume load_glb returns points, colors
                if points_xyz_np is not None:
                    frame = points_xyz_np # Use points as 'frame' data for GLB
                    # colors_np are also available if needed later
                    ret = True
                    playback_state_ref["current_frame"] = new_sequence_frame_index + 1
                    new_sequence_frame_index += 1
                    read_successful = True
                else:
                    print(f"Error loading GLB frame: {current_glb_path}")
                    end_of_stream = True # Stop on error
        elif input_mode == "Screen Capture":
            try:
                with mss.mss() as sct:
                    capture_area = None
                    if selected_window_hwnd != 0:
                        # Capture specific window using HWND
                        # Get window rect (left, top, right, bottom)
                        rect = win32gui.GetWindowRect(selected_window_hwnd)
                        # Check if window is minimized (rect might be off-screen)
                        if rect[0] < -1000 or rect[1] < -1000:
                             print(f"Warning: Target window (HWND {selected_window_hwnd}) might be minimized. Capturing monitor instead.")
                             capture_area = sct.monitors[selected_monitor_index]
                        else:
                             # Adjust for potential DWM composition offsets if needed (usually not necessary with GetWindowRect)
                             capture_area = {'left': rect[0], 'top': rect[1], 'width': rect[2] - rect[0], 'height': rect[3] - rect[1]}
                             # Basic validation
                             if capture_area['width'] <= 0 or capture_area['height'] <= 0:
                                 print(f"Warning: Invalid window dimensions for HWND {selected_window_hwnd}. Capturing monitor.")
                                 capture_area = sct.monitors[selected_monitor_index]

                    else:
                        # Capture selected monitor (index 0 is 'all', 1+ are physical)
                        if selected_monitor_index < len(sct.monitors):
                             capture_area = sct.monitors[selected_monitor_index]
                        else:
                             print(f"Warning: Invalid monitor index {selected_monitor_index}, defaulting to primary (1).")
                             capture_area = sct.monitors[1] # Fallback to primary

                    sct_img = sct.grab(capture_area)
                    img_bgra = np.array(sct_img)
                    frame = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
                    ret = True
                    read_successful = True
                    new_sequence_frame_index = 0
                    playback_state_ref["current_frame"] = frame_count
            except Exception as e_screen:
                print(f"Error capturing screen: {e_screen}")
                traceback.print_exc() # Print full traceback for debugging
                ret = False
        elif input_mode == "File (Video/Image)" and is_image: # Handle single image case
             if frame_count == 0:
                 frame = image_frame
                 if frame is not None:
                     ret = True
                     read_successful = True
                 else:
                     ret = False
             else: # Only process image once
                 end_of_stream = True
        # --- End read/load attempt ---

        # Update timing *once* if read was successful this cycle
        if read_successful:
            frame_read_delta_t = current_time - last_frame_read_time
            last_frame_read_time_ref[0] = current_time # Update ref

        # Handle end of stream condition
        if end_of_stream:
            print(f"End of {input_mode}: {frame_source_name}")
            data_queue.put(("status", f"Finished processing: {frame_source_name}"))
            # Return timing delta even on the last frame before stopping
            return None, None, None, False, new_sequence_frame_index, frame_read_delta_t, True

        # Return results for this cycle (if not end of stream)
        return frame, points_xyz_np, colors_np, ret, new_sequence_frame_index, frame_read_delta_t, False

    else: # Not time to read next frame yet
        sleep_time = target_delta - time_since_last_read # Calculate sleep_time here
        time.sleep(max(0.001, sleep_time))
        return None, None, None, False, new_sequence_frame_index, frame_read_delta_t, False # Indicate no frame read


def _apply_sharpening(rgb_frame, edge_params_ref):
    """Applies sharpening to the RGB frame if enabled."""
    enable_sharpening = edge_params_ref.get("enable_sharpening", False)
    sharpness_amount = edge_params_ref.get("sharpness", 1.5) # Use sharpness from edge_params
    if enable_sharpening and sharpness_amount > 0 and rgb_frame is not None:
        try:
            blurred = cv2.GaussianBlur(rgb_frame, (0, 0), 3)
            # Adjust addWeighted parameters based on sharpness_amount
            # sharpness_amount=1.0 means no change, >1 means sharpen
            alpha = 1.0 + (sharpness_amount - 1.0) * 0.5 # Scale effect
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


def _process_inference_results(predictions, rgb_frame_processed, device, edge_params_ref, prev_depth_map, smoothed_mean_depth, smoothed_points_xyz, frame_h, frame_w):
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

    if predictions is None:
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz

    points_xyz = None
    current_depth_map = None

    # Extract points or calculate from depth/rays
    if 'rays' in predictions and 'depth' in predictions:
        current_depth_map = predictions['depth'].squeeze()
        rays = predictions['rays'].squeeze()
        if rays.shape[0] == 3 and rays.ndim == 3: rays = rays.permute(1, 2, 0)
        if current_depth_map.shape == rays.shape[:2]:
            depth_to_multiply = current_depth_map.unsqueeze(-1)
            points_xyz = rays * depth_to_multiply
        else: print("Warning: Depth and Ray shape mismatch.")
    elif "points" in predictions:
        points_xyz = predictions["points"]
    else:
        print("Warning: No points/depth found in predictions.")
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz

    if points_xyz is None or points_xyz.numel() == 0:
        print("Warning: No valid points generated.")
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz

    # --- Edge Detection & Depth Processing (if depth map available) ---
    combined_edge_map = None
    depth_gradient_map = None
    per_pixel_depth_motion_map = None
    final_alpha_map = None

    if current_depth_map is not None:
        try:
            depth_thresh1 = edge_params_ref["depth_threshold1"]
            depth_thresh2 = edge_params_ref["depth_threshold2"]
            rgb_thresh1 = edge_params_ref["rgb_threshold1"]
            rgb_thresh2 = edge_params_ref["rgb_threshold2"]

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
                depth_motion_threshold = 0.1 # TODO: Make configurable?
                per_pixel_depth_motion = torch.clamp(depth_diff / depth_motion_threshold, 0.0, 1.0)
                # normalized_depth = torch.clamp(current_depth_map / 10.0, 0.0, 1.0)
                # distance_modulated_motion = per_pixel_depth_motion * (1.0 - normalized_depth)
                per_pixel_depth_motion_map = per_pixel_depth_motion # Use unmodulated for now

            new_prev_depth_map = current_depth_map.clone()

            # Depth scaling based on smoothed mean depth (optional, maybe remove?)
            # current_mean_depth = current_depth_map.mean().item()
            # if new_smoothed_mean_depth is None: new_smoothed_mean_depth = current_mean_depth
            # else: new_smoothed_mean_depth = 0.1 * current_mean_depth + 0.9 * new_smoothed_mean_depth # Simple EMA
            # if current_mean_depth > 1e-6: scale_factor = new_smoothed_mean_depth / current_mean_depth
            # else: scale_factor = 1.0
            # scaled_depth_map = current_depth_map * scale_factor
            scaled_depth_map_for_queue = current_depth_map # Queue original depth for now

        except Exception as e_edge:
            print(f"Error during edge/depth processing: {e_edge}")
            combined_edge_map = None
            edge_map_viz = None
            depth_gradient_map = None
            per_pixel_depth_motion_map = None
            scaled_depth_map_for_queue = current_depth_map # Still queue original depth if possible

    # --- Temporal Smoothing ---
    enable_point_smoothing = edge_params_ref["enable_point_smoothing"]
    if enable_point_smoothing:
        if new_smoothed_points_xyz is None or new_smoothed_points_xyz.shape != points_xyz.shape:
            print("Initializing/Resetting smoothing state.")
            new_smoothed_points_xyz = points_xyz.clone()
            # Assume full alpha initially if smoothing just started
            if points_xyz.ndim == 3: final_alpha_map = torch.ones_like(points_xyz[:,:,0])
            elif points_xyz.ndim == 2: final_alpha_map = torch.ones(points_xyz.shape[0], device=device)
        else:
            # Calculate alpha map
            min_alpha_points = edge_params_ref["min_alpha_points"]
            max_alpha_points = edge_params_ref["max_alpha_points"]

            # Base alpha on motion (if available)
            if per_pixel_depth_motion_map is not None and per_pixel_depth_motion_map.shape == points_xyz.shape[:2]:
                 motion_factor_points = per_pixel_depth_motion_map ** 2.0 # Square to emphasize motion
                 base_alpha_map = min_alpha_points + (max_alpha_points - min_alpha_points) * motion_factor_points
            else: # Default to max alpha if no motion map
                 base_alpha_map = torch.full_like(points_xyz[:,:,0], max_alpha_points) if points_xyz.ndim == 3 else torch.full((points_xyz.shape[0],), max_alpha_points, device=device)

            # Modulate alpha by edges (if enabled and available)
            enable_edge_aware = edge_params_ref["enable_edge_aware"]
            if enable_edge_aware and combined_edge_map is not None and combined_edge_map.shape == base_alpha_map.shape:
                edge_mask_tensor = torch.from_numpy(combined_edge_map / 255.0).float().to(device)
                edge_influence = edge_params_ref["influence"]
                local_influence = edge_influence

                # Modulate edge influence by gradient (if available)
                grad_influence_scale = edge_params_ref["gradient_influence_scale"]
                if depth_gradient_map is not None and depth_gradient_map.shape == base_alpha_map.shape:
                    local_influence = edge_influence * torch.clamp(depth_gradient_map * grad_influence_scale, 0.0, 1.0)

                final_alpha_map = torch.lerp(base_alpha_map, torch.ones_like(base_alpha_map), edge_mask_tensor * local_influence)
            else:
                final_alpha_map = base_alpha_map

            # Apply smoothing
            final_alpha_map_unsqueezed = final_alpha_map.unsqueeze(-1) if final_alpha_map.ndim == 2 else final_alpha_map.unsqueeze(-1) # Handle 2D/3D points_xyz
            new_smoothed_points_xyz = final_alpha_map_unsqueezed * points_xyz + (1.0 - final_alpha_map_unsqueezed) * new_smoothed_points_xyz
    else: # Smoothing disabled
        new_smoothed_points_xyz = points_xyz
        # Create dummy alpha map for visualization if needed
        if points_xyz.ndim == 3: final_alpha_map = torch.ones_like(points_xyz[:,:,0])
        elif points_xyz.ndim == 2: final_alpha_map = torch.ones(points_xyz.shape[0], device=device)


    # --- Create Smoothing Map Visualization ---
    if final_alpha_map is not None:
        try:
            smoothing_map_vis_np = (final_alpha_map.cpu().numpy() * 255).astype(np.uint8)
            if smoothing_map_vis_np.ndim == 2: # Ensure it's 3 channels for texture
                 smoothing_map_viz = cv2.cvtColor(smoothing_map_vis_np, cv2.COLOR_GRAY2RGB)
            elif smoothing_map_vis_np.ndim == 1: # Handle potential 1D case (unlikely)
                 smoothing_map_viz = np.zeros((frame_h, frame_w, 3), dtype=np.uint8) # Placeholder
            else: smoothing_map_viz = smoothing_map_vis_np # Assume already RGB
        except Exception as e_smooth_viz:
            print(f"Error creating smoothing map viz: {e_smooth_viz}")
            smoothing_map_viz = None
    else: smoothing_map_viz = None


    # --- Prepare Output Point Cloud ---
    points_xyz_to_process = new_smoothed_points_xyz
    if points_xyz_to_process is None or points_xyz_to_process.numel() == 0:
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz

    points_xyz_np_processed = points_xyz_to_process.squeeze().cpu().numpy()

    # --- Transform Coordinates for OpenGL/GLB standard (+X Right, +Y Up, -Z Forward) ---
    if points_xyz_np_processed.ndim >= 2 and points_xyz_np_processed.shape[-1] == 3:
        points_xyz_np_processed[..., 1] *= -1.0 # Flip Y (Down -> Up)
        points_xyz_np_processed[..., 2] *= -1.0 # Flip Z (Forward -> -Forward)
    # ------------------------------------

    # Reshape and get vertex count
    try:
        if points_xyz_np_processed.ndim == 3 and points_xyz_np_processed.shape[0] == 3: # C, H, W format?
            points_xyz_np_processed = np.transpose(points_xyz_np_processed, (1, 2, 0)) # H, W, C
            num_vertices = points_xyz_np_processed.shape[0] * points_xyz_np_processed.shape[1]
            points_xyz_np_processed = points_xyz_np_processed.reshape(num_vertices, 3)
        elif points_xyz_np_processed.ndim == 2 and points_xyz_np_processed.shape[1] == 3: # N, C format
             num_vertices = points_xyz_np_processed.shape[0]
        elif points_xyz_np_processed.ndim == 3 and points_xyz_np_processed.shape[2] == 3: # H, W, C format
             num_vertices = points_xyz_np_processed.shape[0] * points_xyz_np_processed.shape[1]
             points_xyz_np_processed = points_xyz_np_processed.reshape(num_vertices, 3)
        else:
            print(f"Warning: Unexpected points_xyz_np shape after processing: {points_xyz_np_processed.shape}")
            num_vertices = 0
    except Exception as e_reshape:
            print(f"Error reshaping points_xyz_np: {e_reshape}")
            num_vertices = 0

    # --- Sample Colors ---
    if num_vertices > 0 and rgb_frame_processed is not None:
        try:
            if rgb_frame_processed.shape[0] == frame_h and rgb_frame_processed.shape[1] == frame_w:
                colors_np_processed = rgb_frame_processed.reshape(frame_h * frame_w, 3)
                # Ensure colors match potentially reshaped points (this assumes points correspond to pixels)
                if colors_np_processed.shape[0] == num_vertices:
                    if colors_np_processed.dtype == np.uint8:
                        colors_np_processed = colors_np_processed.astype(np.float32) / 255.0
                    elif colors_np_processed.dtype == np.float32:
                        colors_np_processed = np.clip(colors_np_processed, 0.0, 1.0)
                    else:
                        print(f"Warning: Unexpected color dtype {colors_np_processed.dtype}, using white.")
                        colors_np_processed = None
                else:
                     print(f"Warning: Point count ({num_vertices}) mismatch with color pixels ({colors_np_processed.shape[0]}). No colors.")
                     colors_np_processed = None # Mismatch after reshape
            else:
                 print(f"Warning: Dimension mismatch between points ({frame_h}x{frame_w}) and processed frame ({rgb_frame_processed.shape[:2]})")
                 colors_np_processed = None
        except Exception as e_color:
            print(f"Error processing colors: {e_color}")
            colors_np_processed = None
    else:
        colors_np_processed = None # No vertices or no frame

    # Use white if colors failed
    if colors_np_processed is None and num_vertices > 0:
        colors_np_processed = np.ones((num_vertices, 3), dtype=np.float32)


    return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz


def _handle_recording(points_xyz_np, colors_np, recording_state_ref, sequence_frame_index, recorded_frame_counter, is_video, is_glb_sequence, data_queue):
    """Handles saving the current frame to GLB if recording is active."""
    new_recorded_frame_counter = recorded_frame_counter
    is_recording = recording_state_ref.get("is_recording", False)
    output_dir = recording_state_ref.get("output_dir", "recording_output")

    if is_recording and points_xyz_np is not None:
        new_recorded_frame_counter += 1
        # Use sequence_frame_index for GLB naming if available and non-zero, else use recorded_frame_counter
        # Adjust sequence_frame_index because it's incremented *after* loading
        current_playback_index = sequence_frame_index -1 if (is_video or is_glb_sequence) else -1
        save_index = current_playback_index if current_playback_index >= 0 else new_recorded_frame_counter

        glb_filename = os.path.join(output_dir, f"frame_{save_index:05d}.glb")
        # Save points as they are (coordinate system should be standard here)
        # Save the already transformed points (X, Y-up, Z-backward)
        save_glb(glb_filename, points_xyz_np, colors_np)
        if new_recorded_frame_counter % 30 == 0: # Log every 30 frames
            data_queue.put(("status", f"Recording frame {new_recorded_frame_counter}..."))

    return new_recorded_frame_counter


def _queue_results(data_queue, vertices_flat, colors_flat, num_vertices, rgb_frame_orig, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, t_capture, sequence_frame_index, video_total_frames, current_recorded_count, frame_count, frame_read_delta_t, depth_process_delta_t, latency_ms):
    """Puts the processed data into the queue for the main thread."""
    if not data_queue.full():
            data_queue.put((vertices_flat, colors_flat, num_vertices,
                            rgb_frame_orig,
                            scaled_depth_map_for_queue,
                            edge_map_viz,
                            smoothing_map_viz,
                            t_capture,
                            sequence_frame_index,
                            video_total_frames,
                            current_recorded_count,
                            frame_read_delta_t, depth_process_delta_t, latency_ms)) # Add timing info
    else:
        print(f"Warning: Viewer queue full, dropping frame {frame_count}.")
        # Optionally put a status message instead of dropping silently
        # data_queue.put(("status", f"Viewer queue full, dropping frame {frame_count}"))


# --- Main Inference Thread Function ---
# --- Inference Thread Function ---
# --- Main Inference Thread Function (Refactored) ---
def inference_thread_func(data_queue, exit_event, model_name, inference_interval,
                          scale_factor_ref, edge_params_ref,
                          input_mode, input_filepath, playback_state,
                          recording_state, live_processing_mode,
                          selected_monitor_index, selected_window_hwnd): # Added screen capture args
    """Loads model, captures/loads data, runs inference, processes, and queues results."""
    print(f"Inference thread started. Mode: {input_mode}, File: {input_filepath if input_filepath else 'N/A'}")
    data_queue.put(("status", f"Inference thread started ({input_mode})..."))

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
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # --- Load Model (if needed) ---
        if input_mode != "GLB Sequence":
            print(f"Loading UniK3D model: {model_name}...")
            data_queue.put(("status", f"Loading model: {model_name}..."))
            print(f"Using device: {device}")
            model = UniK3D.from_pretrained(f"lpiccinelli/{model_name}")
            model = model.to(device)
            model.eval()
            print("Model loaded.")
            data_queue.put(("status", "Model loaded."))
        else:
            print("GLB Sequence mode: Skipping model load.")
            data_queue.put(("status", "GLB Sequence mode: Skipping model load."))


        # --- Initialize Input Source ---
        cap, image_frame, is_video, is_image, is_glb_sequence, glb_files, \
        frame_source_name, video_total_frames, video_fps, input_mode, error_message = \
            _initialize_input_source(input_mode, input_filepath, data_queue, playback_state) # Use renamed variable

        if error_message:
            return # Error already queued by helper

        # --- Main Loop ---
        frame_count = 0
        sequence_frame_index = 0
        # last_inference_time = time.time() # Replaced by depth_processed_time
        last_frame_read_time_ref = [time.time()] # Use mutable list/ref for helper function
        recorded_frame_counter = 0
        smoothed_points_xyz = None
        prev_depth_map = None
        smoothed_mean_depth = None
        prev_scale_factor = None # For detecting scale changes
        last_depth_processed_time = time.time() # For depth/processing FPS calculation

        while not exit_event.is_set():
            frame = None
            points_xyz_np_loaded = None
            colors_np_loaded = None
            ret = False
            frame_read_delta_t = 0.0
            end_of_stream = False
            t_capture = time.time() # Capture time at loop start for latency

            # --- Frame Acquisition Logic ---
            if input_mode == "Live" and is_video and cap:
                if live_processing_mode == "Real-time":
                    # Grab latest frame, discard older ones
                    grabbed = False
                    # Limit grabs? Or just grab until retrieve? Let's try simple grab/retrieve first.
                    # for _ in range(5): # Optional: Limit grabs to avoid blocking?
                    #     if not cap.grab(): break
                    if cap.grab(): # Grab the latest frame internally
                         t_capture = time.time() # Update capture time closer to actual frame
                         ret, frame = cap.retrieve() # Decode and return the grabbed frame
                         if ret:
                             current_time = time.time()
                             frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                             last_frame_read_time_ref[0] = current_time
                             # sequence_frame_index doesn't make sense here, maybe use frame_count?
                             playback_state["current_frame"] = frame_count + 1 # Update for UI display?
                         else:
                             print("Warning: Failed to retrieve frame in Real-time mode.")
                             time.sleep(0.01) # Avoid busy loop on retrieve error
                             continue # Skip processing this cycle
                    else:
                        print("Warning: Failed to grab frame in Real-time mode.")
                        time.sleep(0.01) # Avoid busy loop on grab error
                        continue # Skip processing this cycle

                else: # Buffered mode for Live Camera
                    frame, _, _, ret, sequence_frame_index, frame_read_delta_t, end_of_stream = \
                        _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index,
                                           playback_state, last_frame_read_time_ref, video_fps,
                                           is_video, is_image, is_glb_sequence, image_frame,
                                           frame_count, data_queue, frame_source_name,
                                           selected_monitor_index, selected_window_hwnd) # Pass screen capture args
            else: # File, GLB Sequence, or Image mode
                 frame, points_xyz_np_loaded, colors_np_loaded, ret, \
                 sequence_frame_index, frame_read_delta_t, end_of_stream = \
                     _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index,
                                        playback_state, last_frame_read_time_ref, video_fps,
                                        is_video, is_image, is_glb_sequence, image_frame,
                                        frame_count, data_queue, frame_source_name,
                                        selected_monitor_index, selected_window_hwnd) # Pass screen capture args

            # --- Loop/End/Skip Checks ---
            # Check if sequence looped or restarted (index reset by _read_or_load_frame)
            # Reset recorder counter accordingly
            # Use frame_count > 1 to avoid resetting on the very first frame
            if frame_count > 1 and sequence_frame_index <= 1 and input_mode != "Live": # Only reset for sequences/files
                 if recorded_frame_counter > 0: # Only reset if it was actually counting
                    print("DEBUG: Resetting recorded frame counter due to loop/restart.")
                    recorded_frame_counter = 0

            if end_of_stream:
                break # Exit loop if end of file/sequence and not looping

            if not ret: # If no frame was read/retrieved (e.g., paused or error)
                continue

            # --- Process Frame ---
            frame_count += 1
            print(f"Processing frame {frame_count} (Seq Idx: {sequence_frame_index})")
            # last_inference_time = time.time() # Replaced by depth_processed_time

            # Reset per-frame data
            rgb_frame_orig = None
            points_xyz_np_processed = None
            colors_np_processed = None
            num_vertices = 0
            scaled_depth_map_for_queue = None
            edge_map_viz = None
            smoothing_map_viz = None

            # --- Timing & State Reset ---
            depth_process_delta_t = 0.0 # Time since last successful depth processing
            # Check for scale factor changes to reset smoothing
            current_scale = scale_factor_ref[0]
            if prev_scale_factor is not None and abs(current_scale - prev_scale_factor) > 1e-3:
                print(f"Scale factor changed ({prev_scale_factor:.2f} -> {current_scale:.2f}). Resetting smoothing state.")
                smoothed_points_xyz = None
                prev_depth_map = None
                smoothed_mean_depth = None
            prev_scale_factor = current_scale


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
                else:
                    continue # Skip if GLB load failed
                # Mark depth processed time for GLB
                current_time = time.time()
                depth_process_delta_t = current_time - last_depth_processed_time
                last_depth_processed_time = current_time

            else: # Live, Video File, or Image File
                # --- Preprocess Frame (Scale, Sharpen) ---
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

                # --- Run Inference ---
                data_queue.put(("status", f"Running inference on frame {frame_count}..."))
                predictions = _run_model_inference(model, rgb_frame_processed, device) # Use processed frame for inference

                # --- Process Results ---
                data_queue.put(("status", f"Processing results for frame {frame_count}..."))
                points_xyz_np_processed, colors_np_processed, num_vertices, \
                scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, \
                prev_depth_map, smoothed_mean_depth, smoothed_points_xyz = \
                    _process_inference_results(predictions, rgb_frame_processed, device,
                                               edge_params_ref, prev_depth_map,
                                               smoothed_mean_depth, smoothed_points_xyz,
                                               frame_h, frame_w)
                # Mark depth processed time for inference
                current_time = time.time()
                depth_process_delta_t = current_time - last_depth_processed_time
                last_depth_processed_time = current_time


            # --- Handle Recording ---
            recorded_frame_counter = _handle_recording(points_xyz_np_processed, colors_np_processed,
                                                       recording_state, sequence_frame_index, # Changed recording_state_ref to recording_state
                                                       recorded_frame_counter, is_video, is_glb_sequence,
                                                       data_queue)

            # --- Calculate Latency & Queue Results ---
            latency_ms = (time.time() - t_capture) * 1000.0
            vertices_flat = points_xyz_np_processed.flatten() if points_xyz_np_processed is not None else None
            colors_flat = colors_np_processed.flatten() if colors_np_processed is not None else None

            # --- Queue Results (with Real-time handling) ---
            if input_mode == "Live" and live_processing_mode == "Real-time" and data_queue.full():
                print(f"Warning: Viewer queue full in Real-time mode, dropping frame {frame_count}.")
                # Skip queuing to prioritize processing next frame
            else:
                # Queue normally for Buffered mode or if queue is not full
                _queue_results(data_queue, vertices_flat, colors_flat, num_vertices,
                               rgb_frame_orig, scaled_depth_map_for_queue, edge_map_viz,
                               smoothing_map_viz, t_capture, sequence_frame_index,
                               video_total_frames, recorded_frame_counter, frame_count,
                               frame_read_delta_t, depth_process_delta_t, latency_ms) # Pass timing

            # Short sleep for live mode to prevent busy loop
            if is_video and input_mode == "Live":
                time.sleep(0.005)

    except Exception as e_thread:
        print(f"Error in inference thread: {e_thread}")
        traceback.print_exc()
        data_queue.put(("error", str(e_thread)))
        data_queue.put(("status", "Inference thread error!"))
    finally:
        if cap and is_video:
            cap.release()
        print("Inference thread finished.")
        data_queue.put(("status", "Inference thread finished."))


# --- Viewer Class ---
class LiveViewerWindow(pyglet.window.Window):
    def __init__(self, model_name, inference_interval=1,
                 disable_point_smoothing=False, # This arg is now effectively ignored
                 *args, **kwargs):
        # Store args before calling super
        self._model_name = model_name
        self._inference_interval = inference_interval
        # self._initial_disable_point_smoothing = disable_point_smoothing # No longer needed

        super().__init__(*args, **kwargs)

        self.selected_model_name = self._model_name # Initialize UI selection state

        self.vertex_list = None
        self.frame_count_display = 0
        self.current_point_count = 0
        self.last_update_time = time.time()
        self.point_cloud_fps = 0.0
        self.last_capture_timestamp = None

        # --- Control State Variables (Initialized in load_settings) ---
        self.input_mode = None
        self.input_filepath = None
        self.render_mode = None
        self.falloff_factor = None
        self.saturation = None
        self.brightness = None
        self.contrast = None
        self.sharpness = None
        self.enable_sharpening = None
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
        self.gradient_influence_scale = None
        self.playback_speed = None
        self.loop_video = None
        self.is_recording = None # Added
        self.recording_output_dir = None # Added
        self.show_camera_feed = None
        self.show_depth_map = None
        self.show_edge_map = None
        self.show_smoothing_map = None
        # Overlay Toggles
        self.show_fps_overlay = None
        self.show_points_overlay = None
        self.show_input_fps_overlay = None
        self.show_depth_fps_overlay = None
        self.show_latency_overlay = None
        self.live_processing_mode = None # Added attribute
        self.scale_factor_ref = None # Initialized in load_settings
        self.edge_params_ref = {} # Dictionary to pass edge params to thread
        self.playback_state = {} # Dictionary for playback control
        self.recording_state = {} # Dictionary for recording control

        # Local UI state variables (not saved in settings, but needed for immediate UI feedback)
        self.is_playing = True # Default to playing when starting/loading
        # Stats values
        self.input_fps = 0.0
        self.depth_fps = 0.0
        self.latency_ms = 0.0

        # Redundant state variables removed (now handled by self.playback_state, self.recording_state, and direct attributes like self.is_recording)

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
        # Ensure default input mode is set correctly after loading settings
        if self.input_mode not in ["Live Camera", "File (Video/Image)", "GLB Sequence", "Screen Capture"]:
            print(f"DEBUG: Input mode '{self.input_mode}' from settings invalid or missing, defaulting to 'Live Camera'.")
            self.input_mode = "Live Camera"

        # --- Status Display ---
        # self.ui_batch = pyglet.graphics.Batch() # Batch removed, using ImGui now
        self.status_message = "Initializing..."
        # --- Overlay Stats Setup (Separate Batches) ---
        # self.overlay_batch = pyglet.graphics.Batch() # Removed single batch
        label_color = (200, 200, 200, 200) # Semi-transparent white
        y_pos = self.height - 20
        self.fps_batch = pyglet.graphics.Batch()
        self.fps_label = pyglet.text.Label("", x=self.width - 10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.fps_batch, color=label_color)
        y_pos -= 20
        self.points_batch = pyglet.graphics.Batch()
        self.points_label = pyglet.text.Label("", x=self.width - 10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.points_batch, color=label_color)
        y_pos -= 20
        self.input_fps_batch = pyglet.graphics.Batch()
        self.input_fps_label = pyglet.text.Label("", x=self.width - 10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.input_fps_batch, color=label_color)
        y_pos -= 20
        self.depth_fps_batch = pyglet.graphics.Batch()
        self.depth_fps_label = pyglet.text.Label("", x=self.width - 10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.depth_fps_batch, color=label_color)
        y_pos -= 20
        self.latency_batch = pyglet.graphics.Batch()
        self.latency_label = pyglet.text.Label("", x=self.width - 10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.latency_batch, color=label_color)
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
        self._data_queue = queue.Queue(maxsize=30) # Increased queue size
        self._exit_event = threading.Event()
        self.inference_thread = None

        # Start the inference thread
        # self.scale_factor_ref and self.edge_params_ref initialized in load_settings
        self.start_inference_thread() # Start with loaded/default settings

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


    def start_inference_thread(self): # Removed arguments, uses self attributes
        # Stop existing thread first if running
        if self.inference_thread and self.inference_thread.is_alive():
            print("DEBUG: Stopping existing inference thread...")
            self.status_message = "Stopping previous source..." # Update status
            self._exit_event.set()
            self.inference_thread.join(timeout=2.0) # Increase timeout slightly
            if self.inference_thread.is_alive():
                 print("Warning: Inference thread did not stop quickly.")
                 # Optionally force kill? Risky.
            self.inference_thread = None
            self._exit_event.clear() # Reset event for new thread
            # Clear vertex list when source changes
            if self.vertex_list:
                try: self.vertex_list.delete()
                except: pass
                self.vertex_list = None
                self.current_point_count = 0


        # Ensure refs are initialized
        if not hasattr(self, 'scale_factor_ref') or self.scale_factor_ref is None:
             self.scale_factor_ref = [self.input_scale_factor]
        if not hasattr(self, 'edge_params_ref') or not self.edge_params_ref: # Keep ref name for dict itself
             self._update_edge_params()
        if not hasattr(self, 'playback_state') or not self.playback_state:
             self.update_playback_state() # Initialize playback state
        if not hasattr(self, 'recording_state') or not self.recording_state:
             self.update_recording_state() # Initialize recording state

        # Start new thread with current settings
        self.inference_thread = threading.Thread(
            target=inference_thread_func,
            args=(self._data_queue, self._exit_event, self._model_name, self._inference_interval,
                  self.scale_factor_ref,
                  self.edge_params_ref,
                  self.input_mode, # Pass current mode
                  self.input_filepath, # Pass current filepath
                  self.playback_state, # Pass playback state dict
                  self.recording_state, # Pass recording state dict
                  self.live_processing_mode, # Pass live processing mode
                  getattr(self, 'selected_monitor_index', 1), # Pass screen capture args w/ defaults
                  getattr(self, 'selected_window_hwnd', 0)),
            daemon=True
        )
        self.inference_thread.start()
        print(f"DEBUG: Inference thread started (Mode: {self.input_mode}).")
        self.status_message = f"Starting {self.input_mode}..." # Update status

    def _update_edge_params(self):
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
        self.edge_params_ref["enable_sharpening"] = self.enable_sharpening
        self.edge_params_ref["sharpness"] = self.sharpness

    def update_playback_state(self):
        """Updates the playback state dictionary passed to the inference thread."""
        self.playback_state["is_playing"] = self.is_playing
        self.playback_state["speed"] = self.playback_speed
        self.playback_state["loop"] = self.loop_video
        # Restart flag is set here, consumed by thread
        # self.playback_state["restart"] = False # Resetting here might cause issues if update is called before thread consumes it
        # self.playback_state_ref["restart"] = False # Don't reset restart flag here
        # Total frames and current frame are updated by the thread

    def update_recording_state(self):
        """Updates the recording state dictionary passed to the inference thread."""
        self.recording_state["is_recording"] = self.is_recording
        self.recording_state["output_dir"] = self.recording_output_dir


    def _process_queue_data(self, latest_data):
        """Unpacks data from queue and updates state, debug images, and timing deltas."""
        try:
            # Unpack new data format including timing info
            vertices_data, colors_data, num_vertices_actual, \
            rgb_frame_np, depth_map_tensor, edge_map_viz_np, \
            smoothing_map_viz_np, t_capture, \
            current_frame_idx, total_frames, \
            recorded_count, \
            frame_read_delta_t, depth_process_delta_t, latency_ms = latest_data # Added timing unpack

            # Update state variables
            self.last_capture_timestamp = t_capture
            # Update playback state directly in the shared dict for UI consistency
            self.playback_state["current_frame"] = current_frame_idx
            self.playback_state["total_frames"] = total_frames
            # Update recording state directly in the shared dict
            self.recording_state["frames_saved"] = recorded_count

            # Store latest timing deltas and latency
            self.frame_read_delta_t = frame_read_delta_t
            self.depth_process_delta_t = depth_process_delta_t
            self.latency_ms = latency_ms

            # Update latest debug image arrays
            self.latest_rgb_frame = rgb_frame_np
            self.latest_edge_map = edge_map_viz_np
            self.latest_smoothing_map = smoothing_map_viz_np

            # Process depth map for visualization
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

            # Return processed data needed for vertex list update
            return vertices_data, colors_data, num_vertices_actual

        except Exception as e_unpack:
            print(f"Error unpacking or processing data: {e_unpack}")
            traceback.print_exc()
            # Also reset timing deltas on error? Maybe not necessary.
            return None, None, 0 # Return default values on error

    def _update_debug_textures(self):
        """Updates OpenGL textures for debug views from latest image arrays."""
        if not self.debug_textures_initialized:
            return

        try:
            # Update Camera Feed Texture
            if self.latest_rgb_frame is not None and self.camera_texture is not None:
                h, w, _ = self.latest_rgb_frame.shape
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.latest_rgb_frame.ctypes.data)

            # Update Depth Map Texture
            if self.latest_depth_map_viz is not None and self.depth_texture is not None:
                h, w, _ = self.latest_depth_map_viz.shape
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self.latest_depth_map_viz.ctypes.data) # OpenCV uses BGR

            # Update Edge Map Texture
            if self.latest_edge_map is not None and self.edge_texture is not None:
                h, w, _ = self.latest_edge_map.shape
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.edge_texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.latest_edge_map.ctypes.data) # Edge map viz is RGB

            # Update Smoothing Map Texture
            if self.latest_smoothing_map is not None and self.smoothing_texture is not None:
                h, w, _ = self.latest_smoothing_map.shape
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.smoothing_texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.latest_smoothing_map.ctypes.data) # Smoothing map viz is RGB

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # Unbind

        except Exception as e_tex:
            print(f"Error updating debug textures: {e_tex}")
            # Optionally disable debug views on error?

    def _update_vertex_list(self, vertices_data, colors_data, num_vertices):
        """Updates the main point cloud vertex list."""
        self.current_point_count = num_vertices # Update point count regardless

        if vertices_data is not None and colors_data is not None and num_vertices > 0:
            # Y-inversion is done before this method in _process_queue_data
            # No further transformation needed here for display coordinates
            vertices_for_display = vertices_data

            # Delete existing list if it exists
            if self.vertex_list:
                try: self.vertex_list.delete()
                except Exception: pass # Ignore error if already deleted
                self.vertex_list = None

            # Create new vertex list
            try:
                self.vertex_list = self.shader_program.vertex_list(
                    num_vertices,
                    gl.GL_POINTS,
                    vertices=('f', vertices_for_display), # Data should already be Y-flipped for display
                    colors=('f', colors_data)
                )
                self.frame_count_display += 1 # Increment display counter on successful update
            except Exception as e_create:
                 print(f"Error creating vertex list: {e_create}")
                 traceback.print_exc()
                 self.vertex_list = None # Ensure list is None on error
                 self.current_point_count = 0
        else:
            # If no valid vertices, clear the list to avoid drawing stale points
            if self.vertex_list:
                try: self.vertex_list.delete()
                except Exception: pass
                self.vertex_list = None
            self.current_point_count = 0
            # Still increment frame count even if no points? Maybe not.
            # self.frame_count_display += 1

    def update(self, dt):
        """Scheduled function to process data from the inference thread."""
        ema_alpha=0.1 # Define EMA alpha locally
        new_data_processed = False
        try:
            while True: # Process all available data in the queue
                latest_data = self._data_queue.get_nowait()
                new_data_processed = True # Mark that we processed something

                if isinstance(latest_data, tuple) and isinstance(latest_data[0], str):
                    # Handle status/error/warning messages
                    if latest_data[0] == "status": self.status_message = latest_data[1]
                    elif latest_data[0] == "error": self.status_message = f"ERROR: {latest_data[1]}"; print(f"ERROR: {latest_data[1]}")
                    elif latest_data[0] == "warning": self.status_message = f"WARN: {latest_data[1]}"; print(f"WARN: {latest_data[1]}")
                    continue # Process next item in queue
                else:
                    # Process actual vertex and image data
                    vertices_data, colors_data, num_vertices = self._process_queue_data(latest_data)

                    # Update debug textures based on the processed data
                    self._update_debug_textures()

                    # Update the main vertex list
                    self._update_vertex_list(vertices_data, colors_data, num_vertices)

        except queue.Empty:
            pass # No more data in the queue

        # Calculate smoothed FPS values and update overlay labels if new data was processed
        if new_data_processed:
            # --- Input FPS (from frame read delta) ---
            if hasattr(self, 'frame_read_delta_t') and self.frame_read_delta_t > 1e-6:
                current_input_fps = 1.0 / self.frame_read_delta_t
                # Apply EMA smoothing
                self.input_fps = ema_alpha * current_input_fps + (1.0 - ema_alpha) * self.input_fps
            # --- Depth FPS (from depth processing delta) ---
            if hasattr(self, 'depth_process_delta_t') and self.depth_process_delta_t > 1e-6:
                current_depth_fps = 1.0 / self.depth_process_delta_t
                # Apply EMA smoothing
                self.depth_fps = ema_alpha * current_depth_fps + (1.0 - ema_alpha) * self.depth_fps
            # --- Render FPS (calculated based on main thread update rate) ---
            current_time = time.time()
            time_delta = current_time - self.last_update_time
            if time_delta > 1e-6:
                 # Apply EMA smoothing to render FPS as well
                 current_render_fps = 1.0 / time_delta
                 self.point_cloud_fps = ema_alpha * current_render_fps + (1.0 - ema_alpha) * self.point_cloud_fps
            self.last_update_time = current_time

            # --- Update Overlay Label Text & Visibility ---
            if hasattr(self, 'fps_label'):
                self.fps_label.visible = self.show_fps_overlay
                if self.show_fps_overlay: self.fps_label.text = f"Render FPS: {self.point_cloud_fps:.1f}"
            if hasattr(self, 'points_label'):
                self.points_label.visible = self.show_points_overlay
                if self.show_points_overlay: self.points_label.text = f"Points: {self.current_point_count}"
            if hasattr(self, 'input_fps_label'):
                self.input_fps_label.visible = self.show_input_fps_overlay
                if self.show_input_fps_overlay: self.input_fps_label.text = f"Input FPS: {self.input_fps:.1f}"
            if hasattr(self, 'depth_fps_label'):
                self.depth_fps_label.visible = self.show_depth_fps_overlay
                if self.show_depth_fps_overlay: self.depth_fps_label.text = f"Depth FPS: {self.depth_fps:.1f}"
            if hasattr(self, 'latency_label') and hasattr(self, 'latency_ms'):
                self.latency_label.visible = self.show_latency_overlay
                if self.show_latency_overlay: self.latency_label.text = f"Latency: {self.latency_ms:.1f} ms"
    def update_camera(self, dt):
        """Scheduled function to handle camera movement based on key states."""
        io = imgui.get_io()
        if io.want_capture_keyboard or io.want_capture_mouse: return # Check mouse capture too

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
        # Update state dictionaries
        self._update_edge_params()
        # Update playback state dictionary
        self.update_playback_state()
        # Update recording state dictionary
        self.update_recording_state()


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
                    default_val = DEFAULT_SETTINGS[key]
                    loaded_val = loaded_settings.get(key, default_val)
                    try:
                        if isinstance(default_val, bool): setattr(self, key, bool(loaded_val))
                        elif isinstance(default_val, int): setattr(self, key, int(loaded_val))
                        elif isinstance(default_val, float): setattr(self, key, float(loaded_val))
                        else: setattr(self, key, loaded_val)
                    except (ValueError, TypeError):
                         print(f"Warning: Could not convert loaded setting '{key}' ({loaded_val}), using default.")
                         setattr(self, key, default_val)
                print(f"DEBUG: Settings loaded from {filename}")
            else:
                print(f"DEBUG: Settings file {filename} not found, using defaults.")
                # Defaults are already set
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            # Ensure defaults are set if loading fails
            for key, value in DEFAULT_SETTINGS.items():
                setattr(self, key, value)

        # Ensure reference dicts are updated/created *after* loading/defaults
        self.scale_factor_ref = [self.input_scale_factor]
        self._update_edge_params()
        self.update_playback_state()
        self.update_recording_state() # Initialize recording state


    def _browse_file(self, browse_type="file"): # Add browse_type argument
        """Opens a file or directory dialog based on browse_type."""
        root = tk.Tk()
        root.withdraw() # Hide the main tkinter window
        file_path = None # Initialize file_path

        if browse_type == "directory":
            file_path = filedialog.askdirectory(title="Select GLB Sequence Directory")
            if file_path:
                print(f"DEBUG: Directory selected: {file_path}")
                self.input_filepath = file_path
                # self.input_mode = "GLB Sequence" # Mode is already set by UI combo
                self.status_message = f"Directory selected: {os.path.basename(file_path)}"
            else:
                print("DEBUG: Directory selection cancelled.")
        elif browse_type == "file":
            file_path = filedialog.askopenfilename(
                title="Select Video or Image File",
                filetypes=[("Media Files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                           ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                           ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                           ("All Files", "*.*")]
            )
            if file_path:
                print(f"DEBUG: File selected: {file_path}")
                self.input_filepath = file_path
                # self.input_mode = "File (Video/Image)" # Mode is already set by UI combo
                self.status_message = f"File selected: {os.path.basename(file_path)}"
            else:
                print("DEBUG: File selection cancelled.")
        else:
             print(f"Warning: Unknown browse_type '{browse_type}'")

        root.destroy() # Destroy the tkinter instance

        if file_path: # Only reset playback if a path was actually selected
            # Reset playback state for new file/directory
            self.is_playing = True # Reset local UI control state
            # Reset relevant fields in the shared state dictionary
            self.playback_state["current_frame"] = 0
            self.playback_state["total_frames"] = 0
            self.playback_state["restart"] = False # Ensure restart flag is clear
            # Update the rest of the shared state from local attributes
            self.update_playback_state()
            # DO NOT Restart inference thread here - user clicks "Start Source"
            # self.start_inference_thread()
         # --- Removed old logic below ---




    def _draw_ui(self):
        """Draws the ImGui user interface."""
        # UI drawing logic will be moved here
        # --- ImGui Frame ---
        imgui.new_frame()

        # --- Main Controls Window ---
        imgui.set_next_window_position(10, 10, imgui.ONCE)
        imgui.set_next_window_size(350, self.height - 20, imgui.ONCE) # Adjust height dynamically
        imgui.begin("UniK3D Controls")

        # --- Status Display (Always Visible) ---
        status_text = "Status: Idle"
        status_color = (0.5, 0.5, 0.5, 1.0) # Gray
        thread_is_running = self.inference_thread and self.inference_thread.is_alive()

        if thread_is_running:
            if self.input_mode == "Live":
                status_text = "Status: Live Feed Active"
                status_color = (0.1, 1.0, 0.1, 1.0) # Green
            elif self.input_mode == "File":
                status_text = f"Status: Processing File ({os.path.basename(self.input_filepath)})"
                status_color = (0.1, 0.6, 1.0, 1.0) # Blue
            elif self.input_mode == "GLB Sequence":
                 status_text = f"Status: Playing GLB Sequence ({os.path.basename(self.input_filepath)})"
                 status_color = (1.0, 0.6, 0.1, 1.0) # Orange
        elif "Error" in self.status_message:
             status_text = f"Status: Error ({self.status_message})"
             status_color = (1.0, 0.1, 0.1, 1.0) # Red
        elif "Finished" in self.status_message:
             status_text = f"Status: Finished"
             status_color = (0.5, 0.5, 0.5, 1.0) # Gray
        else:
             status_text = f"Status: {self.status_message}" # Show initializing etc.

        imgui.text_colored(status_text, *status_color)
        imgui.separator()

        # --- Tab Bar ---
        if imgui.begin_tab_bar("MainTabs"):

            # --- Input/Output Tab ---
            if imgui.begin_tab_item("Input/Output")[0]:
                imgui.text("Input Source")

                # --- Collapsing Headers for Input Modes ---

                # --- Live Camera Section ---
                live_flags = imgui.TREE_NODE_DEFAULT_OPEN if self.input_mode == "Live Camera" else 0
                header_live_open, _ = imgui.collapsing_header("Live Camera", flags=live_flags)
                if header_live_open:
                    imgui.indent()
                    # Camera Index Input
                    changed_cam_idx, self.live_camera_index = imgui.input_int("Camera Index", self.live_camera_index if hasattr(self, 'live_camera_index') else 0, 1)
                    # Activate Button
                    is_active = self.input_mode == "Live Camera" and self.inference_thread and self.inference_thread.is_alive()
                    if imgui.button("Activate##Live" if not is_active else "Stop##Live"):
                        if is_active:
                            print("DEBUG: Stopping Live Camera source...")
                            self.start_inference_thread() # Stops current thread
                        else:
                            print("DEBUG: Activating Live Camera...")
                            self.input_mode = "Live Camera"
                            self.input_filepath = "" # Clear path
                            # TODO: Pass self.live_camera_index to start_inference_thread/init
                            self.start_inference_thread()
                    imgui.unindent()

                # --- File Section ---
                file_flags = imgui.TREE_NODE_DEFAULT_OPEN if self.input_mode == "File (Video/Image)" else 0
                header_file_open, _ = imgui.collapsing_header("File (Video/Image)", flags=file_flags)
                if header_file_open:
                    imgui.indent()
                    if imgui.button("Browse File...##File"):
                        self._browse_file(browse_type="file")
                    imgui.same_line()
                    imgui.text(f"Path: {self.input_filepath if self.input_filepath else 'None selected'}")
                    # Activate Button
                    activate_disabled = not self.input_filepath
                    is_active = self.input_mode == "File (Video/Image)" and self.inference_thread and self.inference_thread.is_alive()
                    if activate_disabled:
                        imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                        imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                    if imgui.button("Activate##File" if not is_active else "Stop##File"):
                         if is_active:
                            print("DEBUG: Stopping File source...")
                            self.start_inference_thread() # Stops current thread
                         elif not activate_disabled:
                            print("DEBUG: Activating File source...")
                            self.input_mode = "File (Video/Image)"
                            self.start_inference_thread()
                    if activate_disabled:
                        imgui.pop_style_var()
                        imgui.internal.pop_item_flag()
                    imgui.unindent()

                # --- GLB Sequence Section ---
                glb_flags = imgui.TREE_NODE_DEFAULT_OPEN if self.input_mode == "GLB Sequence" else 0
                header_glb_open, _ = imgui.collapsing_header("GLB Sequence", flags=glb_flags)
                if header_glb_open:
                    imgui.indent()
                    if imgui.button("Browse Directory...##GLB"):
                        self._browse_file(browse_type="directory")
                    imgui.same_line()
                    imgui.text(f"Path: {self.input_filepath if self.input_filepath else 'None selected'}")
                    # Activate Button
                    activate_disabled = not self.input_filepath
                    is_active = self.input_mode == "GLB Sequence" and self.inference_thread and self.inference_thread.is_alive()
                    if activate_disabled:
                        imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                        imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                    if imgui.button("Activate##GLB" if not is_active else "Stop##GLB"):
                         if is_active:
                            print("DEBUG: Stopping GLB Sequence source...")
                            self.start_inference_thread() # Stops current thread
                         elif not activate_disabled:
                            print("DEBUG: Activating GLB Sequence...")
                            self.input_mode = "GLB Sequence"
                            self.start_inference_thread()
                    if activate_disabled:
                        imgui.pop_style_var()
                        imgui.internal.pop_item_flag()
                    imgui.unindent()

                # --- Screen Capture Section ---
                screen_flags = imgui.TREE_NODE_DEFAULT_OPEN if self.input_mode == "Screen Capture" else 0
                header_screen_open, _ = imgui.collapsing_header("Screen Capture", flags=screen_flags)
                if header_screen_open:
                    imgui.indent()
                    # Monitor Selection
                    monitors = mss.mss().monitors
                    monitor_names = [f"Monitor {i}: {m['width']}x{m['height']} @ ({m['left']},{m['top']})" for i, m in enumerate(monitors)]
                    if not hasattr(self, 'selected_monitor_index'): self.selected_monitor_index = 1 # Default to primary
                    if self.selected_monitor_index >= len(monitors): self.selected_monitor_index = 1 # Ensure valid index
                    changed_monitor, self.selected_monitor_index = imgui.combo("Monitor", self.selected_monitor_index, monitor_names)

                    # Window Selection (Windows Only)
                    windows = _get_window_list()
                    window_names = ["Entire Selected Monitor"] + [f"{title} (HWND:{hwnd})" for hwnd, title in windows]
                    if not hasattr(self, 'selected_window_hwnd'): self.selected_window_hwnd = 0 # 0 means entire monitor
                    current_window_index = 0
                    if self.selected_window_hwnd != 0:
                        try: current_window_index = next(i for i, (hwnd, _) in enumerate(windows, 1) if hwnd == self.selected_window_hwnd)
                        except StopIteration: self.selected_window_hwnd = 0 # Reset if window not found
                    changed_window, current_window_index = imgui.combo("Target Window", current_window_index, window_names)
                    if changed_window: self.selected_window_hwnd = 0 if current_window_index == 0 else windows[current_window_index - 1][0]

                    # Activate Button
                    is_active = self.input_mode == "Screen Capture" and self.inference_thread and self.inference_thread.is_alive()
                    if imgui.button("Activate##Screen" if not is_active else "Stop##Screen"):
                         if is_active:
                            print("DEBUG: Stopping Screen Capture source...")
                            self.start_inference_thread() # Stops current thread
                         else:
                            print("DEBUG: Activating Screen Capture...")
                            self.input_mode = "Screen Capture"
                            self.input_filepath = "" # Clear path
                            # TODO: Pass self.selected_monitor_index and self.selected_window_hwnd
                            self.start_inference_thread()
                    imgui.unindent()

                # --- Old Start/Stop Button Removed ---

                # --- Live Processing Mode (Only show if Live Camera is selected) ---
                # (Keep existing logic for this, but maybe move it under the conditional display)
                if self.input_mode == "Live Camera": # Use self.input_mode
                    imgui.separator()
                    imgui.text("Live Processing Mode:")
                    mode_changed = False
                    if imgui.radio_button("Real-time (Low Latency)", self.live_processing_mode == "Real-time"):
                        if self.live_processing_mode != "Real-time":
                            self.live_processing_mode = "Real-time"
                            mode_changed = True
                    imgui.same_line()
                    if imgui.radio_button("Buffered (Process All Frames)", self.live_processing_mode == "Buffered"):
                         if self.live_processing_mode != "Buffered":
                            self.live_processing_mode = "Buffered"
                            mode_changed = True
                    if mode_changed:
                        # Restart thread with new mode if it's currently running
                        if self.inference_thread and self.inference_thread.is_alive():
                            print(f"Switching live processing mode to {self.live_processing_mode}, restarting thread...")
                            self.start_inference_thread()
                    # Duplicate radio button logic removed here

                imgui.separator()
                imgui.text("Recording") # Keep recording section
                rec_button_text = " Stop Recording " if self.is_recording else " Start Recording "
                if imgui.button(rec_button_text):
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        try:
                            os.makedirs(self.recording_output_dir, exist_ok=True)
                            self.recorded_frame_count = 0 # Reset counter
                            self.status_message = f"Recording started to {self.recording_output_dir}"
                        except Exception as e_dir:
                            print(f"Error creating recording directory: {e_dir}")
                            self.status_message = "Error creating recording dir!"
                            self.is_recording = False # Abort recording
                    else:
                        self.status_message = f"Recording stopped. {self.recorded_frame_count} frames saved."
                    self.update_recording_state() # Update thread state

                imgui.same_line()
                # Read recorded frame count from recording_state dict (updated by thread)
                saved_count = self.recording_state.get("frames_saved", 0)
                imgui.text(f"({saved_count} frames saved)")

                changed_dir, self.recording_output_dir = imgui.input_text(
                    "Output Dir", self.recording_output_dir, 256
                )
                if changed_dir and not self.recording_state.get("is_recording", False): # Update ref only if not recording
                     self.update_recording_state()

                # Model Selection moved here earlier

                # Live Processing Mode (Only show if Live Camera is selected)
                if self.input_mode == "Live":
                    imgui.separator()
                    imgui.text("Live Processing Mode:")
                    mode_changed = False
                    if imgui.radio_button("Real-time (Low Latency)", self.live_processing_mode == "Real-time"):
                        if self.live_processing_mode != "Real-time":
                            self.live_processing_mode = "Real-time"
                            mode_changed = True
                    imgui.same_line()
                    if imgui.radio_button("Buffered (Process All Frames)", self.live_processing_mode == "Buffered"):
                         if self.live_processing_mode != "Buffered":
                            self.live_processing_mode = "Buffered"
                            mode_changed = True
                    if mode_changed:
                        # Restart thread with new mode if it's currently running
                        if self.inference_thread and self.inference_thread.is_alive():
                            print(f"Switching live processing mode to {self.live_processing_mode}, restarting thread...")
                            self.start_inference_thread()
                    if imgui.radio_button("Buffered (Process All Frames)", self.live_processing_mode == "Buffered"):
                         if self.live_processing_mode != "Buffered":
                            self.live_processing_mode = "Buffered"
                            mode_changed = True
                    if mode_changed:
                        # Restart thread with new mode if it's currently running
                        if self.inference_thread and self.inference_thread.is_alive():
                            print(f"Switching live processing mode to {self.live_processing_mode}, restarting thread...")
                            self.start_inference_thread()

                imgui.separator()
                imgui.text("Recording")
                rec_button_text = " Stop Recording " if self.is_recording else " Start Recording "
                if imgui.button(rec_button_text):
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        try:
                            os.makedirs(self.recording_output_dir, exist_ok=True)
                            self.recorded_frame_count = 0 # Reset counter
                            self.status_message = f"Recording started to {self.recording_output_dir}"
                        except Exception as e_dir:
                            print(f"Error creating recording directory: {e_dir}")
                            self.status_message = "Error creating recording dir!"
                            self.is_recording = False # Abort recording
                    else:
                        self.status_message = f"Recording stopped. {self.recorded_frame_count} frames saved."
                    self.update_recording_state() # Update thread state

                imgui.same_line()
                # Read recorded frame count from recording_state dict (updated by thread)
                saved_count = self.recording_state.get("frames_saved", 0)
                imgui.text(f"({saved_count} frames saved)")

                changed_dir, self.recording_output_dir = imgui.input_text(
                    "Output Dir", self.recording_output_dir, 256
                )
                if changed_dir and not self.recording_state.get("is_recording", False): # Update ref only if not recording
                     self.update_recording_state()

                imgui.separator()
                imgui.text("Model Selection")
                # --- Model Selection Combo ---
                model_names = ["unik3d-vits", "unik3d-vitb", "unik3d-vitl"]
                try:
                    current_model_index = model_names.index(self.selected_model_name)
                except ValueError:
                    current_model_index = model_names.index(self._model_name) # Fallback to initial if current selection invalid
                    self.selected_model_name = self._model_name

                changed_model, current_model_index = imgui.combo(
                    "Model", current_model_index, model_names
                )
                if changed_model:
                    self.selected_model_name = model_names[current_model_index]

                imgui.same_line()
                if imgui.button("Apply Model"):
                    if self.selected_model_name != self._model_name:
                        print(f"Switching model to {self.selected_model_name}...")
                        self._model_name = self.selected_model_name # Update the name used by thread start
                        self.start_inference_thread() # Restart thread with new model
                    else:
                        print("DEBUG: Selected model is already active.")
                # --- End Model Selection ---

                imgui.end_tab_item()
            # --- End Input/Output Tab ---

            # --- Playback Tab (Conditional) ---
            show_playback_tab = self.input_mode != "Live" and self.playback_state.get("total_frames", 0) > 0
            if show_playback_tab:
                if imgui.begin_tab_item("Playback")[0]:
                    play_button_text = " Pause " if self.is_playing else " Play  "
                    if imgui.button(play_button_text): # Button text based on self.is_playing
                        self.is_playing = not self.is_playing # Toggle local UI state variable
                        self.update_playback_state() # Update the shared state dict
                    imgui.same_line()
                    if imgui.button("Restart"): # Button triggers restart flag
                        self.playback_state["restart"] = True # Signal thread to restart via shared dict
                        self.is_playing = True # Assume play after restart for UI button state
                        self.update_playback_state() # Update shared dict (including is_playing)
                    imgui.same_line()
                    # Use loop_video attribute for checkbox state, update shared dict on change
                    loop_changed, self.loop_video = imgui.checkbox("Loop", self.loop_video)
                    if loop_changed: self.update_playback_state()

                    # Use playback_speed attribute for slider state, update shared dict on change
                    speed_changed, self.playback_speed = imgui.slider_float("Speed", self.playback_speed, 0.1, 4.0, "%.1fx")
                    if speed_changed: self.update_playback_state()

                    # Read current/total frames directly from playback_state dict (updated by thread)
                    current_f = self.playback_state.get("current_frame", 0)
                    total_f = self.playback_state.get("total_frames", 0)
                    imgui.text(f"Frame: {current_f} / {total_f}")
                    progress = float(current_f) / total_f if total_f > 0 else 0.0
                    imgui.progress_bar(progress, (-1, 0), f"{current_f}/{total_f}")

                    imgui.end_tab_item()
            # --- End Playback Tab ---

            # --- Processing Tab ---
            if imgui.begin_tab_item("Processing")[0]:
                # Smoothing Section
                imgui.text("Temporal Smoothing")
                changed_smooth, self.enable_point_smoothing = imgui.checkbox("Enable##SmoothEnable", self.enable_point_smoothing)
                if changed_smooth: self._update_edge_params()

                imgui.indent()
                changed_min_alpha, self.min_alpha_points = imgui.slider_float("Min Alpha", self.min_alpha_points, 0.0, 1.0)
                if changed_min_alpha: self._update_edge_params()
                changed_max_alpha, self.max_alpha_points = imgui.slider_float("Max Alpha", self.max_alpha_points, 0.0, 1.0)
                if changed_max_alpha: self._update_edge_params()
                if imgui.button("Reset##SmoothAlpha"):
                    self.min_alpha_points = DEFAULT_SETTINGS["min_alpha_points"]
                    self.max_alpha_points = DEFAULT_SETTINGS["max_alpha_points"]
                    self.update_edge_params_ref()
                imgui.unindent()

                imgui.separator()
                imgui.text("Edge-Aware Smoothing")
                changed_edge_aware, self.enable_edge_aware_smoothing = imgui.checkbox("Enable##EdgeAware", self.enable_edge_aware_smoothing)
                if changed_edge_aware: self._update_edge_params()
                imgui.indent()
                changed_edge_influence, self.edge_smoothing_influence = imgui.slider_float("Edge Influence", self.edge_smoothing_influence, 0.0, 1.0)
                if changed_edge_influence: self._update_edge_params()
                changed_grad_influence, self.gradient_influence_scale = imgui.slider_float("Gradient Scale", self.gradient_influence_scale, 0.0, 5.0)
                if changed_grad_influence: self._update_edge_params()
                if imgui.button("Reset##EdgeAwareParams"):
                    self.edge_smoothing_influence = DEFAULT_SETTINGS["edge_smoothing_influence"]
                    self.gradient_influence_scale = DEFAULT_SETTINGS["gradient_influence_scale"]
                    self._update_edge_params()
                imgui.unindent()

                imgui.separator()
                imgui.text("Input Preprocessing")
                changed_sharpen, self.enable_sharpening = imgui.checkbox("Sharpen Input", self.enable_sharpening)
                if changed_sharpen: self._update_edge_params()
                imgui.indent()
                changed_sharp_amount, self.sharpness = imgui.slider_float("Sharpness Amount", self.sharpness, 0.0, 3.0)
                if changed_sharp_amount: self._update_edge_params()
                imgui.unindent()

                changed_scale, self.input_scale_factor = imgui.slider_float("Input Scale", self.input_scale_factor, 0.1, 1.0, "%.2f")
                if changed_scale:
                    self.scale_factor_ref[0] = self.input_scale_factor # Update ref immediately

                # --- Model Selection Moved to Input/Output Tab ---

                imgui.separator()
                changed_edge_smooth, self.enable_edge_aware_smoothing = imgui.checkbox("Enable##EdgeAwareEnable", self.enable_edge_aware_smoothing)
                if changed_edge_smooth: self._update_edge_params()

                if self.enable_edge_aware_smoothing:
                    imgui.indent()
                    imgui.columns(2, "edge_thresholds", border=False)
                    changed_d_thresh1, self.depth_edge_threshold1 = imgui.slider_float("Depth Thresh 1", self.depth_edge_threshold1, 1.0, 255.0)
                    if changed_d_thresh1: self._update_edge_params()
                    changed_d_thresh2, self.depth_edge_threshold2 = imgui.slider_float("Depth Thresh 2", self.depth_edge_threshold2, 1.0, 255.0)
                    if changed_d_thresh2: self._update_edge_params()
                    imgui.next_column()
                    changed_rgb_thresh1, self.rgb_edge_threshold1 = imgui.slider_float("RGB Thresh 1", self.rgb_edge_threshold1, 1.0, 255.0)
                    if changed_rgb_thresh1: self._update_edge_params()
                    changed_rgb_thresh2, self.rgb_edge_threshold2 = imgui.slider_float("RGB Thresh 2", self.rgb_edge_threshold2, 1.0, 255.0)
                    if changed_rgb_thresh2: self._update_edge_params()
                    imgui.columns(1)

                    changed_edge_inf, self.edge_smoothing_influence = imgui.slider_float("Edge Influence", self.edge_smoothing_influence, 0.0, 1.0)
                    if changed_edge_inf: self._update_edge_params()

                    changed_grad_inf, self.gradient_influence_scale = imgui.slider_float("Gradient Scale", self.gradient_influence_scale, 0.0, 5.0)
                    if changed_grad_inf: self._update_edge_params()

                    if imgui.button("Reset##EdgeParams"):
                        self.depth_edge_threshold1 = DEFAULT_SETTINGS["depth_edge_threshold1"]
                        self.depth_edge_threshold2 = DEFAULT_SETTINGS["depth_edge_threshold2"]
                        self.rgb_edge_threshold1 = DEFAULT_SETTINGS["rgb_edge_threshold1"]
                        self.rgb_edge_threshold2 = DEFAULT_SETTINGS["rgb_edge_threshold2"]
                        self.edge_smoothing_influence = DEFAULT_SETTINGS["edge_smoothing_influence"]
                        self.gradient_influence_scale = DEFAULT_SETTINGS["gradient_influence_scale"]
                        self._update_edge_params()
                    imgui.unindent()

                imgui.separator()
                imgui.text("Image Sharpening")
                changed_sharp_enable, self.enable_sharpening = imgui.checkbox("Enable##SharpEnable", self.enable_sharpening)
                if changed_sharp_enable: self._update_edge_params()
                if self.enable_sharpening:
                    imgui.indent()
                    changed_sharp_amt, self.sharpness = imgui.slider_float("Amount", self.sharpness, 0.1, 5.0)
                    if changed_sharp_amt: self._update_edge_params()
                    if imgui.button("Reset##Sharpness"):
                        self.sharpness = DEFAULT_SETTINGS["sharpness"]
                        self._update_edge_params()
                    imgui.unindent()

                imgui.end_tab_item()
            # --- End Processing Tab ---

            # --- Rendering Tab ---
            if imgui.begin_tab_item("Rendering")[0]:
                imgui.text("Point Style")
                if imgui.radio_button("Square##RenderMode", self.render_mode == 0): self.render_mode = 0
                imgui.same_line()
                if imgui.radio_button("Circle##RenderMode", self.render_mode == 1): self.render_mode = 1
                imgui.same_line()
                if imgui.radio_button("Gaussian##RenderMode", self.render_mode == 2): self.render_mode = 2

                if self.render_mode == 2: # Gaussian Params
                    imgui.indent()
                    changed_falloff, self.falloff_factor = imgui.slider_float("Falloff", self.falloff_factor, 0.1, 20.0)
                    if imgui.button("Reset##Falloff"): self.falloff_factor = DEFAULT_SETTINGS["falloff_factor"]
                    imgui.unindent()

                imgui.separator()
                imgui.text("Point Size & Scale")
                changed_boost, self.point_size_boost = imgui.slider_float("Size Boost", self.point_size_boost, 0.1, 10.0)
                if imgui.button("Reset##PointSize"): self.point_size_boost = DEFAULT_SETTINGS["point_size_boost"]

                changed_scale, self.input_scale_factor = imgui.slider_float("Input Scale", self.input_scale_factor, 0.1, 1.0)
                if changed_scale: self.scale_factor_ref[0] = self.input_scale_factor # Update shared ref for thread
                if imgui.button("Reset##InputScale"):
                    self.input_scale_factor = DEFAULT_SETTINGS["input_scale_factor"]
                    self.scale_factor_ref[0] = self.input_scale_factor

                imgui.separator()
                imgui.text("Color Adjustments")
                changed_sat, self.saturation = imgui.slider_float("Saturation", self.saturation, 0.0, 3.0)
                if imgui.button("Reset##Saturation"): self.saturation = DEFAULT_SETTINGS["saturation"]

                changed_brt, self.brightness = imgui.slider_float("Brightness", self.brightness, 0.0, 2.0)
                if imgui.button("Reset##Brightness"): self.brightness = DEFAULT_SETTINGS["brightness"]

                changed_con, self.contrast = imgui.slider_float("Contrast", self.contrast, 0.1, 3.0)
                if imgui.button("Reset##Contrast"): self.contrast = DEFAULT_SETTINGS["contrast"]

                imgui.end_tab_item()
            # --- End Rendering Tab ---

            # --- Debug Tab ---
            if imgui.begin_tab_item("Debug")[0]:
                imgui.text("Show Debug Views:")
                _, self.show_camera_feed = imgui.checkbox("Camera Feed", self.show_camera_feed)
                _, self.show_depth_map = imgui.checkbox("Depth Map", self.show_depth_map)
                _, self.show_edge_map = imgui.checkbox("Edge Map", self.show_edge_map)
                _, self.show_smoothing_map = imgui.checkbox("Smoothing Map", self.show_smoothing_map)

                imgui.separator()
                imgui.text("Performance Info:")
                # Display all stats here
                imgui.text(f"Points Rendered: {self.current_point_count}")
                imgui.text(f"Render FPS: {self.point_cloud_fps:.1f}")
                imgui.text(f"Input FPS: {self.input_fps:.1f}")
                imgui.text(f"Depth FPS: {self.depth_fps:.1f}")
                imgui.text(f"Latency: {self.latency_ms:.1f} ms")

                imgui.separator()
                imgui.text("Show Stats in Overlay:")
                # Individual toggles for overlay stats
                _, self.show_fps_overlay = imgui.checkbox("Render FPS##Overlay", self.show_fps_overlay)
                _, self.show_points_overlay = imgui.checkbox("Points##Overlay", self.show_points_overlay)
                _, self.show_input_fps_overlay = imgui.checkbox("Input FPS##Overlay", self.show_input_fps_overlay)
                _, self.show_depth_fps_overlay = imgui.checkbox("Depth FPS##Overlay", self.show_depth_fps_overlay)
                _, self.show_latency_overlay = imgui.checkbox("Latency##Overlay", self.show_latency_overlay)
                # Add more debug info if needed

                imgui.end_tab_item()
            # --- End Debug Tab ---

            # --- Settings Tab ---
            if imgui.begin_tab_item("Settings")[0]:
                if imgui.button("Load Settings"): self.load_settings()
                imgui.same_line()
                if imgui.button("Save Settings"): self.save_settings()
                imgui.same_line()
                if imgui.button("Reset All Defaults"): self.reset_settings()
                imgui.text(f"Settings File: viewer_settings.json")
                # Could add more settings-related info here if needed
                imgui.end_tab_item()
            # --- End Settings Tab ---

            imgui.end_tab_bar()
        # --- End Tab Bar ---

        imgui.end() # End main controls window

        # --- Debug View Windows (Rendered as separate windows now) ---
        if self.show_camera_feed and self.camera_texture and self.latest_rgb_frame is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(10, self.height - 250, imgui.ONCE) # Position top-leftish
            is_open, self.show_camera_feed = imgui.begin("Camera Feed", closable=True)
            if is_open:
                imgui.image(self.camera_texture, self.latest_rgb_frame.shape[1], self.latest_rgb_frame.shape[0])
            imgui.end()

        if self.show_depth_map and self.depth_texture and self.latest_depth_map_viz is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(340, self.height - 250, imgui.ONCE) # Position next to camera feed
            is_open, self.show_depth_map = imgui.begin("Depth Map", closable=True)
            if is_open:
                imgui.image(self.depth_texture, self.latest_depth_map_viz.shape[1], self.latest_depth_map_viz.shape[0])
            imgui.end()

        if self.show_edge_map and self.edge_texture and self.latest_edge_map is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(10, self.height - 500, imgui.ONCE) # Position below camera feed
            is_open, self.show_edge_map = imgui.begin("Edge Map", closable=True)
            if is_open:
                imgui.image(self.edge_texture, self.latest_edge_map.shape[1], self.latest_edge_map.shape[0])
            imgui.end()

        if self.show_smoothing_map and self.smoothing_texture and self.latest_smoothing_map is not None:
             imgui.set_next_window_size(320, 240, imgui.ONCE)
             imgui.set_next_window_position(340, self.height - 500, imgui.ONCE) # Position below depth map
             is_open, self.show_smoothing_map = imgui.begin("Smoothing Alpha Map", closable=True)
             if is_open:
                 imgui.image(self.smoothing_texture, self.latest_smoothing_map.shape[1], self.latest_smoothing_map.shape[0])
             imgui.end()
        # --- End Debug View Windows ---


        # Render ImGui
        # Ensure correct GL state for ImGui rendering (standard alpha blend)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST) # ImGui draws in 2D

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())
        # --- End ImGui Frame ---

    def _render_scene(self):
        """Renders the 3D point cloud scene."""
        # 3D rendering logic will be moved here
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
            # Sharpness is applied in inference thread, but shader still has uniform
            self.shader_program['sharpness'] = self.sharpness if self.enable_sharpening else 1.0

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

    def on_draw(self):
        # Clear the main window buffer
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Render the 3D scene
        self._render_scene()
        # --- Draw Stats Overlay (Individual Labels) ---
        # Set GL state for drawing text overlay
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST) # Draw text on top

        # Draw labels individually based on toggles
        # Draw batches individually based on toggles
        if self.show_fps_overlay and hasattr(self, 'fps_batch'): self.fps_batch.draw()
        if self.show_points_overlay and hasattr(self, 'points_batch'): self.points_batch.draw()
        if self.show_input_fps_overlay and hasattr(self, 'input_fps_batch'): self.input_fps_batch.draw()
        if self.show_depth_fps_overlay and hasattr(self, 'depth_fps_batch'): self.depth_fps_batch.draw()
        if self.show_latency_overlay and hasattr(self, 'latency_batch'): self.latency_batch.draw()

        # Restore depth test before drawing ImGui (which might disable it again)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # --- End Stats Overlay ---

        # Draw the ImGui UI using the helper method
        self._draw_ui()

        # Final GL state restoration (if needed, though ImGui usually handles its own)
        # gl.glEnable(gl.GL_DEPTH_TEST) # Already enabled after overlay


    # --- Input Handlers ---
    def on_resize(self, width, height):
        gl.glViewport(0, 0, max(1, width), max(1, height))
        self._aspect_ratio = float(width) / height if height > 0 else 1.0 # Update renamed variable
        # Update overlay label positions on resize
        y_pos = height - 20
        if hasattr(self, 'fps_label'):
            self.fps_label.x = width - 10; self.fps_label.y = y_pos; y_pos -= 20
        if hasattr(self, 'points_label'):
            self.points_label.x = width - 10; self.points_label.y = y_pos; y_pos -= 20
        if hasattr(self, 'input_fps_label'):
            self.input_fps_label.x = width - 10; self.input_fps_label.y = y_pos; y_pos -= 20
        if hasattr(self, 'depth_fps_label'):
            self.depth_fps_label.x = width - 10; self.depth_fps_label.y = y_pos; y_pos -= 20
        if hasattr(self, 'latency_label'):
            self.latency_label.x = width - 10; self.latency_label.y = y_pos; y_pos -= 20
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
            self.inference_thread.join(timeout=2.0) # Increased timeout
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