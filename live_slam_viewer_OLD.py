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
import tkinter as tk
from tkinter import filedialog
import glob # For finding files
import re # For sorting files numerically
import trimesh # For GLB saving/loading
import mss  # For screen capture (virtual camera)
from collections import deque
try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    print("ERROR: pytorch_wavelets not installed; install via 'pip install pytorch-wavelets'")

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
    "falloff_factor": 1.0,
    "saturation": 1.0,
    "brightness": 1.0,
    "contrast": 1.0,
    "sharpness": 1.0,
    "enable_sharpening": False,
    "point_size_boost": 1.0,
    "input_scale_factor": 1.0,
    "size_scale_factor": 0.001, # Scale factor for depth sizing (default inverse-square)
    "enable_point_smoothing": False,
    "min_alpha_points": 0.0,
    "max_alpha_points": 1.0,
    "enable_edge_aware_smoothing": False,
    "depth_edge_threshold1": 50.0,
    "depth_edge_threshold2": 150.0,
    "rgb_edge_threshold1": 50.0,
    "rgb_edge_threshold2": 150.0,
    "edge_smoothing_influence": 0.0,
    "gradient_influence_scale": 1.0,
    "playback_speed": 1.0, # For video/GLB sequence files
    "loop_video": True, # For video/GLB sequence files
    "is_recording": False, # Recording state
    "recording_output_dir": "recording_output", # Default output dir
    "show_camera_feed": False,
    "show_depth_map": False,
    "show_edge_map": False,
    "show_smoothing_map": False,
    "show_wavelet_map": False,
    # Overlay Toggles
    "show_fps_overlay": False,
    "show_points_overlay": False,
    "show_input_fps_overlay": False,
    "show_depth_fps_overlay": False,
    "show_latency_overlay": False,
    "live_processing_mode": "Real-time", # "Real-time" or "Buffered"
    "input_camera_fov": 60.0,  # FOV of the input camera in degrees
    "min_point_size": 1.0,        # Min pixel size
    "enable_max_size_clamp": False, # Enable max size clamp?
    "max_point_size": 50.0,       # Max pixel size (if clamped)
    # --- Point Thickening --- 
    "enable_point_thickening": False,
    "thickening_duplicates": 0,     # Num duplicates per point (total points = original * (1 + duplicates))
    "thickening_variance": 0.0,   # StdDev of random perturbation
    "thickening_depth_bias": 0.0,   # Strength of backward push along ray (0=none)
    "depth_exponent": 2.0,        # Exponent for depth-based sizing (direct square)
    "screen_capture_monitor_index": 0, # 0 for entire desktop, 1+ for specific monitors
    # --- Visual Debugging --- (Additions)
    "debug_show_input_distance": False,
    "debug_show_raw_diameter": False,
    "debug_show_density_factor": False,
    "debug_show_final_size": False,
    # Geometry Debugging
    "debug_show_input_frustum": False,
    "debug_show_viewer_frustum": False,
    "debug_show_input_rays": False,
    "debug_show_world_axes": True, # Default axes to True
    "wavelet_packet_window_size": 64,
    "wavelet_packet_type": "db4",
    "fft_size": 512,
    "dmd_time_window": 10,
    "enable_cuda_transform": True,
    "planar_projection": False,       # ADDED: Toggle to use planar projection instead of spherical rays for point generation
    "use_orthographic": False,        # ADDED: Toggle between perspective and orthographic viewer projection
    "orthographic_size": 5.0,         # ADDED: Ortho camera half-height (world units)
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

# Helper function to convert hex color codes to ImGui Vec4
def hex_to_imvec4(hex_color, alpha=1.0):
    """Converts a hex color string (e.g., "#RRGGBB") to an imgui.Vec4."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    # Handle RGB or RGBA hex strings
    if lv == 6: # RGB
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        return imgui.Vec4(rgb[0], rgb[1], rgb[2], alpha)
    elif lv == 8: # RGBA
        rgba = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
        return imgui.Vec4(rgba[0], rgba[1], rgba[2], rgba[3])
    else: # Invalid format, return default (e.g., gray)
        print(f"Warning: Invalid hex color format '{hex_color}'. Using gray.")
        return imgui.Vec4(0.5, 0.5, 0.5, 1.0)


# Simple shaders (Using 'vertices' instead of 'position')
vertex_source = '''#version 150 core
    in vec3 vertices;
    in vec3 colors; // Expects normalized floats (0.0-1.0)

    out vec3 vertex_colors;
    // Debug outputs
    out float debug_inputDistance_out;
    out float debug_rawDiameter_out;
    out float debug_densityFactor_out;
    out float debug_finalSize_out;

    uniform mat4 projection;
    uniform mat4 view;
    uniform float inputScaleFactor; // Controlled via ImGui
    uniform float pointSizeBoost;   // Controlled via ImGui
    uniform vec2 viewportSize;      // Width, height of viewport in pixels
    uniform float inputFocal;       // Focal length of input camera in pixel units
    uniform float minPointSize;     // Minimum point size in pixels
    uniform float maxPointSize;     // Maximum point size in pixels (if clamp enabled)
    uniform bool enableMaxSizeClamp;// Toggle for max size clamp
    uniform float depthExponent;    // Exponent applied to depth for sizing
    uniform float sizeScaleFactor;  // Tunable scale factor for depth sizing
    uniform bool planarProjectionActive; // NEW UNIFORM

    void main() {
        // Transform to view and clip space
        vec4 viewPos = view * vec4(vertices, 1.0);
        vec4 clipPos = projection * viewPos;
        gl_Position = clipPos;
        vertex_colors = colors;

        // --- Point Sizing based on INPUT Camera Distance (Inverse Square Law) and Density Compensation ---
        float inputDistance = length(vertices);
        inputDistance = max(inputDistance, 0.0001);

        float baseSize = inputFocal * inputScaleFactor;
        float diameter = 2.0 * baseSize * sizeScaleFactor * pointSizeBoost * pow(inputDistance, depthExponent);

        float densityCompensationFactor = 1.0;
        if (!planarProjectionActive) {
            // Apply spherical density compensation only if not in planar projection mode
            vec3 inputRay = normalize(vertices); 
            float cosInputLatitude = sqrt(1.0 - clamp(inputRay.y * inputRay.y, 0.0, 1.0));
            densityCompensationFactor = 1.0 / max(1e-5, cosInputLatitude);
        }

        diameter *= densityCompensationFactor;

        float finalSize = max(diameter, minPointSize);
        if (enableMaxSizeClamp) {
            finalSize = min(finalSize, maxPointSize);
        }
        gl_PointSize = finalSize;

        debug_inputDistance_out = inputDistance;
        debug_rawDiameter_out = 2.0 * baseSize * sizeScaleFactor * pointSizeBoost * pow(inputDistance, depthExponent); 
        debug_densityFactor_out = densityCompensationFactor;
        debug_finalSize_out = finalSize;
    }
'''

# Modified Fragment Shader with Controls
fragment_source = """#version 150 core
    in vec3 geom_colors; // Input from Geometry Shader
    in vec2 texCoord;    // Input texture coordinate from Geometry Shader
    // Receive debug values
    in float debug_inputDistance_frag;
    in float debug_rawDiameter_frag;
    in float debug_densityFactor_frag;
    in float debug_finalSize_frag;

    out vec4 final_color;

    // Debug uniforms
    uniform bool debug_show_input_distance;
    uniform bool debug_show_raw_diameter;
    uniform bool debug_show_density_factor;
    uniform bool debug_show_final_size;

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

    // Simple heatmap function (blue -> green -> red)
    vec3 heatmap(float value) {
        value = clamp(value, 0.0, 1.0); // Ensure value is in [0, 1]
        float r = clamp(mix(0.0, 1.0, value * 2.0), 0.0, 1.0);
        float g = clamp(mix(1.0, 0.0, abs(value - 0.5) * 2.0), 0.0, 1.0);
        float b = clamp(mix(1.0, 0.0, (value - 0.5) * 2.0), 0.0, 1.0);
        return vec3(r, g, b);
    }

    void main() {

        // --- Debug Visualizations ---
        if (debug_show_input_distance) {
            // Map distance (e.g., 0-20) to a heatmap
            float normalized_dist = clamp(debug_inputDistance_frag / 20.0, 0.0, 1.0);
            final_color = vec4(heatmap(normalized_dist), 1.0);
            return; // Skip normal processing
        }
        if (debug_show_raw_diameter) {
             // Map diameter (e.g., 0-50 pixels) to grayscale
             float gray = clamp(debug_rawDiameter_frag / 50.0, 0.0, 1.0);
             final_color = vec4(gray, gray, gray, 1.0);
             return;
        }
        if (debug_show_density_factor) {
             // Map factor (e.g., 1.0 to 5.0+) to heatmap
             float normalized_factor = clamp((debug_densityFactor_frag - 1.0) / 4.0, 0.0, 1.0);
             final_color = vec4(heatmap(normalized_factor), 1.0);
             return;
        }
         if (debug_show_final_size) {
             // Map final size (e.g., 0-50 pixels) to grayscale
             float gray = clamp(debug_finalSize_frag / 50.0, 0.0, 1.0);
             final_color = vec4(gray, gray, gray, 1.0);
             return;
        }
        // --- End Debug Visualizations ---

        // --- Image Processing --- (Only runs if no debug mode active)
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
    // Receive debug values from vertex shader
    in float debug_inputDistance_out[];
    in float debug_rawDiameter_out[];
    in float debug_densityFactor_out[];
    in float debug_finalSize_out[];

    out vec3 geom_colors;    // Pass color to fragment shader
    out vec2 texCoord;       // Pass texture coordinate to fragment shader
    // Pass debug values to fragment shader
    out float debug_inputDistance_frag;
    out float debug_rawDiameter_frag;
    out float debug_densityFactor_frag;
    out float debug_finalSize_frag;

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
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        gl_Position = centerPosition + vec4( halfSizeX, -halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(1.0, 0.0); // Bottom-right
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        gl_Position = centerPosition + vec4(-halfSizeX,  halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(0.0, 1.0); // Top-left
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
        EmitVertex();

        gl_Position = centerPosition + vec4( halfSizeX,  halfSizeY, 0.0, 0.0);
        geom_colors = vertex_colors[0];
        texCoord = vec2(1.0, 1.0); // Top-right
        // Pass debug values
        debug_inputDistance_frag = debug_inputDistance_out[0];
        debug_rawDiameter_frag = debug_rawDiameter_out[0];
        debug_densityFactor_frag = debug_densityFactor_out[0];
        debug_finalSize_frag = debug_finalSize_out[0];
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
            # --- Load GLB Sequence ---
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
            # --- Load Video or Image File ---
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
        # Initialize screen capture as virtual camera
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
        # Return None for critical objects if error occurred
        return None, None, False, False, False, [], "Error", 0, 30, None, error_message

    return cap, image_frame, is_video, is_image, is_glb_sequence, glb_files, frame_source_name, video_total_frames, video_fps, input_mode, None


def _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index, playback_state_ref, last_frame_read_time_ref, video_fps, is_video, is_image, is_glb_sequence, image_frame, frame_count, data_queue, frame_source_name):
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
        if is_video and cap:
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
        elif is_glb_sequence:
            if new_sequence_frame_index >= len(glb_files): # End of sequence
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
                    end_of_stream = True # Stop on error
        elif is_image:
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
    points_xyz = None # Initialize points_xyz
    newly_calculated_bias_map_for_queue = None # New: For sending back captured bias
    main_screen_coeff_viz_for_queue = None # New: For render_mode 3 main screen visualization

    # --- Get WPT parameters and current render mode --- (New)
    render_mode = edge_params_ref.get("render_mode", DEFAULT_SETTINGS["render_mode"])
    wavelet_packet_type = edge_params_ref.get("wavelet_packet_type", DEFAULT_SETTINGS["wavelet_packet_type"])
    wavelet_packet_window_size = edge_params_ref.get("wavelet_packet_window_size", DEFAULT_SETTINGS["wavelet_packet_window_size"])
    # --- End WPT parameters ---

    # --- Get bias map and toggle state --- 
    apply_bias = edge_params_ref.get("apply_depth_bias", False)
    bias_map = edge_params_ref.get("depth_bias_map", None)
    # --- End Get bias --- 

    if predictions is None:
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz, newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue

    current_depth_map = None
    raw_model_depth_for_bias_capture = None # Store raw depth before any modification

    if 'depth' in predictions:
        current_depth_map = predictions['depth'].squeeze().float() # Get original depth
        raw_model_depth_for_bias_capture = current_depth_map.clone() # Clone for potential bias capture

        # --- Check for Bias Capture Trigger ---
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
                    newly_calculated_bias_map_for_queue = None # Ensure it's None on error
            # Note: The trigger in edge_params_ref is not cleared here;
            # it's managed by the main thread to be a one-shot signal.

        # --- Apply Bias Correction (if enabled and valid) ---
        if apply_bias and bias_map is not None and current_depth_map is not None:
            if bias_map.shape == current_depth_map.shape:
                # Ensure bias map is on the correct device before subtracting
                if bias_map.device != current_depth_map.device:
                    bias_map = bias_map.to(current_depth_map.device)
                
                original_min = torch.min(current_depth_map).item() # For debug
                corrected_depth = current_depth_map - bias_map
                # Ensure depth doesn't become non-positive
                corrected_depth = torch.clamp(corrected_depth, min=0.01) 
                corrected_min = torch.min(corrected_depth).item() # For debug
                print(f"DEBUG: Applied depth bias. Original Min: {original_min:.3f}, Corrected Min: {corrected_min:.3f}")
                current_depth_map = corrected_depth # Overwrite with corrected map
            else:
                print(f"Warning: Skipping bias correction - bias map shape {bias_map.shape} != current depth shape {current_depth_map.shape}")
        # --- End Bias Correction ---

        # --- Apply WPT if render_mode is 3 (Wavelet/FFT) --- (New)
        if render_mode == 3 and current_depth_map is not None and current_depth_map.numel() > 0:
            print("DEBUG: Applying WPT to current_depth_map for render_mode 3")
            try:
                original_depth_for_scaling = current_depth_map.clone() # Store original for scaling

                d_in = current_depth_map.unsqueeze(0).unsqueeze(0) # Add batch and channel dims
                # J = levels of decomposition
                J = int(math.log2(max(1, wavelet_packet_window_size))) 
                # Ensure J is not too large for the input size
                min_dim = min(d_in.shape[-2], d_in.shape[-1])
                max_J_possible = int(math.log2(min_dim)) if min_dim > 0 else 0
                if J > max_J_possible:
                    print(f"Warning: Requested J={J} for WPT is too large for depth map size ({d_in.shape[-2]}x{d_in.shape[-1]}). Clamping to J={max_J_possible}.")
                    J = max_J_possible

                if J > 0: # Only proceed if at least one level of decomposition is possible
                    dwt_op = DWTForward(J=J, wave=wavelet_packet_type, mode='zero').to(device)
                    idwt_op = DWTInverse(wave=wavelet_packet_type, mode='zero').to(device)
                    Yl, Yh = dwt_op(d_in)
                    wpt_processed_depth = idwt_op((Yl, Yh)).squeeze(0).squeeze(0)
                    
                    # --- NEW: Compute per-pixel WPT energy for Gaussian splatting ---
                    # Use the highest-frequency subbands at the last level
                    # Yh is a list of length J, each [B, C, 3, H, W] (LH, HL, HH)
                    # We'll use the last level (finest scale)
                    lh = Yh[-1][0, 0, 0, :, :].abs().cpu().numpy() # LH
                    hl = Yh[-1][0, 0, 1, :, :].abs().cpu().numpy() # HL
                    hh = Yh[-1][0, 0, 2, :, :].abs().cpu().numpy() # HH
                    # Energy (sum of squares)
                    energy = (lh**2 + hl**2 + hh**2)
                    # Normalize energy for size/opacity
                    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-7)
                    # Map to reasonable size/opacity ranges
                    size_map = 2.0 + 8.0 * energy_norm # [2, 10] pixels
                    opacity_map = 0.2 + 0.8 * energy_norm # [0.2, 1.0]
                    # Color: map LH, HL, HH to RGB (normalize for display)
                    lh_n = (lh - lh.min()) / (lh.max() - lh.min() + 1e-7)
                    hl_n = (hl - hl.min()) / (hl.max() - hl.min() + 1e-7)
                    hh_n = (hh - hh.min()) / (hh.max() - hh.min() + 1e-7)
                    color_map = np.stack([lh_n, hl_n, hh_n], axis=-1)
                    # Points: get 3D positions as before
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
                    # Flatten all for splatting
                    points_xyz_np_processed = points_xyz.reshape(-1, 3)
                    colors_np_processed = color_map.reshape(-1, 3)
                    sizes_np_processed = size_map.flatten()
                    opacities_np_processed = opacity_map.flatten()
                    num_vertices = points_xyz_np_processed.shape[0]
                    # Store for queue (add to return tuple at end)
                    # Also update main_screen_coeff_viz_for_queue as before
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
        # --- End WPT Application ---

        # --- Check current_depth_map state before CoeffViz ---
        if render_mode == 3:
            if current_depth_map is not None:
                print(f"DEBUG CoeffViz Pre-Check: current_depth_map exists. Shape: {current_depth_map.shape}, Numel: {current_depth_map.numel()}, Device: {current_depth_map.device}")
            else:
                print("DEBUG CoeffViz Pre-Check: current_depth_map is None before attempting CoeffViz.")
        # --- End Check ---

        # --- Start: Generate Coefficient Visualization if render_mode == 3 ---
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

                if actual_J_viz > 0:
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
                    main_screen_coeff_viz_for_queue = None # Explicitly set to None
            except Exception as e_coeff_viz:
                print(f"ERROR CoeffViz: Error generating coefficient visualization for main screen: {e_coeff_viz}")
                traceback.print_exc()
                main_screen_coeff_viz_for_queue = None # Ensure it's None on error
        else:
            if render_mode == 3: # Only print this if mode is 3 but other conditions failed
                 print(f"DEBUG CoeffViz: Skipped main screen coeff viz. current_depth_map valid: {current_depth_map is not None and current_depth_map.numel() > 0}")
            main_screen_coeff_viz_for_queue = None # Ensure it is None if not render_mode 3 or depth invalid
        # --- End: Generate Coefficient Visualization --- 

        # Now proceed with planar or spherical projection using the (potentially WPT-processed or bias-corrected) current_depth_map
        if edge_params_ref.get('planar_projection', False):
            H_orig, W_orig = (480,640)
            if current_depth_map is not None:
                 H_orig, W_orig = current_depth_map.shape
            # else: # No need for the print here if we are not forcing depth
                 # print(f"DIAGNOSTIC: Planar mode, but no initial depth from model. Using default H,W: {H_orig}x{W_orig}")

            # --- REMOVE FORCED CONSTANT DEPTH FOR PLANAR CASE ---
            # forced_depth_value_planar = 5.0
            # print(f"DIAGNOSTIC: Planar mode - Forcing constant depth: {forced_depth_value_planar}")
            # current_depth_map = torch.full((H_orig, W_orig), forced_depth_value_planar, device=device, dtype=torch.float32)
            # H, W = current_depth_map.shape 
            # --- END REMOVAL ---
            # Ensure H, W are from the actual current_depth_map if it exists, else use defaults for grid gen if needed (though it should exist if 'depth' in predictions)
            if current_depth_map is not None:
                H, W = current_depth_map.shape
            else: # Should not happen if 'depth' in predictions, but as a fallback for H,W if current_depth_map became None unexpectedly
                H, W = H_orig, W_orig 
            
            input_fov_deg = edge_params_ref.get('input_camera_fov', 60.0)
            # --- Safeguard for None FOV value from ref dict --- 
            if input_fov_deg is None:
                 print("WARNING: input_fov_deg was None in edge_params_ref! Defaulting to 60.0.")
                 input_fov_deg = 60.0
            # --- End Safeguard ---

            fov_y_rad = torch.tensor(math.radians(input_fov_deg), device=device, dtype=torch.float32)
            f_y = H / (2 * torch.tan(fov_y_rad / 2.0))
            f_x = f_y 
            cx = torch.tensor(W / 2.0, device=device, dtype=torch.float32)
            cy = torch.tensor(H / 2.0, device=device, dtype=torch.float32)
            jj = torch.arange(0, H, device=device, dtype=torch.float32)
            ii = torch.arange(0, W, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(jj, ii, indexing='ij')
            depth_values = current_depth_map # This will now be the model's actual depth
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
                rays = rays / rays_norm_val # Corrected variable name
            
            # --- DIAGNOSTIC: Force constant depth for spherical case ---
            forced_depth_value_spherical = 5.0 
            # Determine shape for the forced depth map
            H_sph, W_sph = (480,640) # Default if rays or original current_depth_map is None
            if current_depth_map is not None:
                H_sph, W_sph = current_depth_map.shape
                print(f"DIAGNOSTIC: Spherical mode - Overriding model depth with constant {forced_depth_value_spherical}")
            elif rays is not None and rays.ndim ==3: # If no depth from model, but we have rays
                H_sph, W_sph, _ = rays.shape
                print(f"DIAGNOSTIC: Spherical mode - Model provided no depth, creating constant depth {forced_depth_value_spherical} based on ray dimensions.")
            else:
                 print(f"DIAGNOSTIC: Spherical mode - Cannot determine shape for forced depth. Skipping override.")
            
            # Only override/create if we have a valid shape
            if H_sph > 0 and W_sph > 0:
                 current_depth_map = torch.full((H_sph, W_sph), forced_depth_value_spherical, device=device, dtype=torch.float32)
            # --- END DIAGNOSTIC ---

            if current_depth_map is not None and rays is not None and current_depth_map.shape == rays.shape[:2]:
                depth_to_multiply = current_depth_map.unsqueeze(-1)
                points_xyz = rays * depth_to_multiply
            else:
                print(f"Warning: Spherical rays shape {rays.shape if rays is not None else 'N/A'} and overridden/original depth map shape {current_depth_map.shape if current_depth_map is not None else 'N/A'} mismatch or depth missing.")
        else:
            print("Warning: Depth map present but no UniK3D rays found and planar projection is off. Cannot generate points from depth.")
    elif 'rays' in predictions: # Only rays, no depth from model initially for spherical case
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

    elif "points" in predictions: # If model directly outputs points (e.g. GLB direct load)
        points_xyz = predictions["points"]
        # Ensure points_xyz is H, W, C or N, C.
        if points_xyz.ndim == 3 and points_xyz.shape[0] == 1 and points_xyz.shape[-1] == 3 : # Check for [1, N, 3]
            points_xyz = points_xyz.squeeze(0) # Convert to [N, 3]
        # If model output points are already in a different convention, that needs to be handled
        # Assuming 'points' from predictions are in the (+X Right, +Y Down, +Z Forward) convention
        # or will be handled by the GLB loading specific path if that's where they originate.
    else:
        print("Warning: No points or depth found in UniK3D predictions.")
        # points_xyz remains None

    if points_xyz is None or points_xyz.numel() == 0:
        print("Warning: No valid points generated from predictions.")
        return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz, newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue

    # --- Edge Detection & Depth Processing (Run for all modes now) ---
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

    # --- Point Thickening (Operates on smoothed points, BEFORE coord flip) ---
    points_to_thicken = new_smoothed_points_xyz # Use smoothed points as base
    enable_thickening = edge_params_ref.get("enable_point_thickening", False)
    num_duplicates = edge_params_ref.get("thickening_duplicates", 0)
    variance = edge_params_ref.get("thickening_variance", 0.0)
    depth_bias = edge_params_ref.get("thickening_depth_bias", 0.0)

    if enable_thickening and num_duplicates > 0 and points_to_thicken.numel() > 0:
        try:
            # Ensure points_to_thicken is on CPU and NumPy for processing
            if points_to_thicken.is_cuda:
                points_np = points_to_thicken.squeeze().cpu().numpy()
            else:
                points_np = points_to_thicken.squeeze().numpy()

            # Handle potential different shapes from model output
            if points_np.ndim == 3: # H, W, C
                 N = points_np.shape[0] * points_np.shape[1]
                 points_np = points_np.reshape(N, 3)
            elif points_np.ndim == 2 and points_np.shape[1] == 3: # N, C
                 N = points_np.shape[0]
            else:
                 raise ValueError(f"Unexpected points shape for thickening: {points_np.shape}")

            # Get original colors (needs similar reshape logic)
            # Assuming rgb_frame_processed matches dimensions if points were HxWxC
            if rgb_frame_processed is not None and rgb_frame_processed.ndim == 3:
                colors_np = (rgb_frame_processed.reshape(N, 3).astype(np.float32) / 255.0)
            else: # Fallback: Use white if no color frame or mismatch
                colors_np = np.ones((N, 3), dtype=np.float32)

            # Calculate ray directions (vector from origin to point)
            norms = np.linalg.norm(points_np, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1e-6 # Avoid division by zero
            ray_directions = points_np / norms

            # Generate duplicates
            num_new_points = N * num_duplicates
            duplicate_points = np.zeros((num_new_points, 3), dtype=np.float32)
            duplicate_colors = np.zeros((num_new_points, 3), dtype=np.float32)

            # Random offsets (Gaussian noise)
            random_offsets = np.random.normal(0.0, variance, size=(num_new_points, 3)).astype(np.float32)

            # Biased offsets along ray direction (positive Z is 'forward'/'away' here)
            # Use positive ray direction for backward push in this coord system
            bias_magnitudes = np.random.uniform(0.0, depth_bias, size=(num_new_points, 1)).astype(np.float32)
            # Repeat ray directions for each duplicate
            repeated_ray_dirs = np.repeat(ray_directions, num_duplicates, axis=0)
            biased_offsets = repeated_ray_dirs * bias_magnitudes

            # Calculate final duplicate positions
            repeated_original_points = np.repeat(points_np, num_duplicates, axis=0)
            duplicate_points = repeated_original_points + random_offsets + biased_offsets

            # Assign colors to duplicates
            duplicate_colors = np.repeat(colors_np, num_duplicates, axis=0)

            # Combine original and duplicates
            points_xyz_np_thickened = np.vstack((points_np, duplicate_points))
            colors_np_thickened = np.vstack((colors_np, duplicate_colors))

        except Exception as e_thicken:
            print(f"Error during point thickening: {e_thicken}")
            traceback.print_exc()
            # Fallback to original smoothed points if thickening fails
            if new_smoothed_points_xyz.is_cuda:
                 points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().cpu().numpy()
            else:
                 points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().numpy()
            # Recalculate colors_np_thickened based on potentially reshaped points_xyz_np_thickened
            N_fallback = points_xyz_np_thickened.shape[0]
            if rgb_frame_processed is not None and rgb_frame_processed.ndim == 3 and rgb_frame_processed.size == N_fallback*3:
                 colors_np_thickened = (rgb_frame_processed.reshape(N_fallback, 3).astype(np.float32) / 255.0)
            else: colors_np_thickened = np.ones((N_fallback, 3), dtype=np.float32)

    else:
        # Thickening disabled or no duplicates requested, use smoothed points
        if new_smoothed_points_xyz.is_cuda:
             points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().cpu().numpy()
        else:
             points_xyz_np_thickened = new_smoothed_points_xyz.squeeze().numpy()
        # Reshape/get colors for non-thickened points
        if points_xyz_np_thickened.ndim == 3: # H, W, C
            N_orig = points_xyz_np_thickened.shape[0] * points_xyz_np_thickened.shape[1]
            points_xyz_np_thickened = points_xyz_np_thickened.reshape(N_orig, 3)
        elif points_xyz_np_thickened.ndim == 2: # N, C
            N_orig = points_xyz_np_thickened.shape[0]
        else: N_orig = 0

        if N_orig > 0 and rgb_frame_processed is not None and rgb_frame_processed.ndim == 3 and rgb_frame_processed.size == N_orig * 3:
            colors_np_thickened = (rgb_frame_processed.reshape(N_orig, 3).astype(np.float32) / 255.0)
        else: colors_np_thickened = np.ones((N_orig, 3), dtype=np.float32)

    # Ensure points are reshaped correctly before transform if they came directly from smoothing
    if points_xyz_np_thickened.ndim == 3: # H, W, C format?
        num_vertices_final = points_xyz_np_thickened.shape[0] * points_xyz_np_thickened.shape[1]
        points_xyz_np_processed = points_xyz_np_thickened.reshape(num_vertices_final, 3)
    elif points_xyz_np_thickened.ndim == 2: # N, C format
        num_vertices_final = points_xyz_np_thickened.shape[0]
        points_xyz_np_processed = points_xyz_np_thickened # Already correct shape
    else:
        print(f"Warning: Unexpected points shape after thickening/smoothing: {points_xyz_np_thickened.shape}")
        num_vertices_final = 0
        points_xyz_np_processed = np.array([], dtype=np.float32).reshape(0,3)
        colors_np_processed = np.array([], dtype=np.float32).reshape(0,3)

    # Assign processed colors (already calculated in thickening or fallback)
    colors_np_processed = colors_np_thickened if num_vertices_final > 0 else np.array([], dtype=np.float32).reshape(0,3)
    num_vertices = num_vertices_final # Update final vertex count

    # --- Transform Coordinates for OpenGL/GLB standard (+X Right, +Y Up, -Z Forward) ---
    # Apply this AFTER thickening/smoothing
    if points_xyz_np_processed.ndim >= 2 and points_xyz_np_processed.shape[-1] == 3:
        points_xyz_np_processed[..., 1] *= -1.0 # Flip Y (Down -> Up)
        points_xyz_np_processed[..., 2] *= -1.0 # Flip Z (Forward -> -Forward)
    # ------------------------------------

    # Reshape and get vertex count (This block seems redundant now)
    # ... (Remove the reshape block here as points_xyz_np_processed is now guaranteed N x 3) ...

    # --- Sample Colors --- (This block seems redundant now)
    # ... (Remove color sampling block here as colors_np_processed is now set)

    # Use white if colors failed (Should be handled by fallback logic now)
    # ... (Remove this fallback) ...

    # Return processed points, colors, counts, and debug maps
    return points_xyz_np_processed, colors_np_processed, num_vertices, \
           scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, \
           new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz, \
           newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue # Added new viz content


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


def _queue_results(data_queue, vertices_flat, colors_flat, num_vertices,
                   rgb_frame_orig, scaled_depth_map_for_queue, edge_map_viz,
                   smoothing_map_viz, t_capture, sequence_frame_index,
                   video_total_frames, current_recorded_count, frame_count,
                   frame_read_delta_t, depth_process_delta_t, latency_ms,
                   newly_calculated_bias_map_for_main_thread, # Existing last custom item
                   main_screen_coeff_viz_content): # New item for coeff viz
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
                            frame_read_delta_t, depth_process_delta_t, latency_ms,
                            newly_calculated_bias_map_for_main_thread,
                            main_screen_coeff_viz_content)) # Added new item
    else:
        print(f"Warning: Viewer queue full, dropping frame {frame_count}.")


# --- Main Inference Thread Function ---
# --- Inference Thread Function ---
# --- Main Inference Thread Function (Refactored) ---
def inference_thread_func(data_queue, exit_event,
                          model, device, # Accept model and device as arguments
                          inference_interval,
                          scale_factor_ref, edge_params_ref,
                          input_mode, input_filepath, playback_state,
                          recording_state, live_processing_mode,
                          screen_capture_monitor_index):
    """Captures/loads data, runs inference, processes, and queues results."""
    # Model loading is now done in the main thread
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
    # model = None # Removed
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Removed

    try:
        # --- Skip Model Loading (already loaded) ---
        if input_mode == "GLB Sequence":
            print("GLB Sequence mode: Skipping inference.")
            # data_queue.put(("status", "GLB Sequence mode: Skipping inference.")) # Status already sent

        # --- Initialize Input Source ---
        cap, image_frame, is_video, is_image, is_glb_sequence, glb_files, \
        frame_source_name, video_total_frames, video_fps, input_mode_returned, error_message = \
            _initialize_input_source(input_mode, input_filepath, data_queue, playback_state)

        # If initialization changed the mode (e.g., File->GLB Sequence), update local var
        if input_mode_returned != input_mode:
            print(f"Input mode auto-detected as: {input_mode_returned}")
            input_mode = input_mode_returned


        # Start timestamp for virtual live playback (File/GLB)
        media_start_time = time.time()

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
                                           frame_count, data_queue, frame_source_name)
            elif input_mode == "Screen" and is_video:
                # Capture screen as virtual camera using the selected monitor index
                try:
                    with mss.mss() as sct:
                        # Ensure monitor index is valid before using it
                        monitor_to_capture = sct.monitors[screen_capture_monitor_index] if screen_capture_monitor_index < len(sct.monitors) else sct.monitors[0] # Fallback to monitor 0
                        sct_img = sct.grab(monitor_to_capture)
                    
                    # Ensure frame is BGR (convert from BGRA if needed)
                    img_arr = np.array(sct_img)
                    if img_arr.shape[2] == 4:
                        frame = img_arr[:, :, :3] # Take only BGR channels
                    elif img_arr.shape[2] == 3:
                        frame = img_arr # Assume it's already BGR
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
                    # Attempt fallback to primary monitor (index 1 if exists, else 0) on error?
                    try:
                        # Fallback logic needs sct instance! Re-initialize mss here.
                        print(f"Attempting fallback capture...")
                        with mss.mss() as sct_fallback: # Re-initialize mss
                            fallback_index = 1 if len(sct_fallback.monitors) > 1 else 0
                            print(f"Attempting fallback capture on monitor {fallback_index}...")
                            # Ensure fallback index is valid too
                            fallback_monitor = sct_fallback.monitors[fallback_index] if fallback_index < len(sct_fallback.monitors) else sct_fallback.monitors[0]
                            sct_img = sct_fallback.grab(fallback_monitor) # Use fallback_monitor
                        
                        # Ensure frame is BGR (convert from BGRA if needed)
                        img_arr = np.array(sct_img)
                        if img_arr.shape[2] == 4:
                            frame = img_arr[:, :, :3] # Take only BGR channels
                        elif img_arr.shape[2] == 3:
                            frame = img_arr # Assume it's already BGR
                        else:
                            print(f"ERROR: Fallback screen capture has unexpected shape {img_arr.shape}")
                            ret = False
                            frame = None

                        if frame is not None:
                            # Fix indentation here
                            ret = True
                            current_time = time.time()
                            frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                            last_frame_read_time_ref[0] = current_time
                            sequence_frame_index = frame_count + 1
                            playback_state["current_frame"] = sequence_frame_index
                            print(f"Fallback capture successful.")
                    except Exception as e_fallback:
                            print(f"Fallback screen capture failed: {e_fallback}")
                            ret = False # Indicate failure if fallback also fails
                            time.sleep(0.1) # Sleep longer on persistent error
                            frame = None # Ensure frame is None on complete failure
                            # No 'continue' here if fallback succeeded
                            if not ret: # If fallback also failed
                                time.sleep(0.05)
                                continue # Skip processing this cycle

            elif input_mode in ["File", "GLB Sequence"] and not is_image:
                # Virtual live mode: drop frames based on real-time timing
                # Handle restart flag by resetting the media start time
                if playback_state.get("restart", False):
                    media_start_time = time.time()
                    playback_state["restart"] = False

                elapsed = time.time() - media_start_time
                frame_idx = int(elapsed * video_fps)
                total_frames = video_total_frames
                loop_video = playback_state.get("loop", True)
                
                if total_frames > 0:
                    # Apply looping or end-of-stream logic
                    if frame_idx >= total_frames:
                        if loop_video:
                            frame_idx = frame_idx % total_frames
                        else:
                            end_of_stream = True

                    if not end_of_stream:
                        # Seek and read video frames
                        if is_video and cap:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                        # Load GLB sequence frames
                        elif is_glb_sequence:
                            if frame_idx < len(glb_files):
                                points_xyz_np_loaded, colors_np_loaded = load_glb(glb_files[frame_idx])
                                ret = points_xyz_np_loaded is not None
                            else:
                                end_of_stream = True

                        # Update frame index and shared state (1-based for display)
                        sequence_frame_index = frame_idx + 1
                        playback_state["current_frame"] = sequence_frame_index

                        # Update timing reference for frame read delta
                        current_time = time.time()
                        frame_read_delta_t = current_time - last_frame_read_time_ref[0]
                        last_frame_read_time_ref[0] = current_time
            else:
                # Image or fallback file handling
                frame, points_xyz_np_loaded, colors_np_loaded, ret, \
                sequence_frame_index, frame_read_delta_t, end_of_stream = \
                    _read_or_load_frame(input_mode, cap, glb_files, sequence_frame_index,
                                       playback_state, last_frame_read_time_ref, video_fps,
                                       is_video, is_image, is_glb_sequence, image_frame,
                                       frame_count, data_queue, frame_source_name)

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
                # --- Preprocess Frame ---
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
                # data_queue.put(("status", f"Running inference on frame {frame_count}...")) # DEBUG Verbose
                predictions = _run_model_inference(model, rgb_frame_processed, device) # Use pre-loaded model

                # --- Process Results ---
                data_queue.put(("status", f"Processing results for frame {frame_count}..."))
                points_xyz_np_processed, colors_np_processed, num_vertices, \
                scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, \
                prev_depth_map, smoothed_mean_depth, smoothed_points_xyz, \
                newly_calculated_bias_map_for_queue, main_screen_coeff_viz_for_queue = \
                    _process_inference_results(predictions, rgb_frame_processed, device,
                                               edge_params_ref, prev_depth_map,
                                               smoothed_mean_depth, smoothed_points_xyz,
                                               frame_h, frame_w, input_mode)
                # Mark depth processed time for inference
                current_time = time.time()
                depth_process_delta_t = current_time - last_depth_processed_time
                last_depth_processed_time = current_time


            # --- Handle Recording ---
            recorded_frame_counter = _handle_recording(points_xyz_np_processed, colors_np_processed,
                                                       recording_state, sequence_frame_index, # Corrected variable name
                                                       recorded_frame_counter, is_video, is_glb_sequence,
                                                       data_queue)

            # --- Calculate Latency & Queue Results ---
            latency_ms = (time.time() - t_capture) * 1000.0 if t_capture else 0.0 # Handle potential None t_capture?
            vertices_flat = points_xyz_np_processed.flatten() if points_xyz_np_processed is not None and points_xyz_np_processed.size > 0 else None
            colors_flat = colors_np_processed.flatten() if colors_np_processed is not None and colors_np_processed.size > 0 else None

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
                               frame_read_delta_t, depth_process_delta_t, latency_ms,
                               newly_calculated_bias_map_for_queue, # Pass existing bias map
                               main_screen_coeff_viz_for_queue)      # Pass new coeff viz numpy
            print(f"DEBUG: Queuing {num_vertices if vertices_flat is not None else 0} vertices (frame {frame_count}).") 

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
                 disable_point_smoothing=False,
                 *args, **kwargs):
        # Store args before calling super
        self._model_name = model_name
        self._inference_interval = inference_interval

        print("DEBUG: LiveViewerWindow.__init__ - Start") # Added print

        super().__init__(*args, **kwargs)

        print("DEBUG: LiveViewerWindow.__init__ - After super().__init__") # Added print

        # --- Model Loading (moved to main thread __init__) ---
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            print(f"Loading UniK3D model: {self._model_name} on device: {self.device}...")
            self.model = UniK3D.from_pretrained(f"lpiccinelli/{self._model_name}")
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e_load:
            print(f"FATAL: Error loading model {self._model_name}: {e_load}")
            traceback.print_exc()
            # Exit if model loading fails, as it's essential for most modes
            pyglet.app.exit()
            return
        # --- End Model Loading ---

        print("DEBUG: LiveViewerWindow.__init__ - After Model Load") # Added print

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
        self.size_scale_factor = None # Added
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
        self.show_wavelet_map = None  # Flag to display Wavelet overlay
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
        # Debug Flags (Additions)
        self.debug_show_input_distance = None
        self.debug_show_raw_diameter = None
        self.debug_show_density_factor = None
        self.debug_show_final_size = None
        # Geometry Debug Flags
        self.debug_show_input_frustum = None
        self.debug_show_viewer_frustum = None
        self.debug_show_input_rays = None
        self.debug_show_world_axes = None
        # Orthographic projection settings
        self.use_orthographic = None
        self.orthographic_size = None
        # Planar ray generation toggle
        self.planar_projection = None

        # Store latest vertex data for debug drawing (rays)
        self.latest_points_for_debug = None

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
        self.latest_wavelet_map = None # Initialize latest_wavelet_map
        self.camera_texture = None
        self.depth_texture = None
        self.edge_texture = None
        self.smoothing_texture = None
        self.debug_textures_initialized = False
        # Store texture dimensions
        self.camera_texture_width = 0
        self.camera_texture_height = 0
        self.depth_texture_width = 0
        self.depth_texture_height = 0
        self.edge_texture_width = 0
        self.edge_texture_height = 0
        self.smoothing_texture_width = 0
        self.smoothing_texture_height = 0

        # Load initial settings (this initializes control variables and refs)
        self.load_settings() # This will now load/default screen_capture_monitor_index

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

        # --- Main Point Cloud Shader --- 
        print("DEBUG: Creating main point cloud shaders...")
        try:
            vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
            frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
            geom_shader = pyglet.graphics.shader.Shader(geometry_source, 'geometry')
            self.shader_program = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader, geom_shader)
            print(f"DEBUG: Main shader program created.")
        except pyglet.graphics.shader.ShaderException as e:
            print(f"FATAL: Main shader compilation error: {e}")
            pyglet.app.exit()
            return
        except Exception as e:
             print(f"FATAL: Error during main shader setup: {e}")
             pyglet.app.exit()
             return

        # --- Simple Debug Line Shader --- 
        print("DEBUG: Creating debug line shaders...")
        debug_vertex_source = """#version 150 core
            in vec3 position;
            in vec3 color;
            out vec3 vertex_colors;

            uniform mat4 projection;
            uniform mat4 view;

            void main() {
                gl_Position = projection * view * vec4(position, 1.0);
                vertex_colors = color;
            }
        """
        debug_fragment_source = """#version 150 core
            in vec3 vertex_colors;
            out vec4 final_color;

            void main() {
                final_color = vec4(vertex_colors, 1.0);
            }
        """
        try:
            debug_vert_shader = pyglet.graphics.shader.Shader(debug_vertex_source, 'vertex')
            debug_frag_shader = pyglet.graphics.shader.Shader(debug_fragment_source, 'fragment')
            self.debug_shader_program = pyglet.graphics.shader.ShaderProgram(debug_vert_shader, debug_frag_shader)
            print("DEBUG: Debug shader program created.")
        except pyglet.graphics.shader.ShaderException as e:
            print(f"FATAL: Debug shader compilation error: {e}")
            pyglet.app.exit()
            return
        except Exception as e:
            print(f"FATAL: Error during debug shader setup: {e}")
            pyglet.app.exit()
            return

        print("DEBUG: Scheduling updates...") # Added print
        # Schedule the main update function
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)
        pyglet.clock.schedule_interval(self.update_camera, 1.0 / 60.0)

        # --- Threading Setup ---
        self._data_queue = queue.Queue(maxsize=5)
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

        # Add state for screen share pop-up and selection
        self.show_screen_share_popup = False
        self.temp_monitor_index = 0 # Temporary storage for selection in popup

        print("DEBUG: LiveViewerWindow.__init__ - End") # Added print

        # Prepare depth history buffer for Wavelet/FFT
        self.depth_history = deque(maxlen=DEFAULT_SETTINGS["dmd_time_window"]) # Use default initially
        self.latest_depth_tensor = None  # Latest raw depth tensor on GPU

        # --- Set up a default VAO so ImGui can bind/unbind safely ---
        self._default_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._default_vao)
        gl.glBindVertexArray(self._default_vao)

        self.input_camera_fov = None # Initialized before load_settings
        # Orthographic projection settings
        self.use_orthographic = None
        self.orthographic_size = None
        # Planar ray generation toggle
        self.planar_projection = None

        # --- Calibration State ---
        self.depth_bias_map = None # Stores the calculated bias tensor (device tensor)
        self.apply_depth_bias = False # Toggle for applying the correction
        self.latest_depth_tensor_for_calib = None # Store the raw tensor used for last calc
        self.pending_bias_capture_request = None # New: For triggering bias capture in inference thread

        # --- Texture Shader for Full-Screen Quad (Wavelet/FFT Mode) --- (New)
        self.texture_shader_program = None
        self.texture_quad_vao = None
        self.texture_quad_vbo = None
        try:
            print("DEBUG: Creating texture quad shaders...")
            texture_vert_shader = pyglet.graphics.shader.Shader(texture_vertex_source, 'vertex')
            texture_frag_shader = pyglet.graphics.shader.Shader(texture_fragment_source, 'fragment')
            self.texture_shader_program = pyglet.graphics.shader.ShaderProgram(texture_vert_shader, texture_frag_shader)
            print("DEBUG: Texture quad shader program created.")

            # Quad vertices (x, y) and texture coordinates (s, t)
            quad_vertices = [
                # pos    # tex
                -1, -1,  0, 0,
                 1, -1,  1, 0,
                -1,  1,  0, 1,
                 1,  1,  1, 1,
            ]
            quad_vertices_gl = (gl.GLfloat * len(quad_vertices))(*quad_vertices)

            self.texture_quad_vbo = gl.GLuint()
            gl.glGenBuffers(1, self.texture_quad_vbo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texture_quad_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices_gl, gl.GL_STATIC_DRAW)

            self.texture_quad_vao = gl.GLuint()
            gl.glGenVertexArrays(1, self.texture_quad_vao)
            gl.glBindVertexArray(self.texture_quad_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texture_quad_vbo) # Bind VBO before vertex_attrib_pointer

            # Position attribute
            pos_attrib = self.texture_shader_program['position'].location
            gl.glEnableVertexAttribArray(pos_attrib)
            gl.glVertexAttribPointer(pos_attrib, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, 0) # Stride is 4 floats, offset 0
            # Texture coordinate attribute
            tex_coord_attrib = self.texture_shader_program['texCoord_in'].location
            gl.glEnableVertexAttribArray(tex_coord_attrib)
            gl.glVertexAttribPointer(tex_coord_attrib, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, 2 * 4) # Stride 4 floats, offset 2 floats

            gl.glBindVertexArray(0) # Unbind VAO
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0) # Unbind VBO
            print("DEBUG: Texture quad VAO/VBO created.")

        except pyglet.graphics.shader.ShaderException as e:
            print(f"FATAL: Texture quad shader compilation error: {e}")
            traceback.print_exc()
            pyglet.app.exit()
            return
        except Exception as e_tex_shader:
            print(f"FATAL: Error during texture quad shader setup: {e_tex_shader}")
            traceback.print_exc()
            pyglet.app.exit()
            return
        # --- End Texture Shader Setup ---

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
        if hasattr(self, 'wavelet_texture') and self.wavelet_texture: # Main screen WPT viz texture
            try: gl.glDeleteTextures(1, self.wavelet_texture)
            except: pass
        if hasattr(self, 'imgui_wavelet_debug_texture') and self.imgui_wavelet_debug_texture: # ImGui WPT debug texture
            try: gl.glDeleteTextures(1, self.imgui_wavelet_debug_texture)
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
        self.camera_texture_width = width
        self.camera_texture_height = height

        self.depth_texture = gl.GLuint()
        gl.glGenTextures(1, self.depth_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        self.depth_texture_width = width
        self.depth_texture_height = height

        self.edge_texture = gl.GLuint()
        gl.glGenTextures(1, self.edge_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.edge_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        self.edge_texture_width = width
        self.edge_texture_height = height

        self.smoothing_texture = gl.GLuint()
        gl.glGenTextures(1, self.smoothing_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.smoothing_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        self.smoothing_texture_width = width
        self.smoothing_texture_height = height

        # Create texture for Wavelet/FFT output
        self.wavelet_texture = gl.GLuint()
        gl.glGenTextures(1, self.wavelet_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.wavelet_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        self.wavelet_texture_width = width
        self.wavelet_texture_height = height
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Create texture for ImGui Wavelet/FFT debug output
        self.imgui_wavelet_debug_texture = gl.GLuint()
        gl.glGenTextures(1, self.imgui_wavelet_debug_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.imgui_wavelet_debug_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        # Storing separate dimensions for this texture might be useful if it can differ from wavelet_texture
        self.imgui_wavelet_debug_texture_width = width 
        self.imgui_wavelet_debug_texture_height = height
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # Unbind
        self.debug_textures_initialized = True
        print("DEBUG: Debug textures created/recreated.")


    def start_inference_thread(self): # Removed arguments, uses self attributes
        # Stop existing thread first if running
        if self.inference_thread and self.inference_thread.is_alive():
            print("DEBUG: Stopping existing inference thread...")
            self.status_message = "Stopping previous source..."  # Update status
            # Signal the old thread to exit
            self._exit_event.set()
            # Wait for the old thread to finish
            self.inference_thread.join(timeout=5.0)
            if self.inference_thread.is_alive():
                print("Warning: Inference thread did not stop in time.")
            # Clear the old thread reference
            self.inference_thread = None
            # Flush any pending data to drop stale frames
            try:
                while True:
                    self._data_queue.get_nowait()
            except queue.Empty:
                pass
            # Clear vertex list when source changes
            if self.vertex_list:
                try:
                    self.vertex_list.delete()
                except:
                    pass
                self.vertex_list = None
                self.current_point_count = 0
        # Create a fresh exit event for the new thread to use
        self._exit_event = threading.Event()

        # Ensure refs are initialized
        # self.scale_factor_ref and self.edge_params_ref initialized in load_settings / _update_edge_params
        self._update_edge_params() # Call this to ensure edge_params_ref is current
        self.update_playback_state() # Initialize playback state
        self.update_recording_state() # Initialize recording state
        # Ensure screen capture index attribute exists
        if not hasattr(self, 'screen_capture_monitor_index'):
            self.screen_capture_monitor_index = DEFAULT_SETTINGS['screen_capture_monitor_index']
        # Ensure scale_factor_ref is initialized if not done by load_settings -> _update_edge_params path
        if not hasattr(self, 'scale_factor_ref') or self.scale_factor_ref is None:
            self.scale_factor_ref = [getattr(self, 'input_scale_factor', DEFAULT_SETTINGS['input_scale_factor'])]


        # Start new thread with current settings, including monitor index
        self.inference_thread = threading.Thread(
            target=inference_thread_func,
            args=(self._data_queue, self._exit_event,
                  self.model, # Pass pre-loaded model
                  self.device, # Pass device
                  self._inference_interval,
                  self.scale_factor_ref,
                  self.edge_params_ref,
                  self.input_mode,
                  self.input_filepath,
                  self.playback_state,
                  self.recording_state,
                  self.live_processing_mode,
                  self.screen_capture_monitor_index),
            daemon=True
        )
        self.inference_thread.start()
        print(f"DEBUG: Inference thread started (Mode: {self.input_mode}, Monitor: {self.screen_capture_monitor_index if self.input_mode == 'Screen' else 'N/A'}).")
        self.status_message = f"Starting {self.input_mode}..."

    def _update_edge_params(self):
        """Updates the dictionary passed to the inference thread. Uses getattr for safety during init."""
        # Use getattr for potentially missing attributes during initialization, defaulting to DEFAULT_SETTINGS values
        self.edge_params_ref["enable_point_smoothing"] = getattr(self, "enable_point_smoothing", DEFAULT_SETTINGS["enable_point_smoothing"])
        self.edge_params_ref["min_alpha_points"] = getattr(self, "min_alpha_points", DEFAULT_SETTINGS["min_alpha_points"])
        self.edge_params_ref["max_alpha_points"] = getattr(self, "max_alpha_points", DEFAULT_SETTINGS["max_alpha_points"])
        self.edge_params_ref["enable_edge_aware"] = getattr(self, "enable_edge_aware_smoothing", DEFAULT_SETTINGS["enable_edge_aware_smoothing"])
        # Ensure thresholds are integers after getting the value
        self.edge_params_ref["depth_threshold1"] = int(getattr(self, "depth_edge_threshold1", DEFAULT_SETTINGS["depth_edge_threshold1"]))
        self.edge_params_ref["depth_threshold2"] = int(getattr(self, "depth_edge_threshold2", DEFAULT_SETTINGS["depth_edge_threshold2"]))
        self.edge_params_ref["rgb_threshold1"] = int(getattr(self, "rgb_edge_threshold1", DEFAULT_SETTINGS["rgb_edge_threshold1"]))
        self.edge_params_ref["rgb_threshold2"] = int(getattr(self, "rgb_edge_threshold2", DEFAULT_SETTINGS["rgb_edge_threshold2"]))
        
        self.edge_params_ref["influence"] = getattr(self, "edge_smoothing_influence", DEFAULT_SETTINGS["edge_smoothing_influence"])
        self.edge_params_ref["gradient_influence_scale"] = getattr(self, "gradient_influence_scale", DEFAULT_SETTINGS["gradient_influence_scale"])
        self.edge_params_ref["enable_sharpening"] = getattr(self, "enable_sharpening", DEFAULT_SETTINGS["enable_sharpening"])
        self.edge_params_ref["sharpness"] = getattr(self, "sharpness", DEFAULT_SETTINGS["sharpness"])
        
        # Thickening params
        self.edge_params_ref["enable_point_thickening"] = getattr(self, "enable_point_thickening", DEFAULT_SETTINGS["enable_point_thickening"])
        self.edge_params_ref["thickening_duplicates"] = int(getattr(self, "thickening_duplicates", DEFAULT_SETTINGS["thickening_duplicates"]))
        self.edge_params_ref["thickening_variance"] = getattr(self, "thickening_variance", DEFAULT_SETTINGS["thickening_variance"])
        self.edge_params_ref["thickening_depth_bias"] = getattr(self, "thickening_depth_bias", DEFAULT_SETTINGS["thickening_depth_bias"])
        
        # Projection params
        self.edge_params_ref["planar_projection"] = getattr(self, "planar_projection", DEFAULT_SETTINGS["planar_projection"])
        self.edge_params_ref["input_camera_fov"] = getattr(self, "input_camera_fov", DEFAULT_SETTINGS["input_camera_fov"])

        # --- Calibration Params (using getattr with appropriate defaults) ---
        self.edge_params_ref["apply_depth_bias"] = getattr(self, "apply_depth_bias", False) # Default to False if attr missing
        self.edge_params_ref["depth_bias_map"] = getattr(self, "depth_bias_map", None)     # Default to None if attr missing
        # --- End Calibration Params ---

        # --- Add Render Mode --- (New)
        self.edge_params_ref["render_mode"] = getattr(self, "render_mode", DEFAULT_SETTINGS["render_mode"])
        # --- End Add Render Mode ---

        # --- Bias Capture Trigger ---
        if hasattr(self, 'pending_bias_capture_request') and self.pending_bias_capture_request:
            self.edge_params_ref["trigger_bias_capture"] = self.pending_bias_capture_request
            self.pending_bias_capture_request = None # Consume the request for next update cycle
            print(f"DEBUG: _update_edge_params - Added trigger_bias_capture: {self.edge_params_ref['trigger_bias_capture']}")
        else:
            # Ensure the trigger is not sticky if not actively requested in this cycle
            self.edge_params_ref.pop("trigger_bias_capture", None)
        # --- End Bias Capture Trigger ---

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
            frame_read_delta_t, depth_process_delta_t, latency_ms, \
            newly_arrived_bias_map, main_screen_coeff_viz_numpy = latest_data # Unpack new coeff viz

            # Store the new main screen coefficient visualization content
            self.latest_main_screen_coeff_viz_content = main_screen_coeff_viz_numpy

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
                    
                    # Normalize to 0-255 range for uint8 conversion
                    min_val = np.min(depth_np)
                    max_val = np.max(depth_np)

                    if max_val > min_val: # Avoid division by zero if depth is flat
                        # Clip outliers for better normalization range (e.g., 1st and 99th percentile)
                        # This can help if a few extreme pixels skew the main range.
                        p_low = np.percentile(depth_np, 1)
                        p_high = np.percentile(depth_np, 99)
                        # Ensure p_low is not greater than p_high, can happen in very flat maps
                        if p_low >= p_high: p_low = min_val; p_high = max_val 
                        
                        # Use these percentiles for clipping and scaling if they are more robust
                        # If sticking to min/max, ensure max_val > min_val
                        # For now, let's try a robust scaling first before full percentile clipping.
                        # A simpler robust scaling: if max_val - min_val is tiny, treat as flat.
                        if (max_val - min_val) < 1e-5: # Threshold for being considered flat
                             depth_scaled = np.full_like(depth_np, 128) # Mid-gray for flat
                        else:
                             # Apply clipping before scaling to avoid extreme outliers dominating the range
                             depth_clipped = np.clip(depth_np, min_val, max_val) # Or use p_low, p_high here
                             depth_scaled = 255 * (depth_clipped - np.min(depth_clipped)) / (np.max(depth_clipped) - np.min(depth_clipped) + 1e-6) # add epsilon
                    else: # max_val <= min_val (flat map)
                        depth_scaled = np.zeros_like(depth_np) # Black for flat

                    depth_u8 = depth_scaled.astype(np.uint8)
                    
                    # Apply histogram equalization
                    if depth_u8.size > 0: # Ensure not empty
                         # Check if image is not flat gray, equalizeHist works best on images with some contrast
                         if np.any(depth_u8 != depth_u8[0,0] if depth_u8.ndim > 1 and depth_u8.shape[0]>0 and depth_u8.shape[1]>0 else depth_u8 != depth_u8[0] if depth_u8.ndim > 0 and depth_u8.shape[0]>0 else True):
                            depth_equalized_gray = cv2.equalizeHist(depth_u8)
                            self.latest_depth_map_viz = cv2.cvtColor(depth_equalized_gray, cv2.COLOR_GRAY2RGB)
                         else: # Image is flat, no need to equalize, just convert to RGB
                            self.latest_depth_map_viz = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)
                    else:
                         self.latest_depth_map_viz = None

                except Exception as e_depth_viz:
                    print(f"Error processing depth map for viz: {e_depth_viz}")
                    traceback.print_exc() # Add traceback
                    self.latest_depth_map_viz = None
            else:
                self.latest_depth_map_viz = None

            # Store latest vertex data for debug drawing (rays)
            self.latest_points_for_debug = None

            # Store raw depth tensor and keep history for spatiotemporal processing
            if depth_map_tensor is not None:
                try:
                    # Ensure on correct device
                    self.latest_depth_tensor = depth_map_tensor.to(self.device)
                    self.depth_history.append(self.latest_depth_tensor)
                except Exception as e_depth_hist:
                    print(f"Warning: could not append depth tensor to history: {e_depth_hist}")
                # Generate Wavelet/FFT map image for the debug window
                try:
                    # Stack history [T,1,H,W] and perform spatial wavelet + FFT
                    depth_stack = torch.stack(list(self.depth_history), dim=0).unsqueeze(1)
                    wavelet_window_size = getattr(self, 'wavelet_packet_window_size', DEFAULT_SETTINGS['wavelet_packet_window_size'])
                    wavelet_type = getattr(self, 'wavelet_packet_type', DEFAULT_SETTINGS['wavelet_packet_type'])
                    
                    J = int(math.log2(max(1, wavelet_window_size)))
                    min_dim_hist = min(depth_stack.shape[-2], depth_stack.shape[-1])
                    max_J_possible_hist = int(math.log2(min_dim_hist)) if min_dim_hist > 0 else 0
                    if J > max_J_possible_hist:
                        print(f"Warning: WPT J={J} for history stack too large ({depth_stack.shape}). Clamping to {max_J_possible_hist}.")
                        J = max_J_possible_hist

                    if J > 0:
                        dwt_op = DWTForward(J=J, wave=wavelet_type, mode='zero').to(self.device)
                        idwt_op = DWTInverse(wave=wavelet_type, mode='zero').to(self.device)
                        Yl, Yh = dwt_op(depth_stack) 
                        recon_spatial = idwt_op((Yl, Yh)).squeeze(1)  

                        fft_dims_temporal = tuple(range(recon_spatial.ndim))
                        freq = torch.fft.fftn(recon_spatial, dim=fft_dims_temporal, norm='ortho')
                        recon_full = torch.fft.ifftn(freq, dim=fft_dims_temporal, norm='ortho').real
                        
                        T_hist, H_hist, W_hist = recon_full.shape
                        latest_depth_frame_np = recon_full[-1].cpu().numpy()
                        oldest_depth_frame_np = recon_full[0].cpu().numpy() if T_hist >= 3 else latest_depth_frame_np.copy()
                        middle_depth_frame_np = recon_full[T_hist//2].cpu().numpy() if T_hist >= 3 else latest_depth_frame_np.copy()

                        def normalize_to_uint8_local(frame_np_func):
                            min_val, max_val = np.min(frame_np_func), np.max(frame_np_func)
                            norm_frame = (frame_np_func - min_val) / (max_val - min_val + 1e-7) if (max_val - min_val) > 1e-7 else np.zeros_like(frame_np_func)
                            return (norm_frame * 255).astype(np.uint8)

                        ch_r_uint8 = normalize_to_uint8_local(oldest_depth_frame_np)
                        ch_g_uint8 = normalize_to_uint8_local(middle_depth_frame_np)
                        ch_b_uint8 = normalize_to_uint8_local(latest_depth_frame_np)
                        temporal_composite_bgr = cv2.merge((ch_b_uint8, ch_g_uint8, ch_r_uint8))
                        laplacian = cv2.Laplacian(latest_depth_frame_np, cv2.CV_32F, ksize=3)
                        laplacian_abs_norm_uint8 = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        _, laplacian_edge_mask = cv2.threshold(laplacian_abs_norm_uint8, 50, 255, cv2.THRESH_BINARY)
                        enhanced_viz_bgr = temporal_composite_bgr.copy()
                        if laplacian_edge_mask.ndim == 2 and enhanced_viz_bgr.ndim == 3:
                             enhanced_viz_bgr[laplacian_edge_mask == 255] = [255, 255, 255]
                        self.latest_temporal_laplacian_viz_content = np.ascontiguousarray(enhanced_viz_bgr)
                    else: # J <= 0 for history stack
                        print("Warning: WPT levels J <= 0 for history stack. Cannot compute temporal laplacian wavelet map.")
                        self.latest_temporal_laplacian_viz_content = None # Fallback
                except Exception as e_wavelet_debug_viz:
                    print(f"Warning: failed to generate temporal laplacian wavelet map for debug: {e_wavelet_debug_viz}")
                    traceback.print_exc()
                    self.latest_temporal_laplacian_viz_content = None
            else: # No depth tensor for temporal processing
                 self.latest_temporal_laplacian_viz_content = None

            # Return processed data needed for vertex list update
            return vertices_data, colors_data, num_vertices_actual

        except Exception as e_unpack:
            print(f"ERROR in _process_queue_data: {e_unpack}") # Added ERROR prefix
            traceback.print_exc()
            # Also reset timing deltas on error? Maybe not necessary.
            return None, None, 0 # Return default values on error

    def _update_debug_textures(self):
        """Updates OpenGL textures for debug views from latest image arrays."""
        if not self.debug_textures_initialized:
            return

        try:
            # --- Update Camera Feed Texture --- 
            if self.latest_rgb_frame is not None and self.camera_texture is not None:
                frame_to_upload = self.latest_rgb_frame
                if frame_to_upload.ndim != 3:
                    # print(f"Warning: latest_rgb_frame has unexpected dimensions {frame_to_upload.shape}. Skipping camera texture update.")
                    frame_to_upload = None
                else:
                    h, w, c = frame_to_upload.shape
                    # gl_format = gl.GL_RGB # Default, not strictly needed as cvtColor handles output format for glTexImage2D
                    if self.input_mode == "Screen":
                        try:
                            if c == 4: frame_to_upload = cv2.cvtColor(frame_to_upload, cv2.COLOR_BGRA2RGB)
                            elif c == 3: frame_to_upload = cv2.cvtColor(frame_to_upload, cv2.COLOR_BGR2RGB) # Assume BGR if 3ch from screen
                            else: frame_to_upload = None
                            if frame_to_upload is not None: frame_to_upload = np.ascontiguousarray(frame_to_upload)
                        except cv2.error as e_cvt:
                            print(f"ERROR: Could not convert screen frame to RGB in _update_debug_textures: {e_cvt}.")
                            traceback.print_exc()
                            frame_to_upload = None 
                    elif c == 3: # Assume already RGB from UniK3D or previous cvtColor
                        frame_to_upload = np.ascontiguousarray(frame_to_upload)
                    elif c == 4: # E.g. RGBA image file
                         try:
                             frame_to_upload = cv2.cvtColor(frame_to_upload, cv2.COLOR_RGBA2RGB)
                             frame_to_upload = np.ascontiguousarray(frame_to_upload)
                         except cv2.error:
                             print("Failed to convert RGBA->RGB. Skipping camera texture update.")
                             frame_to_upload = None
                    else: # Unexpected channel count
                        frame_to_upload = None

                if frame_to_upload is not None:
                    h_f, w_f, _ = frame_to_upload.shape # Use different var names to avoid conflict 
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture)
                    if w_f != self.camera_texture_width or h_f != self.camera_texture_height:
                        # print(f"DEBUG UpdateTextures: Reallocating camera_texture from ({self.camera_texture_width}x{self.camera_texture_height}) to ({w_f}x{h_f})")
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_f, h_f, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None) 
                        self.camera_texture_width = w_f
                        self.camera_texture_height = h_f
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_f, h_f, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_to_upload.ctypes.data)

            # --- Update Depth Map Texture --- 
            if self.latest_depth_map_viz is not None and self.depth_texture is not None:
                depth_viz_cont = np.ascontiguousarray(self.latest_depth_map_viz)
                h_d, w_d, _ = depth_viz_cont.shape
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
                if w_d != self.depth_texture_width or h_d != self.depth_texture_height:
                    # print(f"DEBUG UpdateTextures: Reallocating depth_texture from ({self.depth_texture_width}x{self.depth_texture_height}) to ({w_d}x{h_d})")
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_d, h_d, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, None) 
                    self.depth_texture_width = w_d
                    self.depth_texture_height = h_d
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_d, h_d, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, depth_viz_cont.ctypes.data)

            # --- Update Edge Map Texture --- 
            if self.latest_edge_map is not None and self.edge_texture is not None:
                 edge_map_cont = np.ascontiguousarray(self.latest_edge_map)
                 h_e, w_e, _ = edge_map_cont.shape
                 gl.glBindTexture(gl.GL_TEXTURE_2D, self.edge_texture)
                 if w_e != self.edge_texture_width or h_e != self.edge_texture_height:
                    # print(f"DEBUG UpdateTextures: Reallocating edge_texture from ({self.edge_texture_width}x{self.edge_texture_height}) to ({w_e}x{h_e})")
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_e, h_e, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None) 
                    self.edge_texture_width = w_e
                    self.edge_texture_height = h_e
                 gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_e, h_e, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, edge_map_cont.ctypes.data)

            # --- Update Smoothing Map Texture --- 
            if self.latest_smoothing_map is not None and self.smoothing_texture is not None:
                 smoothing_map_cont = np.ascontiguousarray(self.latest_smoothing_map)
                 h_s, w_s, _ = smoothing_map_cont.shape
                 gl.glBindTexture(gl.GL_TEXTURE_2D, self.smoothing_texture)
                 if w_s != self.smoothing_texture_width or h_s != self.smoothing_texture_height:
                    # print(f"DEBUG UpdateTextures: Reallocating smoothing_texture from ({self.smoothing_texture_width}x{self.smoothing_texture_height}) to ({w_s}x{h_s})")
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_s, h_s, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None) 
                    self.smoothing_texture_width = w_s
                    self.smoothing_texture_height = h_s
                 gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_s, h_s, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, smoothing_map_cont.ctypes.data)

            # Update Wavelet/FFT main screen texture (self.wavelet_texture)
            # This texture is used by _render_wavelet_fft() when render_mode == 3.
            if self.wavelet_texture is not None: # Check if texture ID exists
                if hasattr(self, 'latest_main_screen_coeff_viz_content') and self.latest_main_screen_coeff_viz_content is not None:
                    print(f"NEW_LOG_WVLT_TEX: Data available for self.wavelet_texture. Shape: {self.latest_main_screen_coeff_viz_content.shape}. Attempting upload.")
                    main_screen_viz_data = np.ascontiguousarray(self.latest_main_screen_coeff_viz_content)
                    if main_screen_viz_data.ndim == 3 and main_screen_viz_data.shape[2] == 3:
                        h_main, w_main, _ = main_screen_viz_data.shape
                        gl.glBindTexture(gl.GL_TEXTURE_2D, self.wavelet_texture)
                        if w_main != self.wavelet_texture_width or h_main != self.wavelet_texture_height:
                            # print(f"DEBUG UpdateTextures: Reallocating wavelet_texture from ({self.wavelet_texture_width}x{self.wavelet_texture_height}) to ({w_main}x{h_main})")
                            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_main, h_main, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, None) 
                            self.wavelet_texture_width = w_main
                            self.wavelet_texture_height = h_main
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_main, h_main, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, main_screen_viz_data.ctypes.data) 
                    else:
                        print(f"NEW_LOG_WVLT_TEX: latest_main_screen_coeff_viz_content has unexpected shape {main_screen_viz_data.shape}. Not uploading to wavelet_texture.")
                else: 
                    print("NEW_LOG_WVLT_TEX: self.latest_main_screen_coeff_viz_content is None. Not updating self.wavelet_texture. Current content will persist or be black/purple.")
                    if self.wavelet_texture_width > 0 and self.wavelet_texture_height > 0: # Check if texture was ever initialized
                        gl.glBindTexture(gl.GL_TEXTURE_2D, self.wavelet_texture)
                        black_pixel = np.array([10,0,10], dtype=np.uint8) # Dark purple for visibility of blanking
                        black_tex_data = np.full((self.wavelet_texture_height, self.wavelet_texture_width, 3), black_pixel, dtype=np.uint8)
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self.wavelet_texture_width, self.wavelet_texture_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, black_tex_data.ctypes.data)

            # Update ImGui Wavelet/FFT debug texture (self.imgui_wavelet_debug_texture)
            if self.imgui_wavelet_debug_texture is not None: # Check if texture ID exists
                if hasattr(self, 'latest_temporal_laplacian_viz_content') and self.latest_temporal_laplacian_viz_content is not None:
                    # print(f"DEBUG UpdateTextures: Uploading latest_temporal_laplacian_viz_content (shape: {self.latest_temporal_laplacian_viz_content.shape}) to self.imgui_wavelet_debug_texture")
                    imgui_debug_viz_data = np.ascontiguousarray(self.latest_temporal_laplacian_viz_content)
                    if imgui_debug_viz_data.ndim == 3 and imgui_debug_viz_data.shape[2] == 3:
                        h_dbg, w_dbg, _ = imgui_debug_viz_data.shape
                        gl.glBindTexture(gl.GL_TEXTURE_2D, self.imgui_wavelet_debug_texture)
                        if w_dbg != self.imgui_wavelet_debug_texture_width or h_dbg != self.imgui_wavelet_debug_texture_height:
                            # print(f"DEBUG UpdateTextures: Reallocating imgui_wavelet_debug_texture from ({self.imgui_wavelet_debug_texture_width}x{self.imgui_wavelet_debug_texture_height}) to ({w_dbg}x{h_dbg})")
                            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_dbg, h_dbg, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, None) 
                            self.imgui_wavelet_debug_texture_width = w_dbg
                            self.imgui_wavelet_debug_texture_height = h_dbg
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w_dbg, h_dbg, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, imgui_debug_viz_data.ctypes.data)
                    # else:
                        # print(f"DEBUG UpdateTextures: latest_temporal_laplacian_viz_content has unexpected shape {imgui_debug_viz_data.shape}. Not uploading to imgui_wavelet_debug_texture.")
                else:
                    # print("DEBUG UpdateTextures: latest_temporal_laplacian_viz_content is None. Not updating ImGui debug texture.")
                    if self.imgui_wavelet_debug_texture_width > 0 and self.imgui_wavelet_debug_texture_height > 0:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, self.imgui_wavelet_debug_texture)
                        blue_pixel = np.array([0,0,20], dtype=np.uint8) # Dark blue for visibility
                        blue_tex_data = np.full((self.imgui_wavelet_debug_texture_height, self.imgui_wavelet_debug_texture_width, 3), blue_pixel, dtype=np.uint8)
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self.imgui_wavelet_debug_texture_width, self.imgui_wavelet_debug_texture_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, blue_tex_data.ctypes.data)

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) 

        except Exception as e_tex:
            print(f"ERROR in _update_debug_textures: {e_tex}") 
            traceback.print_exc()

    def _update_vertex_list(self, vertices_data, colors_data, num_vertices, view_matrix):
        """Updates the main point cloud vertex list, sorting if needed for Gaussian render mode."""
        # print(f"DEBUG: _update_vertex_list received {num_vertices} vertices.") 
        print(f"DEBUG: _update_vertex_list received {num_vertices} vertices.") # ADDED DEBUG PRINT
        self.current_point_count = num_vertices # Update point count regardless

        # Check if vertices_data and colors_data are None or empty *before* trying processing
        if vertices_data is None or colors_data is None or num_vertices <= 0:
            # If no valid vertices, clear the list
            if self.vertex_list:
                try: self.vertex_list.delete()
                except Exception: pass
                self.vertex_list = None
            self.current_point_count = 0
            return # Exit early

        # Continue only if we have valid data
        # --- Depth Sorting for Gaussian Splats --- 
        if self.render_mode == 2:
            try:
                # Reshape flat arrays back to N x 3
                vertices_np = vertices_data.reshape((num_vertices, 3))
                colors_np = colors_data.reshape((num_vertices, 3))
                # rays_np = rays_data.reshape((num_vertices, 3)) # REMOVED - Unused in current sorting

                # Transform world vertices to view space to get depth
                # Pyglet Mat4 can multiply Vec3s directly, but slower for large arrays.
                # Convert to numpy for faster batch transformation.
                # Add homogeneous coordinate (w=1)
                vertices_homogeneous = np.hstack((vertices_np, np.ones((num_vertices, 1))))
                # Convert pyglet Mat4 to numpy array (column-major to row-major if needed? Test)
                # Pyglet matrices are column-major, numpy expects row-major for standard @ op
                view_np = np.array(view_matrix).reshape((4, 4), order='F') # 'F' for Fortran/Column-major
                view_space_points = vertices_homogeneous @ view_np.T # Transpose view_np for correct multiplication

                # Extract Z depth (larger Z is farther in OpenGL view space)
                view_space_depths = view_space_points[:, 2]

                # Get indices to sort by depth, ascending (nearest first) - FLIPPED
                sort_indices = np.argsort(view_space_depths)

                # Sort vertices and colors
                vertices_sorted = vertices_np[sort_indices]
                colors_sorted = colors_np[sort_indices]

                # Flatten back for vertex list
                vertices_for_display = vertices_sorted.flatten()
                colors_for_display = colors_sorted.flatten()

            except Exception as e_sort:
                print(f"Error during point sorting: {e_sort}")
                traceback.print_exc()
                # Fallback to unsorted data if sorting fails
                vertices_for_display = vertices_data
                colors_for_display = colors_data
        else:
            # Use original data if not Gaussian mode
            vertices_for_display = vertices_data
            colors_for_display = colors_data
        # --- End Sorting ---

        # --- NaN/Inf Check ---
        invalid_vertices = np.isnan(vertices_for_display).any() or np.isinf(vertices_for_display).any()
        invalid_colors = np.isnan(colors_for_display).any() or np.isinf(colors_for_display).any()

        if invalid_vertices or invalid_colors:
            print(f"ERROR: Detected NaN/Inf in vertex data! (Vertices: {invalid_vertices}, Colors: {invalid_colors}). Skipping vertex list update for this frame.")
            # Optionally clear existing list to avoid rendering stale valid data
            if self.vertex_list:
                try: self.vertex_list.delete()
                except Exception: pass
                self.vertex_list = None
            self.current_point_count = 0
            return # Skip the update
        # --- End NaN/Inf Check ---

        # Store the points that will actually be used for rendering, for debug ray visualization
        # These points are already in the correct (+X Right, +Y Up, -Z Forward) world space
        # vertices_for_display is the (potentially sorted) and flattened array
        if vertices_for_display is not None and num_vertices > 0:
            try:
                self.latest_points_for_debug = vertices_for_display.reshape((num_vertices, 3))
            except ValueError as e_reshape:
                print(f"Warning: Could not reshape vertices_for_display for debug rays: {e_reshape}")
                self.latest_points_for_debug = None
        else:
            self.latest_points_for_debug = None # Ensure it's None if no valid points

        # Delete existing list if it exists
        if self.vertex_list:
            try: self.vertex_list.delete()
            except Exception: pass # Ignore error if already deleted
            self.vertex_list = None

        # Create new vertex list with potentially sorted data
        try:
            self.vertex_list = self.shader_program.vertex_list(
                num_vertices,
                gl.GL_POINTS,
                vertices=('f', vertices_for_display),
                colors=('f', colors_for_display)
            )
            self.frame_count_display += 1 # Increment display counter on successful update
        except Exception as e_create:
             print(f"ERROR creating vertex list: {e_create}") # Added ERROR prefix
             traceback.print_exc()
             self.vertex_list = None # Ensure list is None on error
             self.current_point_count = 0

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
                    # Wrap data processing and updates in try-except
                    try:
                        vertices_data, colors_data, num_vertices = self._process_queue_data(latest_data)

                        # Update debug textures based on the processed data
                        self._update_debug_textures()

                        # Get current view matrix for sorting
                        _, current_view = self.get_camera_matrices()

                        # Update the main vertex list (pass view matrix)
                        self._update_vertex_list(vertices_data, colors_data, num_vertices, current_view)
                    except Exception as e_main_update:
                        print(f"ERROR during main thread data update: {e_main_update}")
                        traceback.print_exc()
                        # Continue processing queue if possible, but skip this item's rendering logic

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
         # Adjust near/far planes to minimize clipping
         projection = Mat4.perspective_projection(self._aspect_ratio, z_near=0.001, z_far=10000.0, fov=60.0)
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
        settings_to_save = {key: getattr(self, key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
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
        loaded_settings = {}
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_settings = json.load(f)
                print(f"DEBUG: Settings loaded from {filename}")
            else:
                print(f"DEBUG: Settings file {filename} not found, using defaults.")
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            loaded_settings = {} # Ensure defaults are used on error

        # Initialize/update attributes from defaults first, then override with loaded values
        for key, default_val in DEFAULT_SETTINGS.items():
            loaded_val = loaded_settings.get(key, default_val)
            try:
                # Basic type conversion attempt
                if isinstance(default_val, bool): setattr(self, key, bool(loaded_val))
                elif isinstance(default_val, int): setattr(self, key, int(loaded_val))
                elif isinstance(default_val, float): setattr(self, key, float(loaded_val))
                else: setattr(self, key, loaded_val) # Assume string or correct type
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert loaded setting '{key}' ({loaded_val}), using default.")
                 setattr(self, key, default_val) # Fallback to default on conversion error

        # Ensure reference dicts/lists are updated/created *after* loading/defaults
        self.scale_factor_ref = [self.input_scale_factor]
        self._update_edge_params()
        self.update_playback_state()
        self.update_recording_state()


    def _browse_file(self):
        """Opens a file dialog to select a video, image, or directory."""
        root = tk.Tk()
        root.withdraw() # Hide the main tkinter window
        # Ask for directory first
        dir_path = filedialog.askdirectory(title="Select GLB Sequence Directory")
        if dir_path:
             file_path = dir_path # Treat directory as the selection
             print(f"DEBUG: Directory selected: {file_path}")
             self.input_filepath = file_path
             self.input_mode = "GLB Sequence" # Explicitly set mode
             self.status_message = f"Directory selected: {os.path.basename(file_path)}"
        else: # If directory selection cancelled, ask for file
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
                self.input_mode = "File" # Set mode to generic File
                self.status_message = f"File selected: {os.path.basename(file_path)}"
            else:
                print("DEBUG: File selection cancelled.")
                root.destroy()
                return # Exit if no file/dir selected

        root.destroy() # Destroy the tkinter instance

        # Reset playback state for new file/directory
        self.is_playing = True # Reset local UI control state
        # Reset relevant fields in the shared state dictionary
        self.playback_state["current_frame"] = 0
        self.playback_state["total_frames"] = 0
        self.playback_state["restart"] = False # Ensure restart flag is clear
        # Update the rest of the shared state from local attributes
        self.update_playback_state()
        # Restart inference thread with new source
        self.start_inference_thread()

    def _browse_media_file(self):
        """Opens a dialog to select a video or image file."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Video or Image File",
            filetypes=[("Media Files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                       ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                       ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All Files", "*.*")]
        )
        if not file_path:
            print("DEBUG: File selection cancelled.")
            root.destroy()
            return
        print(f"DEBUG: File selected: {file_path}")
        self.input_filepath = file_path
        self.input_mode = "File"
        self.status_message = f"File selected: {os.path.basename(file_path)}"
        root.destroy()

        # Reset playback state for new file
        self.is_playing = True
        self.playback_state["current_frame"] = 0
        self.playback_state["total_frames"] = 0
        self.playback_state["restart"] = False
        self.update_playback_state()
        self.start_inference_thread()

    def _browse_glb_sequence(self):
        """Opens a dialog to select a directory of GLB sequence."""
        root = tk.Tk()
        root.withdraw()
        dir_path = filedialog.askdirectory(title="Select GLB Sequence Directory")
        if not dir_path:
            print("DEBUG: Directory selection cancelled.")
            root.destroy()
            return
        print(f"DEBUG: Directory selected: {dir_path}")
        self.input_filepath = dir_path
        self.input_mode = "GLB Sequence"
        self.status_message = f"Directory selected: {os.path.basename(dir_path)}"
        root.destroy()

        # Reset playback state for new sequence
        self.is_playing = True
        self.playback_state["current_frame"] = 0
        self.playback_state["total_frames"] = 0
        self.playback_state["restart"] = False
        self.update_playback_state()
        self.start_inference_thread()

    def _switch_input_source(self, mode, filepath, monitor_index=0): # Added monitor_index
        """Switches input mode, resets state, and restarts inference."""
        self.input_mode = mode
        self.input_filepath = filepath
        # Store the chosen monitor index if mode is Screen
        if mode == "Screen":
            self.screen_capture_monitor_index = monitor_index
        self.status_message = f"{mode} selected"

        # Reset playback state for new source
        self.is_playing = True
        self.playback_state["current_frame"] = 0
        self.playback_state["total_frames"] = 0
        self.playback_state["restart"] = False
        self.update_playback_state()

        # Restart the inference thread with the new source and monitor index
        self.start_inference_thread()

    def _define_imgui_windows_and_widgets(self):
        """Defines all ImGui windows and widgets. Called between imgui.new_frame() and imgui.render()."""
        # --- Apply Custom Style --- 
        style = imgui.get_style()
        # Rounding
        style.window_rounding = 4.0
        style.frame_rounding = 4.0
        style.grab_rounding = 4.0
        # Padding
        style.window_padding = (8, 8)
        style.frame_padding = (6, 4)
        # Spacing
        style.item_spacing = (8, 4)
        style.item_inner_spacing = (4, 4)
        # Borders
        style.window_border_size = 1.0
        style.frame_border_size = 0.0 # Looks clean without frame borders
        style.popup_rounding = 4.0
        style.popup_border_size = 1.0
        # Alignment
        style.window_title_align = (0.5, 0.5) # Center title

        # --- End Dracula Style ---

        # --- Custom Style Settings (Apply after defaults) ---
        # Example: Increase rounding
        # style.window_rounding = 5.0
        # style.frame_rounding = 4.0
        # style.grab_rounding = 4.0

        # Colors (Pure Black Dark Mode - More Transparent)
        style.colors[imgui.COLOR_TEXT]                  = hex_to_imvec4("#E0E0E0") # Light Gray Text
        style.colors[imgui.COLOR_TEXT_DISABLED]         = hex_to_imvec4("#666666") # Dark Gray Disabled Text
        style.colors[imgui.COLOR_WINDOW_BACKGROUND]             = hex_to_imvec4("#050505", alpha=0.85) # Near Black, More Transparent
        style.colors[imgui.COLOR_CHILD_BACKGROUND]              = hex_to_imvec4("#0A0A0A", alpha=0.85) # Slightly Off-Black, More Transparent
        style.colors[imgui.COLOR_POPUP_BACKGROUND]              = hex_to_imvec4("#030303", alpha=0.90) # Darker Near Black Popup, More Transparent
        style.colors[imgui.COLOR_BORDER]                = hex_to_imvec4("#444444") # Slightly Lighter Border for visibility
        style.colors[imgui.COLOR_BORDER_SHADOW]         = hex_to_imvec4("#000000", alpha=0.0) # No shadow
        style.colors[imgui.COLOR_FRAME_BACKGROUND]              = hex_to_imvec4("#101010", alpha=0.80) # Very Dark Gray Frame, Transparent
        style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED]      = hex_to_imvec4("#181818", alpha=0.85)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE]       = hex_to_imvec4("#202020", alpha=0.90)
        style.colors[imgui.COLOR_TITLE_BACKGROUND]              = hex_to_imvec4("#050505", alpha=0.85) # Near Black Title, Transparent
        style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE]       = hex_to_imvec4("#151515", alpha=0.90) # Slightly lighter active title, Transparent
        style.colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED]    = hex_to_imvec4("#050505", alpha=0.75)
        style.colors[imgui.COLOR_MENUBAR_BACKGROUND]            = hex_to_imvec4("#050505")
        style.colors[imgui.COLOR_SCROLLBAR_BACKGROUND]          = hex_to_imvec4("#000000") # Black Scrollbar BG
        style.colors[imgui.COLOR_SCROLLBAR_GRAB]        = hex_to_imvec4("#444444") # Mid Gray Grab
        style.colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED]= hex_to_imvec4("#666666")
        style.colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = hex_to_imvec4("#888888")
        style.colors[imgui.COLOR_CHECK_MARK]            = hex_to_imvec4("#FFFFFF") # White Checkmark
        style.colors[imgui.COLOR_SLIDER_GRAB]           = hex_to_imvec4("#BBBBBB") # Light Gray Slider
        style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE]    = hex_to_imvec4("#FFFFFF") # White Active Slider
        style.colors[imgui.COLOR_BUTTON]                = hex_to_imvec4("#252525") # Dark Gray Button
        style.colors[imgui.COLOR_BUTTON_HOVERED]        = hex_to_imvec4("#353535")
        style.colors[imgui.COLOR_BUTTON_ACTIVE]         = hex_to_imvec4("#454545")
        style.colors[imgui.COLOR_HEADER]                = hex_to_imvec4("#181818") # Very Dark Gray Header
        style.colors[imgui.COLOR_HEADER_HOVERED]        = hex_to_imvec4("#282828")
        style.colors[imgui.COLOR_HEADER_ACTIVE]         = hex_to_imvec4("#383838")
        style.colors[imgui.COLOR_SEPARATOR]             = hex_to_imvec4("#333333") # Dark Gray Separator (matches border)
        style.colors[imgui.COLOR_SEPARATOR_HOVERED]     = hex_to_imvec4("#555555")
        style.colors[imgui.COLOR_SEPARATOR_ACTIVE]      = hex_to_imvec4("#777777")
        style.colors[imgui.COLOR_RESIZE_GRIP]           = hex_to_imvec4("#444444") # Mid Gray Grip
        style.colors[imgui.COLOR_RESIZE_GRIP_HOVERED]   = hex_to_imvec4("#666666")
        style.colors[imgui.COLOR_RESIZE_GRIP_ACTIVE]    = hex_to_imvec4("#888888")
        style.colors[imgui.COLOR_TAB]                   = hex_to_imvec4("#0A0A0A") # Slightly Off-Black Tab
        style.colors[imgui.COLOR_TAB_HOVERED]           = hex_to_imvec4("#1A1A1A")
        style.colors[imgui.COLOR_TAB_ACTIVE]            = hex_to_imvec4("#2A2A2A") # Light Gray Active Tab
        style.colors[imgui.COLOR_TAB_UNFOCUSED]         = hex_to_imvec4("#050505") # Near Black Unfocused Tab
        style.colors[imgui.COLOR_TAB_UNFOCUSED_ACTIVE]  = hex_to_imvec4("#151515")
        style.colors[imgui.COLOR_PLOT_LINES]            = hex_to_imvec4("#FFFFFF") # White Plot Lines
        style.colors[imgui.COLOR_PLOT_LINES_HOVERED]    = hex_to_imvec4("#DDDDDD")
        style.colors[imgui.COLOR_PLOT_HISTOGRAM]        = hex_to_imvec4("#BBBBBB") # Light Gray Histogram
        style.colors[imgui.COLOR_PLOT_HISTOGRAM_HOVERED]= hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_TABLE_HEADER_BACKGROUND] = hex_to_imvec4("#181818") # Dark Header for Table
        style.colors[imgui.COLOR_TABLE_BORDER_STRONG]   = hex_to_imvec4("#333333") # Dark Gray Strong Border
        style.colors[imgui.COLOR_TABLE_BORDER_LIGHT]    = hex_to_imvec4("#222222") # Very Dark Gray Light Border
        style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND]      = hex_to_imvec4("#0A0A0A") # Off-Black Row BG
        style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND_ALT]  = hex_to_imvec4("#101010") # Slightly Lighter Alt Row BG
        style.colors[imgui.COLOR_TEXT_SELECTED_BACKGROUND]      = hex_to_imvec4("#444444") # Mid Gray Selection BG
        style.colors[imgui.COLOR_DRAG_DROP_TARGET]      = hex_to_imvec4("#FFFFFF") # White Drag/Drop Target Highlight
        style.colors[imgui.COLOR_NAV_HIGHLIGHT]         = hex_to_imvec4("#FFFFFF") # White Nav Highlight
        style.colors[imgui.COLOR_NAV_WINDOWING_HIGHLIGHT]= hex_to_imvec4("#888888") # Gray Windowing Highlight
        style.colors[imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND]  = hex_to_imvec4("#000000", alpha=0.2) # Dim Black BG
        style.colors[imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND]   = hex_to_imvec4("#000000", alpha=0.5) # Darker Dim Black BG
        # --- End Custom Style ---
        
        # UI drawing logic will be moved here
        # --- ImGui Frame ---
        imgui.new_frame()

        # --- Screen Share Selection Popup ---
        if self.show_screen_share_popup:
            imgui.open_popup("Select Screen Monitor") # Trigger the popup

        # Center the popup
        main_viewport = imgui.get_main_viewport()
        popup_pos = (main_viewport.work_pos[0] + main_viewport.work_size[0] * 0.5,
                     main_viewport.work_pos[1] + main_viewport.work_size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], imgui.ALWAYS, 0.5, 0.5)
        imgui.set_next_window_size(400, 0) # Auto-adjust height

        if imgui.begin_popup_modal("Select Screen Monitor", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Select Monitor to Capture:")
            imgui.separator()

            monitors = []
            try:
                with mss.mss() as sct:
                    monitors = sct.monitors
            except Exception as e_mss:
                imgui.text_colored(f"Error getting monitors: {e_mss}", 1.0, 0.0, 0.0, 1.0)

            # Ensure temp index is valid before creating radio buttons
            if self.temp_monitor_index >= len(monitors):
                self.temp_monitor_index = 0 # Default to entire desktop if invalid

            # Option for entire desktop (monitor 0)
            changed_radio, self.temp_monitor_index = imgui.radio_button("Entire Desktop (All Monitors)", self.temp_monitor_index, 0)

            # Options for individual monitors (monitor 1, 2, ...)
            for i in range(1, len(monitors)):
                 mon = monitors[i]
                 label = f"Monitor {i} ({mon['width']}x{mon['height']} at {mon['left']},{mon['top']})"
                 changed_radio, self.temp_monitor_index = imgui.radio_button(label, self.temp_monitor_index, i)

            imgui.separator()
            if imgui.button("Start Sharing", width=120):
                self.screen_capture_monitor_index = self.temp_monitor_index # Store selection
                self.show_screen_share_popup = False # Hide flag
                self._switch_input_source("Screen", None, self.screen_capture_monitor_index) # Start thread
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                self.show_screen_share_popup = False # Hide flag
                imgui.close_current_popup()

            imgui.end_popup()
        # --- End Screen Share Popup ---


        # --- Main Controls Window ---
        imgui.set_next_window_position(10, 10, imgui.ONCE)
        imgui.set_next_window_size(350, self.height - 20, imgui.ONCE)
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
                imgui.text("Select Input Source:")
                # Combo selection for input source
                options = ["Live", "File", "GLB Sequence", "Screen"]
                labels = ["Live Camera", "Media File", "GLB Sequence", "Screen Share"]
                current_idx = options.index(self.input_mode) if self.input_mode in options else 0
                changed, idx = imgui.combo("Input Source", current_idx, labels)
                if changed and idx != current_idx:
                    mode = options[idx]
                    if mode == "Live":
                        self._switch_input_source("Live", None)
                    elif mode == "File":
                        self._browse_media_file()
                    elif mode == "GLB Sequence":
                        self._browse_glb_sequence()
                    else:
                        self._switch_input_source("Screen", None)
                # Show selected path for file or sequence
                if self.input_mode in ["File", "GLB Sequence"]:
                    imgui.text("Path: " + (self.input_filepath or "None"))
                elif self.input_mode == "Screen":
                    mon_label = "Entire Desktop" if self.screen_capture_monitor_index == 0 else f"Monitor {self.screen_capture_monitor_index}"
                    imgui.text(f"Capturing: {mon_label}")
                imgui.separator()
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

                imgui.separator()
                imgui.text("Ray Generation (Input Cam)")
                current_planar_projection = self.planar_projection if self.planar_projection is not None else DEFAULT_SETTINGS["planar_projection"]
                changed_planar_proj, current_planar_projection = imgui.checkbox("Planar Rays (Pinhole Model)", current_planar_projection)
                if changed_planar_proj:
                    self.planar_projection = current_planar_projection
                    self._update_edge_params()

                if self.planar_projection: # Only show FOV if planar is active
                    imgui.indent()
                    current_fov = self.input_camera_fov if self.input_camera_fov is not None else DEFAULT_SETTINGS["input_camera_fov"]
                    changed_fov, current_fov = imgui.slider_float("Input Camera FOV (Y)", current_fov, 10.0, 120.0, "%.1f deg")
                    if changed_fov:
                        self.input_camera_fov = current_fov
                        self._update_edge_params()
                    if imgui.button("Reset##InputFOV"):
                        self.input_camera_fov = DEFAULT_SETTINGS["input_camera_fov"]
                        self._update_edge_params()
                    imgui.unindent()

                imgui.end_tab_item()
            # --- End Processing Tab ---

            # --- Rendering Tab ---
            if imgui.begin_tab_item("Rendering")[0]:
                imgui.text("Point Style")
                changed_rm0 = imgui.radio_button("Square##RenderMode", self.render_mode == 0)
                if changed_rm0 and self.render_mode != 0: self.render_mode = 0; self._update_edge_params()
                imgui.same_line()
                changed_rm1 = imgui.radio_button("Circle##RenderMode", self.render_mode == 1)
                if changed_rm1 and self.render_mode != 1: self.render_mode = 1; self._update_edge_params()
                imgui.same_line()
                changed_rm2 = imgui.radio_button("Gaussian##RenderMode", self.render_mode == 2)
                if changed_rm2 and self.render_mode != 2: self.render_mode = 2; self._update_edge_params()
                imgui.same_line()
                changed_rm3 = imgui.radio_button("Wavelet/FFT##RenderMode", self.render_mode == 3)
                if changed_rm3 and self.render_mode != 3: self.render_mode = 3; self._update_edge_params()

                if self.render_mode == 2: # Gaussian Params
                    imgui.indent()
                    changed_falloff, self.falloff_factor = imgui.slider_float("Gaussian Falloff", self.falloff_factor, 0.1, 50.0)
                    if imgui.button("Reset##Falloff"): self.falloff_factor = DEFAULT_SETTINGS["falloff_factor"]
                    imgui.unindent()

                if self.render_mode == 3: # Wavelet/FFT Params
                    imgui.indent()
                    changed_wp_window, self.wavelet_packet_window_size = imgui.slider_int("Wavelet Window Size", self.wavelet_packet_window_size, 16, 256)
                    changed_wp_type, self.wavelet_packet_type = imgui.input_text("Wavelet Type", self.wavelet_packet_type, 64)
                    changed_fft_size, self.fft_size = imgui.slider_int("FFT Size", self.fft_size, 128, 2048)
                    changed_dmd_window, self.dmd_time_window = imgui.slider_int("DMD Time Window", self.dmd_time_window, 1, 100)
                    changed_cuda, self.enable_cuda_transform = imgui.checkbox("Enable CUDA Transform", self.enable_cuda_transform)
                    imgui.unindent()
                    # Reset wavelet history and cached output when parameters change
                    if changed_wp_window or changed_wp_type or changed_fft_size or changed_dmd_window or changed_cuda:
                        self.depth_history.clear()
                        self.latest_wavelet_map = None

                imgui.separator()
                imgui.text("Point Sizing (Input Camera Relative)")

                changed_boost, self.point_size_boost = imgui.slider_float("Base Size Boost", self.point_size_boost, 0.1, 50.0)
                if imgui.button("Reset##PointSize"): self.point_size_boost = DEFAULT_SETTINGS["point_size_boost"]

                # Slider for Input Resolution Scale Factor
                changed_scale, self.input_scale_factor = imgui.slider_float("Input Resolution Scale", self.input_scale_factor, 0.25, 4.0, "%.2f")
                if changed_scale: self.scale_factor_ref[0] = self.input_scale_factor # Update shared ref for thread
                if imgui.button("Reset##InputScale"):
                    self.input_scale_factor = DEFAULT_SETTINGS["input_scale_factor"]
                    self.scale_factor_ref[0] = self.input_scale_factor

                # Slider for Global Size Scale Factor
                changed_ssf, self.size_scale_factor = imgui.slider_float("Global Size Scale", self.size_scale_factor, 0.0001, 10.0, "%.4f")
                if changed_ssf:
                    print(f"DEBUG: Global sizeScaleFactor changed to {self.size_scale_factor}")

                imgui.separator()
                imgui.text("Inverse Square Law Params")
                # Slider for Depth Exponent (related to inverse square law)
                changed_depth_exp, self.depth_exponent = imgui.slider_float("Depth Exponent", self.depth_exponent, -4.0, 4.0, "%.2f")
                if changed_depth_exp:
                    print(f"DEBUG: depthExponent changed to {self.depth_exponent}")
                if imgui.button("Reset##DepthExponent"):
                    self.depth_exponent = DEFAULT_SETTINGS["depth_exponent"]

                imgui.separator()
                imgui.text("Color Adjustments")
                changed_sat, self.saturation = imgui.slider_float("Saturation", self.saturation, 0.0, 3.0)
                if imgui.button("Reset##Saturation"): self.saturation = DEFAULT_SETTINGS["saturation"]

                changed_brt, self.brightness = imgui.slider_float("Brightness", self.brightness, 0.0, 2.0)
                if imgui.button("Reset##Brightness"): self.brightness = DEFAULT_SETTINGS["brightness"]

                changed_con, self.contrast = imgui.slider_float("Contrast", self.contrast, 0.1, 3.0)
                if imgui.button("Reset##Contrast"): self.contrast = DEFAULT_SETTINGS["contrast"]

                imgui.separator()
                imgui.text("Viewer Camera Projection")
                current_use_orthographic = self.use_orthographic if self.use_orthographic is not None else DEFAULT_SETTINGS["use_orthographic"]
                changed_ortho, current_use_orthographic = imgui.checkbox("Use Orthographic Projection", current_use_orthographic)
                if changed_ortho:
                    self.use_orthographic = current_use_orthographic
                # No _update_edge_params needed, used directly in get_camera_matrices

                if self.use_orthographic:
                    imgui.indent()
                    current_ortho_size = self.orthographic_size if self.orthographic_size is not None else DEFAULT_SETTINGS["orthographic_size"]
                    changed_ortho_size, current_ortho_size = imgui.slider_float("Ortho Size (Half-Height)", current_ortho_size, 0.1, 100.0, "%.1f")
                    if changed_ortho_size:
                        self.orthographic_size = current_ortho_size
                    if imgui.button("Reset##OrthoSize"):
                        self.orthographic_size = DEFAULT_SETTINGS["orthographic_size"]
                    imgui.unindent()

                imgui.end_tab_item()
            # --- End Rendering Tab ---

            # --- Debug Tab ---
            if imgui.begin_tab_item("Debug")[0]:
                imgui.text("Show Debug Views:")
                _, self.show_camera_feed = imgui.checkbox("Camera Feed", self.show_camera_feed)
                _, self.show_depth_map = imgui.checkbox("Depth Map", self.show_depth_map)
                _, self.show_edge_map = imgui.checkbox("Edge Map", self.show_edge_map)
                _, self.show_smoothing_map = imgui.checkbox("Smoothing Map", self.show_smoothing_map)
                _, self.show_wavelet_map = imgui.checkbox("Wavelet/FFT Map", self.show_wavelet_map)

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

                imgui.separator()
                imgui.text("Visualize Sizing Calculations:")
                # Use changed flags to ensure only one is active?
                # For now, use simple checkboxes.
                changed_dbg_dist, self.debug_show_input_distance = imgui.checkbox("Input Distance", self.debug_show_input_distance)
                changed_dbg_diam, self.debug_show_raw_diameter = imgui.checkbox("Raw Diameter (Pre-Density)", self.debug_show_raw_diameter)
                changed_dbg_dens, self.debug_show_density_factor = imgui.checkbox("Density Factor", self.debug_show_density_factor)
                changed_dbg_size, self.debug_show_final_size = imgui.checkbox("Final Size (gl_PointSize)", self.debug_show_final_size)

                # Logic to ensure only one is active at a time (optional)
                if changed_dbg_dist and self.debug_show_input_distance:
                    self.debug_show_raw_diameter = False
                    self.debug_show_density_factor = False
                    self.debug_show_final_size = False
                elif changed_dbg_diam and self.debug_show_raw_diameter:
                    self.debug_show_input_distance = False
                    self.debug_show_density_factor = False
                    self.debug_show_final_size = False
                elif changed_dbg_dens and self.debug_show_density_factor:
                     self.debug_show_input_distance = False
                     self.debug_show_raw_diameter = False
                     self.debug_show_final_size = False
                elif changed_dbg_size and self.debug_show_final_size:
                     self.debug_show_input_distance = False
                     self.debug_show_raw_diameter = False
                     self.debug_show_density_factor = False

                imgui.separator()
                imgui.text("Visualize Geometry:")
                _, self.debug_show_world_axes = imgui.checkbox("World Axes##Geom", self.debug_show_world_axes)
                _, self.debug_show_input_frustum = imgui.checkbox("Input Frustum##Geom", self.debug_show_input_frustum)
                _, self.debug_show_viewer_frustum = imgui.checkbox("Viewer Frustum##Geom", self.debug_show_viewer_frustum)
                _, self.debug_show_input_rays = imgui.checkbox("Input Rays (Sampled)##Geom", self.debug_show_input_rays)

                imgui.separator()
                imgui.text("Depth Bias Calibration (Dev)")
                if imgui.button("Capture Bias (View Flat Surface)"):
                    # self._capture_depth_bias() # Old direct call removed
                    self.pending_bias_capture_request = "mean_plane"
                    self._update_edge_params() # Signal inference thread
                    self.status_message = "Bias capture requested (Mean Plane Ref)..."
                    print("UI: Bias capture requested (Mean Plane Ref).")

                # Use local variable for checkbox to avoid issues if self.apply_depth_bias is None initially
                current_apply_bias = self.apply_depth_bias if self.apply_depth_bias is not None else False
                changed_apply_bias, current_apply_bias = imgui.checkbox("Apply Depth Bias Correction", current_apply_bias)
                if changed_apply_bias:
                    self.apply_depth_bias = current_apply_bias
                    self._update_edge_params() # Signal change to inference thread
                
                if self.depth_bias_map is not None:
                    imgui.same_line()
                    imgui.text_colored("Bias Captured", 0.0, 1.0, 0.0, 1.0) # Green text
                elif self.latest_depth_tensor_for_calib is not None:
                     imgui.same_line()
                     imgui.text_colored("Raw Depth Captured, Bias Calc Failed?", 1.0, 1.0, 0.0, 1.0) # Yellow text
                else: # If no bias map and no record of capture attempt
                    imgui.same_line()
                    imgui.text_colored("Bias not captured", 0.7, 0.7, 0.7, 1.0) # Grey text

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
            imgui.set_next_window_position(10, self.height - 250, imgui.ONCE) 
            # Corrected: Pass self.show_camera_feed as the p_open argument. closable is implicit.
            is_open, self.show_camera_feed = imgui.begin("Camera Feed", self.show_camera_feed)
            if is_open: 
                available_width = imgui.get_content_region_available()[0]
                orig_h, orig_w, _ = self.latest_rgb_frame.shape
                aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect_ratio
                imgui.image(self.camera_texture, display_width, display_height)
            imgui.end() 

        if self.show_depth_map and self.depth_texture and self.latest_depth_map_viz is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(340, self.height - 250, imgui.ONCE) 
            is_open, self.show_depth_map = imgui.begin("Depth Map", self.show_depth_map)
            if is_open:
                available_width = imgui.get_content_region_available()[0]
                orig_h, orig_w, _ = self.latest_depth_map_viz.shape
                aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect_ratio
                imgui.image(self.depth_texture, display_width, display_height)
            imgui.end()

        if self.show_edge_map and self.edge_texture and self.latest_edge_map is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(10, self.height - 500, imgui.ONCE) 
            is_open, self.show_edge_map = imgui.begin("Edge Map", self.show_edge_map)
            if is_open:
                available_width = imgui.get_content_region_available()[0]
                orig_h, orig_w, _ = self.latest_edge_map.shape
                aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect_ratio
                imgui.image(self.edge_texture, display_width, display_height)
            imgui.end()

        if self.show_smoothing_map and self.smoothing_texture and self.latest_smoothing_map is not None:
             imgui.set_next_window_size(320, 240, imgui.ONCE)
             imgui.set_next_window_position(340, self.height - 500, imgui.ONCE) 
             is_open, self.show_smoothing_map = imgui.begin("Smoothing Alpha Map", self.show_smoothing_map)
             if is_open:
                 available_width = imgui.get_content_region_available()[0]
                 orig_h, orig_w, _ = self.latest_smoothing_map.shape
                 aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                 display_width = available_width
                 display_height = display_width / aspect_ratio
                 imgui.image(self.smoothing_texture, display_width, display_height)
             imgui.end()

        # Ensure self.imgui_wavelet_debug_texture is checked for None before use
        if self.show_wavelet_map and hasattr(self, 'imgui_wavelet_debug_texture') and self.imgui_wavelet_debug_texture is not None and hasattr(self, 'latest_temporal_laplacian_viz_content') and self.latest_temporal_laplacian_viz_content is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(660, self.height - 250, imgui.ONCE)
            is_open, self.show_wavelet_map = imgui.begin("Wavelet/FFT Output", self.show_wavelet_map)
            if is_open:
                available_width = imgui.get_content_region_available()[0]
                tex_to_display = self.imgui_wavelet_debug_texture 
                w_to_display = self.imgui_wavelet_debug_texture_width
                h_to_display = self.imgui_wavelet_debug_texture_height
                
                aspect = float(w_to_display) / float(h_to_display) if h_to_display > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect
                imgui.image(tex_to_display, display_width, display_height)
            imgui.end()
        # --- End Debug View Windows ---


        # Render ImGui
        # Ensure correct GL state for ImGui rendering (standard alpha blend)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST) # ImGui draws in 2D

        # Bind our default VAO so ImGui's renderer captures a valid VAO
        try:
            gl.glBindVertexArray(self._default_vao)
        except Exception:
            pass

        imgui.render()
        # Draw ImGui, catching any GL errors from buffer binding
        try:
            self.imgui_renderer.render(imgui.get_draw_data())
        except Exception as e:
            print(f"Warning: ImGui renderer error (ignored): {e}")
        # Re-bind our default VAO so ImGui's glBindVertexArray(last) will be valid
        try:
            gl.glBindVertexArray(self._default_vao)
        except Exception:
            pass
        # --- End ImGui Frame ---

        # Auto-enable the Wavelet overlay when in Wavelet/FFT render mode
        if self.render_mode == 3:
            self.show_wavelet_map = True

    def _render_scene(self):
        """Renders the 3D point cloud scene (modes 0, 1, 2). Mode 3 is handled by _render_wavelet_fft."""

        print(f"DEBUG RenderScene: Called for mode {self.render_mode}. This should not be mode 3.")

        if self.render_mode == 3:
            print("ERROR: _render_scene was called with render_mode 3. This should be handled by _render_wavelet_fft in on_draw.")
            return # Should not happen if on_draw is correct

        if not self.vertex_list:
            return

        projection, current_view = self.get_camera_matrices()
        try:
            self.shader_program.use()
            self.shader_program['projection'] = projection
            self.shader_program['view'] = current_view
            self.shader_program['viewportSize'] = (float(self.width), float(self.height))
            self.shader_program['inputScaleFactor'] = self.input_scale_factor
            self.shader_program['pointSizeBoost'] = self.point_size_boost
            self.shader_program['renderMode'] = self.render_mode
            self.shader_program['falloffFactor'] = self.falloff_factor
            self.shader_program['saturation'] = self.saturation
            self.shader_program['brightness'] = self.brightness
            self.shader_program['contrast'] = self.contrast
            self.shader_program['sharpness'] = self.sharpness if self.enable_sharpening else 1.0

            current_input_fov = self.input_camera_fov if self.input_camera_fov is not None else DEFAULT_SETTINGS["input_camera_fov"]
            fov_rad = math.radians(current_input_fov)
            input_h = float(self.latest_rgb_frame.shape[0]) if self.latest_rgb_frame is not None else float(self.height)
            input_focal = (input_h * 0.5) / math.tan(fov_rad * 0.5)
            self.shader_program['inputFocal'] = input_focal

            self.shader_program['sizeScaleFactor'] = self.size_scale_factor
            self.shader_program['minPointSize'] = self.min_point_size
            self.shader_program['enableMaxSizeClamp'] = self.enable_max_size_clamp
            self.shader_program['maxPointSize'] = self.max_point_size
            self.shader_program['depthExponent'] = self.depth_exponent
            
            # Pass planar_projection state to shader for conditional density compensation
            self.shader_program['planarProjectionActive'] = self.planar_projection if self.planar_projection is not None else DEFAULT_SETTINGS["planar_projection"]

            self.shader_program['debug_show_input_distance'] = self.debug_show_input_distance
            self.shader_program['debug_show_raw_diameter'] = self.debug_show_raw_diameter
            self.shader_program['debug_show_density_factor'] = self.debug_show_density_factor
            self.shader_program['debug_show_final_size'] = self.debug_show_final_size

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
            elif self.render_mode == 3: # Wavelet/FFT
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
            # --- End setup for Draw 3D Splats ---

            # Draw the splats: ensure vertex_list.draw is called for all point modes
            print(f"DEBUG RenderScene: self.render_mode is {self.render_mode} before check. Drawing points.")
            if self.vertex_list: # Check vertex_list exists
                self.vertex_list.draw(gl.GL_POINTS) # Always draw points from vertex_list

        except Exception as e_render:
             print(f"ERROR during _render_scene (shader use or draw): {e_render}")
             traceback.print_exc()
        finally:
            try:
                self.shader_program.stop()
            except Exception as e_stop_shader:
                print(f"Warning: Error stopping shader program in _render_scene: {e_stop_shader}")

    def on_draw(self):
        # Clear the main window buffer
        gl.glClearColor(0.0, 0.0, 0.0, 1.0) 
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self.render_mode == 3: 
            # For Wavelet/FFT mode, render the 2D texture full-screen via _render_wavelet_fft()
            try:
                print("DEBUG on_draw: render_mode is 3, calling _render_wavelet_fft().")
                self._render_wavelet_fft() 
            except Exception as e_on_draw_wfft:
                print(f"ERROR in on_draw calling _render_wavelet_fft for mode 3: {e_on_draw_wfft}")
                traceback.print_exc()
        else: 
            # For other modes (0, 1, 2), render the 3D point cloud scene via _render_scene()
            try:
                print(f"DEBUG on_draw: render_mode is {self.render_mode}, calling _render_scene().")
                self._render_scene()
            except Exception as e_on_draw_render:
                print(f"ERROR in on_draw calling _render_scene: {e_on_draw_render}")
                traceback.print_exc()

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

        # Draw the ImGui UI using the helper method (wrap in try-except)
        try:
            self._define_imgui_windows_and_widgets()
        except Exception as e_ui_define:
            print(f"ERROR during ImGui widget definition: {e_ui_define}")
            traceback.print_exc()
            # Fall through to the finally block to attempt to end the frame
        finally:
            # This block ensures ImGui's frame is properly ended and rendered,
            # critical for preventing assertion errors on the next frame.

            # Set GL state for ImGui rendering
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(0)  # Ensure no custom shaders are active for ImGui
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0) # Ensure rendering to default framebuffer
            # Active texture unit 0 is usually default for ImGui
            gl.glActiveTexture(gl.GL_TEXTURE0)
            # Unbind any textures from units ImGui might use, though it should manage its own
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # For sampler2D
            # Pyglet's ImGui renderer might expect a VAO to be bound or might bind its own.
            # Binding VAO 0 is generally safest.
            gl.glBindVertexArray(0)

            imgui.render()  # Prepare ImGui draw data

            try:
                # This call includes ImGui::EndFrame implicitly via the renderer
                self.imgui_renderer.render(imgui.get_draw_data())
            except Exception as e_render_imgui:
                print(f"Warning: self.imgui_renderer.render error: {e_render_imgui}")
                traceback.print_exc()
                # If self.imgui_renderer.render fails catastrophically, ImGui's EndFrame might not
                # have been called. The assertion error on the *next* frame would indicate this.
                # Calling imgui.end_frame() here directly can be risky if the renderer
                # is also trying to do it and only partially failed.
                # For now, we rely on the renderer to call EndFrame.

            # Restore some GL state if necessary for subsequent non-ImGui Pyglet drawing,
            # though in this app, ImGui is usually the last thing drawn in on_draw.
            # gl.glEnable(gl.GL_DEPTH_TEST) # Re-enable if other 2D pyglet stuff follows

    def _draw_debug_geometry(self):
        """Draws various debug geometries like frustums and axes using shaders."""
        proj_mat, view_mat = self.get_camera_matrices()
        batch = pyglet.graphics.Batch() # Create a batch for drawing debug elements
        debug_elements = [] # List to hold vertex_list objects

        # --- Setup GL State (Shader independent) --- 
        # Store previous state manually
        prev_line_width = gl.GLfloat()
        gl.glGetFloatv(gl.GL_LINE_WIDTH, prev_line_width)
        depth_test_enabled = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        blend_enabled = gl.glIsEnabled(gl.GL_BLEND)
        depth_mask_enabled = gl.GLboolean()
        gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK, depth_mask_enabled)

        # Configure GL state for debug overlays
        gl.glDisable(gl.GL_BLEND)   # Ensure lines are solid
        gl.glDisable(gl.GL_DEPTH_TEST) # DRAW ON TOP! Disable depth test
        gl.glDepthMask(gl.GL_FALSE)     # Don't write to depth buffer
        gl.glLineWidth(2.0)

        # --- World Axes --- 
        if self.debug_show_world_axes:
            axis_length = 1.0
            vertices = [ 0.0, 0.0, 0.0, axis_length, 0.0, 0.0, # X
                         0.0, 0.0, 0.0, 0.0, axis_length, 0.0, # Y
                         0.0, 0.0, 0.0, 0.0, 0.0, axis_length] # Z
            colors =   [ 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, # Red
                         0.0, 1.0, 0.0, 0.0, 1.0, 0.0, # Green
                         0.0, 0.0, 1.0, 0.0, 0.0, 1.0] # Blue
            debug_elements.append(self.debug_shader_program.vertex_list(
                3 * 2, gl.GL_LINES, batch=batch,
                position=('f', vertices),
                color=('f', colors)
            ))

        # --- Input Frustum --- (Yellow)
        if self.debug_show_input_frustum:
            try:
                current_input_fov_geom = self.input_camera_fov if self.input_camera_fov is not None else DEFAULT_SETTINGS["input_camera_fov"]
                input_fov_rad = math.radians(current_input_fov_geom)
                aspect = 1.0
                if self.latest_rgb_frame is not None and self.latest_rgb_frame.shape[0] > 0:
                    aspect = self.latest_rgb_frame.shape[1] / self.latest_rgb_frame.shape[0]
                near_plane, far_plane = 0.1, 20.0
                h_near = 2.0 * near_plane * math.tan(input_fov_rad / 2.0); w_near = h_near * aspect
                h_far = 2.0 * far_plane * math.tan(input_fov_rad / 2.0); w_far = h_far * aspect
                corners = [ # Model space corners (Y down, -Z forward)
                    ( w_near/2,  h_near/2, -near_plane), (-w_near/2,  h_near/2, -near_plane),
                    (-w_near/2, -h_near/2, -near_plane), ( w_near/2, -h_near/2, -near_plane),
                    ( w_far/2,  h_far/2, -far_plane), (-w_far/2,  h_far/2, -far_plane),
                    (-w_far/2, -h_far/2, -far_plane), ( w_far/2, -h_far/2, -far_plane) ]
                # Convert to render space (+Y up, -Z forward)
                corners_render = [(c[0], -c[1], -c[2]) for c in corners]
                # Define lines for frustum
                indices = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
                vertices = [coord for i in indices for coord in corners_render[i]]
                num_verts = len(indices)
                colors = [1.0, 1.0, 0.0] * num_verts # Yellow for all lines
                debug_elements.append(self.debug_shader_program.vertex_list(
                    num_verts, gl.GL_LINES, batch=batch,
                    position=('f', vertices),
                    color=('f', colors)
                ))
            except Exception as e_inf: print(f"Error drawing input frustum: {e_inf}")

        # --- Viewer Frustum --- (Magenta)
        if self.debug_show_viewer_frustum:
            try:
                vp_mat = proj_mat @ view_mat
                inv_vp_mat = vp_mat.inverse()
                ndc_corners = [
                    (-1,-1,-1), ( 1,-1,-1), ( 1, 1,-1), (-1, 1,-1),
                    (-1,-1, 1), ( 1,-1, 1), ( 1, 1, 1), (-1, 1, 1) ]
                world_corners = []
                for ndc in ndc_corners:
                    ndc_vec = Vec3(ndc[0], ndc[1], ndc[2])
                    world_h = inv_vp_mat @ ndc_vec
                    w = world_h._cdata[3]
                    if abs(w) < 1e-6: w = 1.0
                    world_corners.append((world_h._cdata[0]/w, world_h._cdata[1]/w, world_h._cdata[2]/w))
                indices = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
                vertices = [coord for i in indices for coord in world_corners[i]]
                num_verts = len(indices)
                colors = [1.0, 0.0, 1.0] * num_verts # Magenta
                debug_elements.append(self.debug_shader_program.vertex_list(
                    num_verts, gl.GL_LINES, batch=batch,
                    position=('f', vertices),
                    color=('f', colors)
                ))
            except Exception as e_vf: print(f"Error drawing viewer frustum: {e_vf}")

        # --- Input Rays --- (Cyan, Sampled)
        if self.debug_show_input_rays and self.latest_points_for_debug is not None and len(self.latest_points_for_debug) > 0:
            try:
                num_rays_to_draw = 100
                num_points = len(self.latest_points_for_debug)
                step = max(1, num_points // num_rays_to_draw) if num_points > 0 else 1
                sampled_points = self.latest_points_for_debug[::step]
                vertices = []
                origin = (0.0, 0.0, 0.0)
                for point_coords in sampled_points:
                    vertices.extend(origin)
                    vertices.extend(point_coords)
                num_verts = len(sampled_points) * 2
                colors = [0.0, 1.0, 1.0] * num_verts # Cyan
                debug_elements.append(self.debug_shader_program.vertex_list(
                    num_verts, gl.GL_LINES, batch=batch,
                    position=('f', vertices),
                    color=('f', colors)
                ))
            except Exception as e_rays: print(f"Error drawing input rays: {e_rays}")

        # --- Draw all debug elements --- 
        if debug_elements:
            self.debug_shader_program.use()
            self.debug_shader_program['projection'] = proj_mat
            self.debug_shader_program['view'] = view_mat
            batch.draw()
            self.debug_shader_program.stop()

        # --- Restore GL State --- 
        gl.glLineWidth(prev_line_width.value)
        if depth_test_enabled:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)
        if blend_enabled:
            gl.glEnable(gl.GL_BLEND)
        else:
            gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(depth_mask_enabled.value)

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
        if hasattr(self, 'wavelet_texture') and self.wavelet_texture:
             try: gl.glDeleteTextures(1, self.wavelet_texture)
             except: pass
        if hasattr(self, 'imgui_wavelet_debug_texture') and self.imgui_wavelet_debug_texture:
            try: gl.glDeleteTextures(1, self.imgui_wavelet_debug_texture)
            except: pass
        # --- ImGui Cleanup ---
        if hasattr(self, 'imgui_renderer') and self.imgui_renderer:
            self.imgui_renderer.shutdown()
        # --- End ImGui Cleanup ---
        super().on_close()

    def _render_wavelet_fft(self):
        print("DEBUG RenderWavelet: _render_wavelet_fft called.") # Existing Print
        """Renders the self.wavelet_texture as a full-screen quad."""

        # The WPT/FFT processing to generate the content for self.wavelet_texture
        # is now handled in _process_queue_data and _update_debug_textures.
        # This function just draws the prepared texture.

        if self.texture_shader_program and self.texture_quad_vao and hasattr(self, 'wavelet_texture') and self.wavelet_texture:
            # print("DEBUG RenderWavelet: Resources OK, attempting to draw quad.") # Already have this
            try:
                # GL state for 2D texture rendering
                gl.glDisable(gl.GL_DEPTH_TEST)
                # gl.glEnable(gl.GL_BLEND) # Temporarily disable blend for testing
                gl.glDisable(gl.GL_BLEND) # Ensure it's off for opaque quad
                # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

                self.texture_shader_program.use()
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.wavelet_texture)
                self.texture_shader_program['fboTexture'] = 0 # Sampler uniform - Restore this line

                gl.glBindVertexArray(self.texture_quad_vao)
                gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4) # Draw the quad
                print("DEBUG RenderWavelet: glDrawArrays called for wavelet quad.") # New Print

                # Cleanup
                gl.glBindVertexArray(0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                self.texture_shader_program.stop()
                gl.glEnable(gl.GL_DEPTH_TEST) # Re-enable depth test for other rendering
                gl.glEnable(gl.GL_BLEND) # Re-enable blend for subsequent ImGui/overlay drawing

            except Exception as e_render_quad:
                print(f"Error rendering wavelet texture quad: {e_render_quad}")
                traceback.print_exc()
        else:
            # This fallback means something is wrong with texture/shader setup or wavelet_texture isn't ready
            # For now, just print a warning. Avoid drawing points.
            print("DEBUG RenderWavelet: Resources MISSING for wavelet quad render.") # New Print
            # self.vertex_list.draw(gl.GL_POINTS) # Explicitly DO NOT draw points here

    def _capture_depth_bias(self):
        """Captures the current depth map and calculates an additive bias map
           relative to a smoothed version of the captured depth."""
        # THIS METHOD IS NO LONGER CALLED DIRECTLY.
        # Its logic has been moved to the inference thread triggered by self.pending_bias_capture_request.
        # Kept as a placeholder or for potential future re-use of the smoothing strategy.
        print("WARNING: _capture_depth_bias (smoothed reference) was called but is deprecated. Bias capture now happens in inference thread via trigger.")
        
        # Original smoothed reference logic (for reference, but not active):
        # print("Attempting to capture depth bias (Smoothed Reference Method)...")
        # current_raw_depth = self.latest_depth_tensor 
        
        # if current_raw_depth is None:
        #     print("ERROR: No depth tensor available to capture bias.")
        #     self.status_message = "Error: No depth tensor for bias capture."
        #     self.depth_bias_map = None
        #     self.latest_depth_tensor_for_calib = None
        #     self._update_edge_params()
        #     return

        # try:
        #     D_captured = current_raw_depth.clone().to(self.device).float() 
        #     self.latest_depth_tensor_for_calib = D_captured

        #     if D_captured.numel() == 0:
        #          print("ERROR: Captured depth map is empty.")
        #          self.status_message = "Error: Captured depth map empty."
        #          self.depth_bias_map = None
        #          self.latest_depth_tensor_for_calib = None
        #          self._update_edge_params()
        #          return
            
        #     # --- Calculate Smoothed Version ---
        #     D_captured_np = D_captured.cpu().numpy()
        #     smooth_kernel_size = 51 
        #     if smooth_kernel_size > D_captured_np.shape[0] or smooth_kernel_size > D_captured_np.shape[1]:
        #         print(f"WARNING: Bias smooth kernel ({smooth_kernel_size}) > image ({D_captured_np.shape}). Clamping kernel size.")
        #         smooth_kernel_size = min(D_captured_np.shape[0], D_captured_np.shape[1])
        #         if smooth_kernel_size % 2 == 0: smooth_kernel_size -= 1 
        #         smooth_kernel_size = max(1, smooth_kernel_size) 
            
        #     try:
        #          D_smoothed_np = cv2.GaussianBlur(D_captured_np, (smooth_kernel_size, smooth_kernel_size), 0)
        #          D_smoothed = torch.from_numpy(D_smoothed_np).to(self.device).type_as(D_captured)
        #     except cv2.error as e_blur:
        #          print(f"ERROR: GaussianBlur failed during bias capture: {e_blur}. Kernel Size: {smooth_kernel_size}. Depth shape: {D_captured_np.shape}")
        #          self.status_message = "Error: Bias blur failed."
        #          self.depth_bias_map = None 
        #          self.latest_depth_tensor_for_calib = None
        #          self._update_edge_params()
        #          return
        #     # --- End Calculate Smoothed Version ---

        #     self.depth_bias_map = D_captured - D_smoothed

        #     print(f"  Captured Depth Range: [{torch.min(D_captured):.4f}, {torch.max(D_captured):.4f}], Mean: {torch.mean(D_captured.float()):.4f}")
        #     print(f"  Smoothed Depth Range: [{torch.min(D_smoothed):.4f}, {torch.max(D_smoothed):.4f}], Mean: {torch.mean(D_smoothed.float()):.4f}")
        #     print(f"  Calculated Bias Map Range: [{torch.min(self.depth_bias_map):.4f}, {torch.max(self.depth_bias_map):.4f}], Mean: {torch.mean(self.depth_bias_map.float()):.4f}")

        #     self.apply_depth_bias = True
        #     self.status_message = "Depth bias captured (Smoothed Ref)."
        #     print("Depth bias captured and calculated (Smoothed Ref). Apply checkbox enabled.")
        #     self._update_edge_params()

        # except Exception as e_bias:
        #     print(f"ERROR calculating depth bias: {e_bias}")
        #     traceback.print_exc()
        #     self.status_message = "Error calculating depth bias."
        #     self.depth_bias_map = None
        #     self.latest_depth_tensor_for_calib = None
        #     self._update_edge_params()
        pass # End of deprecated method


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
