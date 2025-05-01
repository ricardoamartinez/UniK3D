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
    "size_scale_factor": 0.001, # Scale factor for z^2 point sizing
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
    "input_camera_fov": 60.0,  # FOV of the input camera in degrees
    "min_point_size": 1.0,        # Min pixel size
    "enable_max_size_clamp": False, # Enable max size clamp?
    "max_point_size": 50.0,       # Max pixel size (if clamped)
    # --- Point Thickening --- 
    "enable_point_thickening": False,
    "thickening_duplicates": 4,     # Num duplicates per point (total points = original * (1 + duplicates))
    "thickening_variance": 0.01,   # StdDev of random perturbation
    "thickening_depth_bias": 0.5,   # Strength of backward push along ray (0=none)
    "depth_exponent": 2.0,        # Exponent for depth-based sizing (2.0=z^2, 1.0=z, -2.0=1/z^2)
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
vertex_source = """#version 150 core
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
    uniform float sizeScaleFactor;  // Tunable scale factor for z^2 sizing
    uniform float minPointSize;     // Minimum point size in pixels
    uniform float maxPointSize;     // Maximum point size in pixels (if clamp enabled)
    uniform bool enableMaxSizeClamp;// Toggle for max size clamp
    // uniform float depthExponent;    // Exponent applied to depth for sizing (Removed for automatic sizing)

    void main() {
        // Transform to view and clip space
        vec4 viewPos = view * vec4(vertices, 1.0);
        vec4 clipPos = projection * viewPos;
        gl_Position = clipPos;
        vertex_colors = colors;

        // --- Point Sizing based on INPUT Camera Distance (Inverse Square Law) and Density Compensation ---
        // Calculate distance from the INPUT camera (origin in model space)
        float inputDistance = length(vertices); // USE INPUT DISTANCE
        inputDistance = max(inputDistance, 0.0001); // Avoid division by zero

        // Point size proportional to the square of the INPUT distance (inverse square law)
        // We use an exponent of -2.0 for inverse square.
        // The base size is adjusted by inputScaleFactor, sizeScaleFactor, and pointSizeBoost.
        float baseSize = inputFocal * inputScaleFactor; // Base size related to input camera properties
        // Use inputDistance for scaling instead of viewerDistance
        float diameter = 2.0 * baseSize * sizeScaleFactor * pointSizeBoost * pow(inputDistance, -2.0); // USE INPUT DISTANCE

        // Density compensation based on the original 360 projection (assuming spherical)
        // Calculate normalized ray direction from INPUT camera to the point
        vec3 inputRay = normalize(vertices); // USE INPUT RAY
        // The Y-component of the normalized ray is sin(latitude).
        // Note: vertices.y is DOWN in the original coordinate system before view transform.
        // We use inputRay.y directly.
        float cosInputLatitude = sqrt(1.0 - clamp(inputRay.y * inputRay.y, 0.0, 1.0)); // USE INPUT RAY
        // Point size should be inversely proportional to the density, which is proportional to cos(latitude)
        float densityCompensationFactor = 1.0 / max(1e-5, cosInputLatitude); // USE INPUT LATITUDE

        diameter *= densityCompensationFactor; // Apply density compensation

        // Apply minimum and optional maximum size clamp
        float finalSize = max(diameter, minPointSize);
        if (enableMaxSizeClamp) {
            finalSize = min(finalSize, maxPointSize);
        }
        gl_PointSize = finalSize;

        // Assign debug values
        debug_inputDistance_out = inputDistance;
        debug_rawDiameter_out = 2.0 * baseSize * sizeScaleFactor * pointSizeBoost * pow(inputDistance, -2.0); // Raw diameter before density comp
        debug_densityFactor_out = densityCompensationFactor;
        debug_finalSize_out = finalSize;
        // --- End Automatic Point Size Calculation ---
    }
"""

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
    return points_xyz_np_processed, colors_np_processed, num_vertices, scaled_depth_map_for_queue, edge_map_viz, smoothing_map_viz, new_prev_depth_map, new_smoothed_mean_depth, new_smoothed_points_xyz # Note: returning original smoothed points for next frame's smoothing


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
                   frame_read_delta_t, depth_process_delta_t, latency_ms):
    """Puts the processed data into the queue for the main thread."""
    if not data_queue.full():
            # Add frame_count back to the tuple if needed by receiver (currently not)
            # For now, keep it out of the tuple to match _process_queue_data
            data_queue.put((vertices_flat, colors_flat, num_vertices,
                            rgb_frame_orig,
                            scaled_depth_map_for_queue,
                            edge_map_viz,
                            smoothing_map_viz,
                            t_capture,
                            sequence_frame_index,
                            video_total_frames,
                            current_recorded_count,
                            frame_read_delta_t, depth_process_delta_t, latency_ms))
    else:
        # Use frame_count in the warning message again
        print(f"Warning: Viewer queue full, dropping frame {frame_count}.")
        # Optionally put a status message instead of dropping silently
        # data_queue.put(("status", f"Viewer queue full, dropping frame {frame_count}"))


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
                prev_depth_map, smoothed_mean_depth, smoothed_points_xyz = \
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
        if not hasattr(self, 'scale_factor_ref') or self.scale_factor_ref is None:
            self.scale_factor_ref = [self.input_scale_factor]
        if not hasattr(self, 'edge_params_ref') or not self.edge_params_ref: # Keep ref name for dict itself
             self._update_edge_params()
        if not hasattr(self, 'playback_state') or not self.playback_state:
             self.update_playback_state() # Initialize playback state
        if not hasattr(self, 'recording_state') or not self.recording_state:
             self.update_recording_state() # Initialize recording state
        # Ensure screen capture index attribute exists
        if not hasattr(self, 'screen_capture_monitor_index'):
            self.screen_capture_monitor_index = DEFAULT_SETTINGS['screen_capture_monitor_index']

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
        # Add thickening params
        self.edge_params_ref["enable_point_thickening"] = self.enable_point_thickening
        self.edge_params_ref["thickening_duplicates"] = int(self.thickening_duplicates) # Ensure int
        self.edge_params_ref["thickening_variance"] = self.thickening_variance
        self.edge_params_ref["thickening_depth_bias"] = self.thickening_depth_bias

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

            # Store latest vertex data for debug drawing (rays)
            self.latest_points_for_debug = None

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
            # Handle potential BGRA from MSS or BGR from OpenCV
            if self.latest_rgb_frame is not None and self.camera_texture is not None:
                frame_to_upload = self.latest_rgb_frame
                # Check dimensions first
                if frame_to_upload.ndim != 3:
                    print(f"Warning: latest_rgb_frame has unexpected dimensions {frame_to_upload.shape}. Skipping camera texture update.")
                    frame_to_upload = None
                else:
                    h, w, c = frame_to_upload.shape
                    gl_format = gl.GL_RGB # Default

                    # Explicitly handle Screen mode frame format (likely BGRA from mss)
                    if self.input_mode == "Screen":
                        try:
                            if c == 4: # BGRA
                                frame_to_upload = cv2.cvtColor(frame_to_upload, cv2.COLOR_BGRA2RGB)
                            elif c == 3: # Maybe BGR?
                                print("Warning: Screen capture frame has 3 channels, assuming BGR and converting to RGB.")
                                frame_to_upload = cv2.cvtColor(frame_to_upload, cv2.COLOR_BGR2RGB)
                            else:
                                 print(f"Warning: Screen capture frame has unexpected channel count {c}. Skipping camera texture update.")
                                 frame_to_upload = None
                            
                            if frame_to_upload is not None:
                                # Ensure contiguous after conversion
                                frame_to_upload = np.ascontiguousarray(frame_to_upload)
                        except cv2.error as e_cvt:
                            print(f"ERROR: Could not convert screen frame to RGB in _update_debug_textures: {e_cvt}. Skipping camera texture update.")
                            traceback.print_exc()
                            frame_to_upload = None # Skip upload
                    # Handle non-Screen mode (likely RGB from cvtColor earlier)
                    elif c == 3:
                        gl_format = gl.GL_RGB
                        # Ensure contiguous just in case
                        frame_to_upload = np.ascontiguousarray(frame_to_upload)
                    # Handle unexpected formats in non-screen mode
                    elif c == 4:
                         print("Warning: Unexpected 4-channel frame in non-screen mode. Trying RGBA->RGB.")
                         try:
                             frame_to_upload = cv2.cvtColor(frame_to_upload, cv2.COLOR_RGBA2RGB)
                             gl_format = gl.GL_RGB
                             frame_to_upload = np.ascontiguousarray(frame_to_upload)
                         except cv2.error:
                             print("Failed to convert RGBA->RGB. Skipping camera texture update.")
                             frame_to_upload = None
                    else:
                         print(f"Warning: Unexpected frame channel count ({c}). Skipping camera texture update.")
                         frame_to_upload = None

                # Perform the upload if frame is valid
                if frame_to_upload is not None:
                    h, w, _ = frame_to_upload.shape 
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture)
                    # Check if texture size needs reallocation
                    if w != self.camera_texture_width or h != self.camera_texture_height:
                        print(f"DEBUG: Reallocating camera_texture from ({self.camera_texture_width}x{self.camera_texture_height}) to ({w}x{h})")
                        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None) # Reallocate
                        self.camera_texture_width = w
                        self.camera_texture_height = h
                    # Upload data (use glTexSubImage2D if texture already allocated, but glTexImage2D is simpler if realloc just happened)
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_to_upload.ctypes.data)

            # --- Update Depth Map Texture --- 
            # (Already handles BGR from applyColorMap)
            if self.latest_depth_map_viz is not None and self.depth_texture is not None:
                depth_viz_cont = np.ascontiguousarray(self.latest_depth_map_viz)
                h, w, _ = depth_viz_cont.shape
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
                if w != self.depth_texture_width or h != self.depth_texture_height:
                    print(f"DEBUG: Reallocating depth_texture from ({self.depth_texture_width}x{self.depth_texture_height}) to ({w}x{h})")
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, None) # Reallocate (format BGR)
                    self.depth_texture_width = w
                    self.depth_texture_height = h
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, depth_viz_cont.ctypes.data)

            # --- Update Edge Map Texture --- 
            # (Should be RGB)
            if self.latest_edge_map is not None and self.edge_texture is not None:
                 edge_map_cont = np.ascontiguousarray(self.latest_edge_map)
                 h, w, _ = edge_map_cont.shape
                 gl.glBindTexture(gl.GL_TEXTURE_2D, self.edge_texture)
                 if w != self.edge_texture_width or h != self.edge_texture_height:
                    print(f"DEBUG: Reallocating edge_texture from ({self.edge_texture_width}x{self.edge_texture_height}) to ({w}x{h})")
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None) # Reallocate (format RGB)
                    self.edge_texture_width = w
                    self.edge_texture_height = h
                 gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, edge_map_cont.ctypes.data)

            # --- Update Smoothing Map Texture --- 
            # (Should be RGB)
            if self.latest_smoothing_map is not None and self.smoothing_texture is not None:
                 smoothing_map_cont = np.ascontiguousarray(self.latest_smoothing_map)
                 h, w, _ = smoothing_map_cont.shape
                 gl.glBindTexture(gl.GL_TEXTURE_2D, self.smoothing_texture)
                 if w != self.smoothing_texture_width or h != self.smoothing_texture_height:
                    print(f"DEBUG: Reallocating smoothing_texture from ({self.smoothing_texture_width}x{self.smoothing_texture_height}) to ({w}x{h})")
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None) # Reallocate (format RGB)
                    self.smoothing_texture_width = w
                    self.smoothing_texture_height = h
                 gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, smoothing_map_cont.ctypes.data)

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # Unbind

        except Exception as e_tex:
            print(f"ERROR in _update_debug_textures: {e_tex}") # Added ERROR prefix
            traceback.print_exc()
            # Optionally disable debug views on error?

    def _update_vertex_list(self, vertices_data, colors_data, num_vertices, view_matrix):
        """Updates the main point cloud vertex list, sorting if needed for Gaussian render mode."""
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

    def _draw_ui(self):
        """Draws the ImGui user interface."""
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
                    changed_falloff, self.falloff_factor = imgui.slider_float("Gaussian Falloff", self.falloff_factor, 0.1, 50.0)
                    if imgui.button("Reset##Falloff"): self.falloff_factor = DEFAULT_SETTINGS["falloff_factor"]
                    imgui.unindent()

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

                # Changed to SliderFloat with larger range
                size_scale_changed, self.size_scale_factor = imgui.slider_float(
                    "Size Scale (z^2)", self.size_scale_factor, 0.0, 1.0, "%.4f"
                )
                if imgui.button("Reset##SizeScale"):
                    self.size_scale_factor = DEFAULT_SETTINGS["size_scale_factor"]

                imgui.separator()
                imgui.text("Inverse Square Law Params")
                # Slider for Depth Exponent (related to inverse square law)
                changed_depth_exp, self.depth_exponent = imgui.slider_float("Depth Exponent", self.depth_exponent, -4.0, 4.0, "%.2f")
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
                # Calculate aspect ratio preserving size
                available_width = imgui.get_content_region_available()[0]
                orig_h, orig_w, _ = self.latest_rgb_frame.shape
                aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect_ratio
                imgui.image(self.camera_texture, display_width, display_height)
            imgui.end()

        if self.show_depth_map and self.depth_texture and self.latest_depth_map_viz is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(340, self.height - 250, imgui.ONCE) # Position next to camera feed
            is_open, self.show_depth_map = imgui.begin("Depth Map", closable=True)
            if is_open:
                # Calculate aspect ratio preserving size
                available_width = imgui.get_content_region_available()[0]
                orig_h, orig_w, _ = self.latest_depth_map_viz.shape
                aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect_ratio
                imgui.image(self.depth_texture, display_width, display_height)
            imgui.end()

        if self.show_edge_map and self.edge_texture and self.latest_edge_map is not None:
            imgui.set_next_window_size(320, 240, imgui.ONCE)
            imgui.set_next_window_position(10, self.height - 500, imgui.ONCE) # Position below camera feed
            is_open, self.show_edge_map = imgui.begin("Edge Map", closable=True)
            if is_open:
                # Calculate aspect ratio preserving size
                available_width = imgui.get_content_region_available()[0]
                orig_h, orig_w, _ = self.latest_edge_map.shape
                aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                display_width = available_width
                display_height = display_width / aspect_ratio
                imgui.image(self.edge_texture, display_width, display_height)
            imgui.end()

        if self.show_smoothing_map and self.smoothing_texture and self.latest_smoothing_map is not None:
             imgui.set_next_window_size(320, 240, imgui.ONCE)
             imgui.set_next_window_position(340, self.height - 500, imgui.ONCE) # Position below depth map
             is_open, self.show_smoothing_map = imgui.begin("Smoothing Alpha Map", closable=True)
             if is_open:
                 # Calculate aspect ratio preserving size
                 available_width = imgui.get_content_region_available()[0]
                 orig_h, orig_w, _ = self.latest_smoothing_map.shape
                 aspect_ratio = float(orig_w) / float(orig_h) if orig_h > 0 else 1.0
                 display_width = available_width
                 display_height = display_width / aspect_ratio
                 imgui.image(self.smoothing_texture, display_width, display_height)
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
            try:
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

                # Compute and pass input camera focal length (pixel units) for sizing
                if self.latest_rgb_frame is not None:
                    input_h = float(self.latest_rgb_frame.shape[0])
                else:
                    input_h = float(self.height) # Fallback if no frame yet
                fov_rad = math.radians(self.input_camera_fov)
                input_focal = (input_h * 0.5) / math.tan(fov_rad * 0.5)
                self.shader_program['inputFocal'] = input_focal

                # Pass the tunable size scale factor
                self.shader_program['sizeScaleFactor'] = self.size_scale_factor

                # Pass new min/max size clamp uniforms
                self.shader_program['minPointSize'] = self.min_point_size
                self.shader_program['enableMaxSizeClamp'] = self.enable_max_size_clamp
                self.shader_program['maxPointSize'] = self.max_point_size

                # Pass depth exponent for sizing (Removed for automatic sizing)
                # self.shader_program['depthExponent'] = self.depth_exponent

                # Pass debug uniforms
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

                # Draw the splats
                # try:
                self.vertex_list.draw(gl.GL_POINTS)
                # except Exception as e:
                #     print(f"Error during vertex_list.draw: {e}")
                # finally:
                # Restore default-ish state after drawing splats
                gl.glDepthMask(gl.GL_TRUE)
                # Keep blend enabled for ImGui/UI, but set to standard alpha
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                self.shader_program.stop()
            except Exception as e_render:
                 print(f"ERROR during _render_scene: {e_render}")
                 traceback.print_exc()
                 # Attempt to stop shader program even if draw failed
                 try: self.shader_program.stop()
                 except: pass
                 # Restore GL state
                 gl.glDepthMask(gl.GL_TRUE)
                 gl.glEnable(gl.GL_BLEND)
                 gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # --- End Draw 3D Splats ---

    def on_draw(self):
        # Clear the main window buffer
        # gl.glClearColor(0.1, 0.1, 0.1, 1.0) # Old gray background
        gl.glClearColor(0.0, 0.0, 0.0, 1.0) # New pure black background
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Render the 3D scene (wrap in try-except)
        try:
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
            self._draw_ui()
        except Exception as e_on_draw_ui:
             print(f"ERROR in on_draw calling _draw_ui: {e_on_draw_ui}")
             traceback.print_exc()

        # --- Draw Debug Geometry --- (New Call)
        try:
            self._draw_debug_geometry()
        except Exception as e_on_draw_debug:
            print(f"ERROR in on_draw calling _draw_debug_geometry: {e_on_draw_debug}")
            traceback.print_exc()

        # Final GL state restoration (if needed, though ImGui usually handles its own)
        # gl.Enable(gl.GL_DEPTH_TEST) # Already enabled after overlay

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
                input_fov_rad = math.radians(self.input_camera_fov)
                aspect = 1.0
                if self.latest_rgb_frame is not None and self.latest_rgb_frame.shape[0] > 0:
                    aspect = self.latest_rgb_frame.shape[1] / self.latest_rgb_frame.shape[0]
                else: aspect = self.width / self.height
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
