import json
import os
import traceback

# Default settings dictionary (copied from live_slam_viewer.py)
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
    "input_scale_factor": 0.5,  # Reduce from 1.0 to 0.5 for better performance
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
    "show_input_fps_overlay": True,
    "show_depth_fps_overlay": True,
    "show_latency_overlay": True,
    "show_dropped_frames_overlay": True,
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
    "planar_projection": False,       # Toggle to use planar projection instead of spherical rays for point generation
    "use_orthographic": False,        # Toggle between perspective and orthographic viewer projection
    "orthographic_size": 5.0,         # Ortho camera half-height (world units)
    "apply_depth_bias": False,        # Toggle for applying depth bias correction
}

def load_settings_from_file(app_window, filename="viewer_settings.json"):
    """Loads settings from a JSON file or uses defaults, updating the app_window attributes."""
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
        loaded_settings = {}

    for key, default_val in DEFAULT_SETTINGS.items():
        loaded_val = loaded_settings.get(key, default_val)
        try:
            if isinstance(default_val, bool): setattr(app_window, key, bool(loaded_val))
            elif isinstance(default_val, int): setattr(app_window, key, int(loaded_val))
            elif isinstance(default_val, float): setattr(app_window, key, float(loaded_val))
            else: setattr(app_window, key, loaded_val)
        except (ValueError, TypeError):
             print(f"Warning: Could not convert loaded setting '{key}' ({loaded_val}), using default.")
             setattr(app_window, key, default_val)

    # Ensure reference dicts/lists are updated/created *after* loading/defaults
    app_window.scale_factor_ref = [app_window.input_scale_factor]
    app_window._update_edge_params() # Assumes app_window has this method
    app_window.update_playback_state() # Assumes app_window has this method
    app_window.update_recording_state() # Assumes app_window has this method
    app_window.status_message = "Settings loaded."


def save_settings_to_file(app_window, filename="viewer_settings.json"):
    """Saves current settings from app_window to a JSON file."""
    settings_to_save = {key: getattr(app_window, key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
    try:
        with open(filename, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
        print(f"DEBUG: Settings saved to {filename}")
        app_window.status_message = f"Settings saved to {filename}"
    except Exception as e:
        print(f"Error saving settings: {e}")
        app_window.status_message = "Error saving settings."

def reset_app_settings(app_window):
    """Resets app_window settings to default values."""
    print("DEBUG: Resetting settings to default.")
    for key, value in DEFAULT_SETTINGS.items():
        setattr(app_window, key, value)
    
    app_window.scale_factor_ref = [app_window.input_scale_factor]
    app_window._update_edge_params()
    app_window.update_playback_state()
    app_window.update_recording_state()
    app_window.status_message = "Settings reset to defaults." 