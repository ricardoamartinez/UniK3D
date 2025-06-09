#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UniK3D Live SLAM Viewer - Main Entry Point

This script launches the UniK3D Live SLAM Viewer, which provides a real-time
visualization of the UniK3D model's depth and 3D point cloud predictions.

The viewer supports:
- Live camera input
- Video file playback
- Image file processing
- GLB sequence playback
- Screen capture
- Recording to GLB files
- Various visualization and processing options
- Wavelet/FFT visualization modes
- Depth bias calibration

Usage:
    python live_slam_viewer.py [--model MODEL_NAME] [--interval INTERVAL]

Arguments:
    --model MODEL_NAME    Name of the UniK3D model to use (default: unik3d-vitl)
    --interval INTERVAL   Run inference every N frames (default: 10)
"""

import argparse
import os
import sys
import traceback
import pyglet

# Add src directory to Python path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import the main viewer window from the unik3d_viewer package
from unik3d_viewer.main_viewer import LiveViewerWindow

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UniK3D Live SLAM Viewer")
    parser.add_argument(
        "--model",
        type=str,
        default="unik3d-vitl",
        help="UniK3D model name (unik3d-vits, unik3d-vitb, unik3d-vitl)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Run inference every N frames (default: 10)",
    )
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()

    print(f"Starting UniK3D Live SLAM Viewer with model: {args.model}")
    print(f"Inference interval: {args.interval} frames")

    # Try to use a high-quality OpenGL configuration with multisampling
    try:
        config = pyglet.gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True)
        window = LiveViewerWindow(
            model_name=args.model,
            inference_interval=args.interval,
            disable_point_smoothing=False,  # Use default smoothing settings
            width=1280,
            height=720,
            caption='UniK3D Live SLAM Viewer',
            resizable=True,
            vsync=True,
            config=config
        )
        print("Using high-quality OpenGL configuration with multisampling.")
    except pyglet.window.NoSuchConfigException:
        # Fall back to default configuration if high-quality is not available
        print("Warning: High-quality OpenGL configuration not available. Falling back to default.")
        window = LiveViewerWindow(
            model_name=args.model,
            inference_interval=args.interval,
            disable_point_smoothing=False,  # Use default smoothing settings
            width=1280,
            height=720,
            caption='UniK3D Live SLAM Viewer',
            resizable=True,
            vsync=True
        )

    try:
        # Start the Pyglet event loop
        pyglet.app.run()
    except Exception as e:
        print("\n--- Uncaught Exception ---")
        traceback.print_exc()
        print("------------------------")
    finally:
        # Ensure ImGui context is destroyed if app exits unexpectedly
        if hasattr(window, 'ui_manager') and window.ui_manager and hasattr(window.ui_manager, 'imgui_renderer'):
            try:
                window.ui_manager.imgui_renderer.shutdown()
                print("DEBUG: ImGui renderer shutdown.")
            except Exception as e_shutdown:
                print(f"Error during ImGui shutdown: {e_shutdown}")

if __name__ == "__main__":
    main()
