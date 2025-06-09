#!/usr/bin/env python3
"""
Main entry point for the Live UniK3D SLAM Viewer application.

This script sets up and launches the main application window for real-time
3D scene reconstruction using UniK3D models with live camera input.
"""

import time
startup_time = time.time()
print(f"🔍 DEBUG: main_app.py starting at {startup_time}")

print(f"🔍 DEBUG: Starting imports at {time.time() - startup_time:.3f}s")
import os
import sys
import argparse
import traceback
import pyglet
import imgui

# Add src directory to Python path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import viewer
from unik3d_viewer.main_viewer import LiveViewerWindow

import_time = time.time()
print(f"🔍 DEBUG: All imports completed in {import_time - startup_time:.3f}s")

def main():
    main_start = time.time()
    print(f"🔍 DEBUG: main() function starting at {main_start - startup_time:.3f}s")
    
    parser_start = time.time()
    parser = argparse.ArgumentParser(description="Live SLAM viewer using UniK3D.")
    parser.add_argument("--model", type=str, default="unik3d-vitl", help="Name of the UniK3D model.")
    parser.add_argument("--interval", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target FPS for inference.")
    args = parser.parse_args()
    parser_time = time.time()
    print(f"🔍 DEBUG: Argument parsing took {parser_time - parser_start:.3f}s")

    config_start = time.time()
    config = pyglet.gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True)
    config_time = time.time()
    print(f"🔍 DEBUG: GL config creation took {config_time - config_start:.3f}s")
    
    window_start = time.time()
    print(f"🔍 DEBUG: Creating LiveViewerWindow at {window_start - startup_time:.3f}s")
    try:
        window = LiveViewerWindow(model_name=args.model, inference_interval=args.interval,
                                  target_inference_fps=args.target_fps,
                                  width=1280, height=720, caption='UniK3D Live Viewer',
                                  resizable=True, vsync=False, config=config)
    except pyglet.window.NoSuchConfigException:
        print("⚠️ Warning: Desired GL config not available. Falling back to default.")
        fallback_start = time.time()
        window = LiveViewerWindow(model_name=args.model, inference_interval=args.interval,
                                  target_inference_fps=args.target_fps,
                                  width=1280, height=720, caption='UniK3D Live Viewer',
                                  resizable=True, vsync=False)
        fallback_time = time.time()
        print(f"🔍 DEBUG: Fallback window creation took {fallback_time - fallback_start:.3f}s")
    
    window_time = time.time()
    print(f"🔍 DEBUG: Window creation completed in {window_time - window_start:.3f}s")
    print(f"🔍 DEBUG: Total time to window ready: {window_time - startup_time:.3f}s")
    
    print(f"🔍 DEBUG: Starting pyglet.app.run() at {time.time() - startup_time:.3f}s")
    try:
        pyglet.app.run()
    except Exception as e_run:
        print(f"Unhandled exception during pyglet.app.run(): {e_run}")
        traceback.print_exc()
    finally:
        print("Application run loop finished or exited via exception.")
        if hasattr(window, 'renderer') and window.renderer: 
            window.renderer.cleanup()
        if hasattr(window, 'ui_manager') and window.ui_manager: 
            window.ui_manager.shutdown()
        print("Exiting application.")


if __name__ == '__main__':
    main() 