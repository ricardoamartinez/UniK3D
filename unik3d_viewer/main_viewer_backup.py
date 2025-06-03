import pyglet
import pyglet.gl as gl
import numpy as np
import os
import argparse
import math
import threading
import time
import queue
import torch
import traceback
import imgui
import cv2
from collections import deque # Keep for LiveViewerWindow.depth_history
import tkinter as tk # Keep for filedialog in main app
from tkinter import filedialog # Keep for filedialog in main app
import ctypes # Added based on review of rendering.py usage
import concurrent.futures
from threading import Lock, Event, Thread

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    print("ERROR: pytorch_wavelets not installed in main_viewer; functionality will be limited.")
    DWTForward = None
    DWTInverse = None

# UniK3D Model
from unik3d.models import UniK3D

# Pyglet imports
from pyglet.math import Mat4, Vec3
from pyglet.window import key

# Local module imports from unik3d_viewer package
from .config import DEFAULT_SETTINGS, load_settings_from_file, save_settings_to_file, reset_app_settings
from .camera_handler import Camera
from .inference_logic import inference_thread_func
from .rendering import Renderer
from .ui import UIManager

class GPUResourceManager:
    """Manages GPU resources with COMPLETE separation between viewer and model processing."""
    
    def __init__(self, device):
        self.device = device
        self.model_stream = None
        self.viewer_stream = None
        
        if torch.cuda.is_available():
            # COMPLETELY SEPARATE streams - zero interaction
            self.model_stream = torch.cuda.Stream()     # For model inference only
            self.viewer_stream = torch.cuda.Stream()    # For viewer only (tiny)
            print("Created ISOLATED CUDA streams: model and viewer")
        
        # VIEWER-ONLY: Tiny, dedicated GPU memory (never grows, never competes)
        self.VIEWER_MAX_VERTICES = 2000  # Small, fixed budget
        self.viewer_vertices_gpu = None
        self.viewer_colors_gpu = None
        self.viewer_vertex_count = 0
        self.viewer_lock = Lock()
        
        # MODEL-ONLY: No interference with viewer
        self.model_output_vertices = None
        self.model_output_colors = None
        self.model_output_count = 0
        self.model_lock = Lock()
        
        self._initialize_isolated_gpu_pools()
    
    def _initialize_isolated_gpu_pools(self):
        """Pre-allocate SEPARATE GPU memory - viewer and model never compete."""
        if torch.cuda.is_available():
            # VIEWER ONLY: Tiny, dedicated GPU memory
            with torch.cuda.stream(self.viewer_stream):
                self.viewer_vertices_gpu = torch.zeros((self.VIEWER_MAX_VERTICES, 3), device=self.device, dtype=torch.float32)
                self.viewer_colors_gpu = torch.zeros((self.VIEWER_MAX_VERTICES, 3), device=self.device, dtype=torch.float32)
                print(f"✅ VIEWER GPU POOL: {self.VIEWER_MAX_VERTICES} vertices (dedicated, isolated)")
            
            # Initialize viewer with small test pattern (immediate)
            self._create_isolated_viewer_pattern()
        else:
            # CPU fallback
            self.viewer_vertices_gpu = torch.zeros((self.VIEWER_MAX_VERTICES, 3), dtype=torch.float32)
            self.viewer_colors_gpu = torch.zeros((self.VIEWER_MAX_VERTICES, 3), dtype=torch.float32)
            self._create_isolated_viewer_pattern()
    
    def _create_isolated_viewer_pattern(self):
        """Create small test pattern in viewer's dedicated GPU memory."""
        # Create small cube pattern (uses viewer's dedicated memory only)
        cube_size = min(500, self.VIEWER_MAX_VERTICES // 4)  # Quarter of budget
        
        # Generate test cube points
        cube_points = []
        cube_colors = []
        
        for i in range(cube_size):
            # Random points in cube
            point = torch.rand(3) * 4.0 - 2.0  # [-2, 2] range
            color = torch.rand(3)  # Random color
            cube_points.append(point)
            cube_colors.append(color)
        
        # Fill viewer's dedicated GPU memory
        with torch.cuda.stream(self.viewer_stream) if torch.cuda.is_available() else torch.no_grad():
            viewer_count = len(cube_points)
            if viewer_count > 0:
                points_tensor = torch.stack(cube_points).to(self.device)
                colors_tensor = torch.stack(cube_colors).to(self.device)
                
                with self.viewer_lock:
                    self.viewer_vertices_gpu[:viewer_count] = points_tensor
                    self.viewer_colors_gpu[:viewer_count] = colors_tensor
                    self.viewer_vertex_count = viewer_count
        
        print(f"🎯 Viewer test pattern: {viewer_count} vertices in dedicated GPU memory")
    
    def update_model_output(self, model_vertices, model_colors, model_count):
        """Store model output - NEVER touches viewer memory."""
        with self.model_lock:
            self.model_output_vertices = model_vertices
            self.model_output_colors = model_colors
            self.model_output_count = model_count
        print(f"📊 Model output stored: {model_count} vertices (isolated from viewer)")
    
    def sample_model_to_viewer_async(self):
        """Background sampling from model to viewer - NON-BLOCKING."""
        def sampling_worker():
            try:
                # Get model data (non-blocking read)
                with self.model_lock:
                    if (self.model_output_vertices is None or 
                        self.model_output_colors is None or 
                        self.model_output_count == 0):
                        return  # No model data yet
                    
                    model_vertices = self.model_output_vertices
                    model_colors = self.model_output_colors
                    model_count = self.model_output_count
                
                # Calculate sampling (much smaller for viewer)
                if model_count > self.VIEWER_MAX_VERTICES:
                    step = max(1, model_count // self.VIEWER_MAX_VERTICES)
                    sample_count = min(self.VIEWER_MAX_VERTICES, model_count // step)
                else:
                    step = 1
                    sample_count = min(self.VIEWER_MAX_VERTICES, model_count)
                
                # Sample using viewer's dedicated stream (isolated)
                with torch.cuda.stream(self.viewer_stream) if torch.cuda.is_available() else torch.no_grad():
                    sampled_vertices = model_vertices[::step][:sample_count]
                    sampled_colors = model_colors[::step][:sample_count]
                    
                    # Update viewer's dedicated GPU memory (atomic)
                    with self.viewer_lock:
                        self.viewer_vertices_gpu[:sample_count] = sampled_vertices
                        self.viewer_colors_gpu[:sample_count] = sampled_colors
                        self.viewer_vertex_count = sample_count
                
                print(f"🔄 Sampled {sample_count}/{model_count} vertices to viewer (background)")
                
            except Exception as e:
                print(f"⚠️ Background sampling error: {e}")
        
        # Run sampling in background thread (never blocks anything)
        Thread(target=sampling_worker, daemon=True).start()
    
    def get_viewer_data_for_rendering(self):
        """Get viewer data for OpenGL rendering - ALWAYS FAST."""
        with self.viewer_lock:
            if self.viewer_vertex_count == 0:
                return None, None, 0
            
            # Convert SMALL viewer data to CPU (always fast)
            if torch.cuda.is_available():
                vertices_cpu = self.viewer_vertices_gpu[:self.viewer_vertex_count].cpu().numpy().flatten()
                colors_cpu = self.viewer_colors_gpu[:self.viewer_vertex_count].cpu().numpy().flatten()
            else:
                vertices_cpu = self.viewer_vertices_gpu[:self.viewer_vertex_count].numpy().flatten()
                colors_cpu = self.viewer_colors_gpu[:self.viewer_vertex_count].numpy().flatten()
            
            return vertices_cpu, colors_cpu, self.viewer_vertex_count

class AsyncDataProcessor:
    """Handles data processing - feeds model output to GPU manager without blocking viewer."""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.data_queue = queue.Queue(maxsize=10)
        self.processed_queue = queue.Queue(maxsize=3)
        self.exit_event = Event()
        self.worker_thread = None
        self.stats_lock = Lock()
        
        # Processing statistics
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_process_time = time.time()
        
    def start(self):
        """Start the async data processing thread."""
        self.worker_thread = Thread(target=self._process_worker, daemon=True)
        self.worker_thread.start()
        print("Started async data processor")
    
    def stop(self):
        """Stop the async data processing thread."""
        self.exit_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
    
    def submit_data(self, data):
        """Submit data for processing (non-blocking)."""
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest data if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data)
                with self.stats_lock:
                    self.frames_dropped += 1
            except queue.Empty:
                pass
    
    def get_processed_data(self):
        """Get processed data (non-blocking)."""
        try:
            return self.processed_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_worker(self):
        """Worker thread for processing data - stores model output in GPU manager."""
        while not self.exit_event.is_set():
            try:
                # Get data with timeout to allow checking exit event
                data = self.data_queue.get(timeout=0.1)
                
                # Process data using model stream (isolated from viewer)
                processed_data = self._process_data_gpu(data)
                
                # Submit processed result
                try:
                    self.processed_queue.put_nowait(processed_data)
                except queue.Full:
                    # Drop oldest processed data if queue is full
                    try:
                        self.processed_queue.get_nowait()
                        self.processed_queue.put_nowait(processed_data)
                    except queue.Empty:
                        pass
                
                with self.stats_lock:
                    self.frames_processed += 1
                    self.last_process_time = time.time()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async data processor: {e}")
                traceback.print_exc()
    
    def _process_data_gpu(self, data):
        """Process data using model stream - NEVER touches viewer resources."""
        try:
            # Handle status messages first
            if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str):
                return {
                    'status_message': data[1] if data[0] == "status" else f"ERROR: {data[1]}",
                    'vertex_count': 0
                }
            
            # Handle full data tuples
            if not isinstance(data, tuple) or len(data) != 16:
                print(f"Warning: Unexpected data format in GPU worker: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                return None
                
            (vertices_data, colors_data, num_vertices, rgb_np, depth_tensor, edge_viz, 
            smooth_viz, t_cap, cur_frame, tot_frames, rec_count, 
            f_read_dt, d_proc_dt, lat_ms, bias_map, main_coeff_viz) = data
            
            print(f"DEBUG: Processing {num_vertices} model vertices")
            
            # Use model stream for processing (isolated from viewer)
            with torch.cuda.stream(self.gpu_manager.model_stream) if torch.cuda.is_available() else torch.no_grad():
                # Process depth visualization
                depth_viz = None
                if depth_tensor is not None:
                    depth_np = depth_tensor.cpu().numpy()
                    min_val, max_val = np.min(depth_np), np.max(depth_np)
                    if max_val > min_val:
                        depth_scaled = 255 * (np.clip(depth_np, min_val, max_val) - min_val) / (max_val - min_val + 1e-6)
                        depth_u8 = depth_scaled.astype(np.uint8)
                        if depth_u8.size > 0:
                            depth_viz = cv2.cvtColor(cv2.equalizeHist(depth_u8), cv2.COLOR_GRAY2RGB)
            
            # Process vertex data and store in GPU manager (isolated from viewer)
            vertex_count = 0
            model_vertices = None
            model_colors = None
            
            if vertices_data is not None and colors_data is not None and num_vertices > 0:
                with torch.cuda.stream(self.gpu_manager.model_stream) if torch.cuda.is_available() else torch.no_grad():
                    try:
                        # Convert to GPU tensors - store in model space (not viewer space)
                        vertices_tensor = torch.from_numpy(vertices_data.reshape(-1, 3)).to(self.gpu_manager.device)
                        colors_tensor = torch.from_numpy(colors_data.reshape(-1, 3)).to(self.gpu_manager.device)
                        
                        # Store model output in GPU manager (isolated)
                        self.gpu_manager.update_model_output(vertices_tensor, colors_tensor, num_vertices)
                        
                        # Trigger background sampling to viewer (non-blocking)
                        self.gpu_manager.sample_model_to_viewer_async()
                        
                        model_vertices = vertices_tensor
                        model_colors = colors_tensor
                        vertex_count = num_vertices
                        
                        print(f"✅ Model output: {vertex_count} vertices (isolated from viewer)")
                    except Exception as e:
                        print(f"Error creating GPU tensors: {e}")
                        vertex_count = 0
                        model_vertices = None
                        model_colors = None
            
            result = {
                'vertex_count': vertex_count,
                'model_vertices': model_vertices,  # Full model output
                'model_colors': model_colors,      # Full model output
                'rgb_frame': rgb_np,
                'depth_viz': depth_viz,
                'edge_viz': edge_viz,
                'smooth_viz': smooth_viz,
                'timestamp': t_cap,
                'frame_info': (cur_frame, tot_frames),
                'recording_count': rec_count,
                'timing': (f_read_dt, d_proc_dt, lat_ms),
                'bias_map': bias_map,
                'coeff_viz': main_coeff_viz
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing data in GPU worker: {e}")
            traceback.print_exc()
            return None

class AsyncRenderer:
    """Handles rendering with GUARANTEED monitor refresh rate - viewer completely isolated."""
    
    def __init__(self, renderer, gpu_manager):
        self.renderer = renderer
        self.gpu_manager = gpu_manager
        self.render_lock = Lock()
        
        # VIEWER-ONLY: Monitor refresh rate rendering (never blocked)
        self.viewer_fps = 0.0
        self.last_render_time = time.time()
        self.render_frame_count = 0
        
        # GUARANTEED: Always have something to render
        self.viewer_ready = True  # Always ready with dedicated GPU memory
        
        print(f"AsyncRenderer initialized with guaranteed monitor-rate rendering")
    
    def render_frame(self):
        """Render frame using dedicated viewer GPU memory - NEVER blocks."""
        current_time = time.time()
        
        # Get viewer data from dedicated GPU memory (always fast)
        vertices_data, colors_data, vertex_count = self.gpu_manager.get_viewer_data_for_rendering()
        
        # ALWAYS render something - never wait
        if vertices_data is not None and colors_data is not None and vertex_count > 0:
            try:
                # Minimal rendering settings for maximum speed
                minimal_settings = {
                    'enable_point_smoothing': False,
                    'enable_edge_aware_smoothing': False,
                    'enable_sharpening': False,
                    'enable_point_thickening': False
                }
                
                # Update OpenGL renderer (main thread only) - pure CPU operation
                self.renderer.update_vertex_list(
                    vertices_data, colors_data, vertex_count, 
                    minimal_settings, None  # Simple view matrix
                )
                
            except Exception as e:
                print(f"Error in dedicated viewer rendering: {e}")
                # Continue rendering - never stop
        
        # ALWAYS render the scene regardless of data availability
        if hasattr(self.renderer, 'render_scene'):
            # Use simple matrices for guaranteed rendering
            identity_matrix = Mat4()
            latest_rgb_shape = (480, 640, 3)  # Default shape
            self.renderer.render_scene(identity_matrix, identity_matrix, {}, latest_rgb_shape)
        
        # Track viewer FPS (should match monitor refresh rate)
        self.render_frame_count += 1
        if current_time - self.last_render_time >= 1.0:
            self.viewer_fps = self.render_frame_count / (current_time - self.last_render_time)
            self.render_frame_count = 0
            self.last_render_time = current_time
            
            # Log viewer FPS (should be monitor refresh rate)
            if self.viewer_fps > 0:
                with self.gpu_manager.viewer_lock:
                    viewer_count = self.gpu_manager.viewer_vertex_count
                print(f"🖥️ VIEWER: {self.viewer_fps:.1f} FPS | {viewer_count} vertices (dedicated GPU memory)")
    
    def stop(self):
        """Stop background processing."""
        # Nothing to stop - viewer uses dedicated resources
        print("AsyncRenderer stopped")

class LiveViewerWindow(pyglet.window.Window):
    def __init__(self, model_name, inference_interval=1, target_inference_fps=10.0, *args, **kwargs):
        import time
        start_time = time.time()
        print(f"🔍 DEBUG: LiveViewerWindow.__init__ - Start at {start_time}")
        
        init_start = time.time()
        super().__init__(*args, **kwargs)
        super_time = time.time()
        print(f"🔍 DEBUG: super().__init__ took {super_time - init_start:.3f}s")

        # CRITICAL: Initialize basic attributes FIRST
        attr_start = time.time()
        self._model_name = model_name
        self._inference_interval = inference_interval
        self._target_inference_fps = target_inference_fps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ONLY essential settings immediately
        for key_attr, value_attr in DEFAULT_SETTINGS.items():
            setattr(self, key_attr, value_attr)
        attr_time = time.time()
        print(f"🔍 DEBUG: Basic attributes setup took {attr_time - attr_start:.3f}s")

        # INSTANT CAMERA VISUALIZATION - NO DELAYS!
        camera_start = time.time()
        print(f"🚀 INSTANT camera visualization starting at {camera_start:.3f}s...")
        
        # Initialize camera state with INSTANT test pattern
        self.preview_cap = None
        self.camera_preview_active = True  # Start immediately with test pattern
        self.camera_ready = False
        self.latest_rgb_frame = self._generate_test_pattern()  # INSTANT visual feedback
        self.status_message = "📹 Camera LIVE!"
        
        # Start INSTANT test pattern animation
        self.test_pattern_frame = 0
        self._start_instant_test_pattern()
        
        # SIMULTANEOUSLY start real camera in background (doesn't block anything)
        def async_camera_worker():
            print("📹 Real camera initializing in background...")
            import cv2
            
            try:
                # Try fastest approach - MSMF without config
                worker_start = time.time()
                cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
                open_time = time.time()
                print(f"📹 Real camera opened in background ({open_time - worker_start:.3f}s)")
                
                if cap.isOpened():
                    # Get first real frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # SEAMLESSLY switch from test pattern to real camera
                        print("✅ Switching from test pattern to REAL camera!")
                        self.preview_cap = cap
                        self.latest_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.camera_ready = True
                        self.status_message = "✅ Real camera LIVE!"
                        total_time = time.time()
                        print(f"✅ REAL CAMERA READY! Background init: {total_time - worker_start:.3f}s")
                        return
                
                # Fallback to auto-detect
                print("📹 MSMF failed, trying auto-detect...")
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print("✅ Switching from test pattern to REAL camera (fallback)!")
                        self.preview_cap = cap
                        self.latest_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.camera_ready = True
                        self.status_message = "✅ Real camera LIVE!"
                        return
                
                print("❌ Real camera failed - keeping test pattern")
                self.status_message = "🎭 Test pattern (camera failed)"
                        
            except Exception as e:
                print(f"❌ Real camera error: {e} - keeping test pattern")
                self.status_message = f"🎭 Test pattern (error: {e})"
        
        # Start real camera worker (non-blocking)
        self.camera_worker_thread = threading.Thread(target=async_camera_worker, daemon=True)
        self.camera_worker_thread.start()
        
        camera_launch_time = time.time()
        print(f"✅ INSTANT camera launched in {camera_launch_time - camera_start:.3f}s (test pattern + real camera loading)")

        # START ALL HEAVY PROCESSES IN PARALLEL IMMEDIATELY
        parallel_start = time.time()
        print(f"🚀 LAUNCHING ALL SYSTEMS IN PARALLEL at {parallel_start:.3f}s...")
        
        # Initialize containers for parallel results
        self.parallel_results = {
            'camera_thread': None,
            'model_loading': False,
            'model_load_error': None,
            'model': None,
            'components_ready': False
        }
        
        # Initialize model attributes for compatibility
        self.model = None
        self.model_loading = True
        self.model_load_error = None
        
        # PARALLEL LAUNCH 1: Camera preview thread
        thread1_start = time.time()
        self._start_camera_preview_thread_immediate()
        thread1_time = time.time()
        print(f"🔍 DEBUG: Camera thread launch took {thread1_time - thread1_start:.3f}s")
        
        # PARALLEL LAUNCH 2: Model loading thread
        thread2_start = time.time()
        self._start_model_loading_immediate()
        thread2_time = time.time()
        print(f"🔍 DEBUG: Model thread launch took {thread2_time - thread2_start:.3f}s")
        
        # PARALLEL LAUNCH 3: Heavy component initialization thread  
        thread3_start = time.time()
        self._start_component_initialization_immediate()
        thread3_time = time.time()
        print(f"🔍 DEBUG: Component thread launch took {thread3_time - thread3_start:.3f}s")
        
        parallel_total_time = time.time()
        print(f"🔍 DEBUG: All parallel launches took {parallel_total_time - parallel_start:.3f}s")
        
        # MINIMAL main thread setup for immediate responsiveness
        setup_start = time.time()
        self.camera = Camera(initial_position=Vec3(0,0,5))
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.mouse_down, self.is_mouse_exclusive = False, False

        # Camera motion tracking
        self.prev_camera_position = Vec3(self.camera.position.x, self.camera.position.y, self.camera.position.z)
        self.prev_camera_rotation_x = self.camera.rotation_x
        self.prev_camera_rotation_y = self.camera.rotation_y
        self.camera_motion_confidence = 0.0 

        # Performance tracking
        self.last_update_time = time.time()
        self.point_cloud_fps = self.input_fps = self.depth_fps = self.latency_ms = 0.0
        self.frame_counter_display = 0
        
        # Data containers
        self.latest_depth_map_viz = self.latest_edge_map = self.latest_smoothing_map = None
        self.latest_wavelet_map = self.latest_main_screen_coeff_viz_content = self.latest_points_for_debug = None
        self.depth_bias_map = None
        self.is_playing = True
        
        # Legacy containers (will be initialized by parallel threads)
        self._data_queue = queue.Queue()
        self._exit_event = threading.Event()
        self.inference_thread = None
        
        # Frame statistics
        self.frames_dropped = 0
        self.last_vertex_update_time = time.time()
        self.vertex_update_interval = 0.0
        self.enable_frame_interpolation = True
        
        # References (will be updated by parallel initialization)
        self.scale_factor_ref = [getattr(self, "input_scale_factor", DEFAULT_SETTINGS["input_scale_factor"])]
        self.edge_params_ref = {}
        self.playback_state = {}
        self.recording_state = {}
        
        # Depth tracking
        self.depth_history = deque(maxlen=getattr(self, "dmd_time_window", DEFAULT_SETTINGS["dmd_time_window"]))
        self.latest_depth_tensor = None
        self.latest_depth_tensor_for_calib = None
        self.pending_bias_capture_request = None
        self.show_screen_share_popup = self.temp_monitor_index = False
        self._current_vertex_count = 0
        
        setup_time = time.time()
        print(f"🔍 DEBUG: Main thread setup took {setup_time - setup_start:.3f}s")
        
        # IMMEDIATE: Basic OpenGL setup for rendering
        gl_start = time.time()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl_time = time.time()
        print(f"🔍 DEBUG: OpenGL setup took {gl_time - gl_start:.3f}s")
        
        # Schedule update loop immediately for responsiveness
        schedule_start = time.time()
        pyglet.clock.schedule(self.update)
        schedule_time = time.time()
        print(f"🔍 DEBUG: Schedule update took {schedule_time - schedule_start:.3f}s")
        
        total_time = time.time()
        print(f"✅ MAIN THREAD READY - Total init time: {total_time - start_time:.3f}s")
        print(f"🎯 Camera is LIVE, Model loading, Components initializing...")

    def _start_camera_preview_thread_immediate(self):
        """Start camera preview thread - now handles test pattern → real camera transition."""
        def camera_preview_worker():
            print("📹 Camera preview worker starting with instant test pattern...")
            
            try:
                frame_count = 0
                while self.camera_preview_active:
                    # If real camera is ready, use it; otherwise test pattern is already running
                    if self.camera_ready and self.preview_cap and self.preview_cap.isOpened():
                        # Real camera mode
                        ret, frame = self.preview_cap.read()
                        if ret and frame is not None:
                            self.latest_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_count += 1
                            if frame_count % 30 == 0:
                                print(f"📹 Real camera: {frame_count} frames captured")
                        else:
                            print("⚠️ Real camera frame read failed")
                            # Fall back to test pattern
                            self.camera_ready = False
                            self.latest_rgb_frame = self._generate_test_pattern()
                    else:
                        # Test pattern mode (handled by _start_instant_test_pattern)
                        # Just wait for real camera or continue with test pattern
                        pass
                    
                    time.sleep(1.0/30.0)  # 30 FPS preview
                    
            except Exception as e:
                print(f"❌ Camera preview error: {e}")
                self.status_message = f"Camera error: {e}"
                # Fall back to test pattern
                self.latest_rgb_frame = self._generate_test_pattern()
                    
        # Start preview thread immediately (test pattern already running)
        self.camera_preview_thread = threading.Thread(target=camera_preview_worker, daemon=True)
        self.camera_preview_thread.start()
        self.parallel_results['camera_thread'] = self.camera_preview_thread
        print("✅ Camera preview thread launched with instant visualization!")

    def _start_model_loading_immediate(self):
        """Load model in parallel thread."""
        def model_loading_worker():
            try:
                print(f"🤖 Background loading UniK3D model: {self._model_name}...")
                self.status_message = f"🤖 Loading {self._model_name}..."
                
                from unik3d.models import UniK3D
                model = UniK3D.from_pretrained(f"lpiccinelli/{self._model_name}").to(self.device).eval()
                
                # Atomic assignment to both locations
                self.parallel_results['model'] = model
                self.parallel_results['model_loading'] = False
                self.parallel_results['model_load_error'] = None
                
                # Direct attributes for compatibility
                self.model = model
                self.model_loading = False
                self.model_load_error = None
                
                print("✅ Model loaded successfully!")
                self.status_message = "✅ Model ready!"
                
            except Exception as e:
                print(f"❌ Model load failed: {e}")
                self.parallel_results['model_load_error'] = str(e)
                self.parallel_results['model_loading'] = False
                
                # Direct attributes for compatibility  
                self.model_load_error = str(e)
                self.model_loading = False
                self.status_message = f"❌ Model failed: {e}"
                
        self.model_loading_thread = threading.Thread(target=model_loading_worker, daemon=True)
        self.model_loading_thread.start()
        print("✅ Model loading thread launched!")

    def _start_component_initialization_immediate(self):
        """Initialize heavy components in parallel thread with async managers."""
        def component_init_worker():
            try:
                print("🔧 Initializing async component managers...")
                
                # Initialize async managers for complete separation
                self.async_managers = {
                    'inference': None,  # Will be created when model is ready
                    'ui': None,         # Will be created when window is ready
                    'window': None      # Will be created on main thread
                }
                
                # Prepare settings for async managers
                self.async_settings = {
                    'input_scale_factor': getattr(self, 'input_scale_factor', DEFAULT_SETTINGS['input_scale_factor']),
                    'input_mode': getattr(self, 'input_mode', 'Live'),
                    'input_filepath': getattr(self, 'input_filepath', ''),
                    'live_processing_mode': getattr(self, 'live_processing_mode', DEFAULT_SETTINGS['live_processing_mode']),
                    'screen_capture_monitor_index': getattr(self, 'screen_capture_monitor_index', 0),
                    'playback_state': {},
                    'recording_state': {},
                    'edge_params': {}
                }
                
                # Load settings asynchronously
                self.load_settings()
                self._update_edge_params()
                self.update_playback_state()
                self.update_recording_state()
                
                # Update async settings with loaded values
                self.async_settings.update({
                    'playback_state': getattr(self, 'playback_state', {}),
                    'recording_state': getattr(self, 'recording_state', {}),
                    'edge_params': getattr(self, 'edge_params_ref', {})
                })
                
                self.parallel_results['components_ready'] = True
                print("✅ Async component managers prepared!")
                
            except Exception as e:
                print(f"❌ Async component init error: {e}")
                self.status_message = f"Component error: {e}"
                
        self.component_init_thread = threading.Thread(target=component_init_worker, daemon=True)
        self.component_init_thread.start()
        print("✅ Async component initialization thread launched!")

    def _initialize_main_thread_components_when_ready(self):
        """Initialize OpenGL components and async managers on main thread when ready."""
        if hasattr(self, 'main_components_initialized'):
            return  # Already done
            
        # Check if parallel preparation is ready
        if not self.parallel_results.get('components_ready', False):
            return  # Not ready yet
            
        print("🎯 Initializing main thread components with ISOLATED GPU resources...")
        
        # Initialize core OpenGL components (must be on main thread)
        self.renderer = Renderer(self.width, self.height)
        self.ui_manager = UIManager(self, self)
        self._initialize_text_overlays()
        
        # Initialize ISOLATED GPU components - COMPLETE separation
        self.gpu_manager = GPUResourceManager(self.device)
        self.data_processor = AsyncDataProcessor(self.gpu_manager)
        
        # Initialize ISOLATED async renderer - uses dedicated viewer GPU memory
        self.async_renderer = AsyncRenderer(self.renderer, self.gpu_manager)
        
        print("✅ ISOLATED rendering: Viewer has dedicated 2000-vertex GPU memory")
        print("✅ Model processing: Uses separate GPU memory space")
        print("✅ ZERO resource competition between viewer and model")
        
        # Initialize async managers for UI responsiveness
        print("🚀 Initializing async UI managers...")
        
        # 1. Window Manager (main thread OpenGL operations)
        self.async_window_manager = AsyncWindowManager(self)
        
        # 2. UI Manager (background UI processing)
        self.async_ui_manager = AsyncUIManager(self, self)
        self.async_ui_manager.start_ui_thread()
        
        # 3. Inference Manager (completely isolated model inference)
        # Will be initialized when model is ready
        
        # Start data processing
        self.data_processor.start()
        
        print("✅ GUARANTEED monitor-rate viewer with zero model interference")
        
        self.main_components_initialized = True
        print("✅ Main thread components with ISOLATED resources ready!")
        
        # Start inference manager if model is ready
        if self.parallel_results.get('model') is not None:
            self.model = self.parallel_results['model']
            self._start_isolated_inference_manager()

    def _start_isolated_inference_manager(self):
        """Start completely isolated inference manager."""
        if hasattr(self, 'inference_manager_started'):
            return  # Already started
            
        if not hasattr(self, 'main_components_initialized') or not self.main_components_initialized:
            return  # Components not ready
            
        if self.parallel_results.get('model') is None:
            return  # Model not ready
            
        print("🚀 Starting completely isolated inference manager...")
        
        # Create isolated inference manager
        self.async_inference_manager = AsyncInferenceManager(
            model=self.parallel_results['model'],
            device=self.device,
            inference_interval=self._inference_interval,
            target_fps=self._target_inference_fps,
            data_queue=self._data_queue,
            exit_event=self._exit_event
        )
        
        # Start isolated inference with current settings
        inference_settings = self.async_settings.copy()
        inference_settings.update({
            'input_mode': getattr(self, 'input_mode', 'Live'),
            'input_filepath': getattr(self, 'input_filepath', ''),
            'live_processing_mode': getattr(self, 'live_processing_mode', True),
            'screen_capture_monitor_index': getattr(self, 'screen_capture_monitor_index', 0),
            'playback_state': getattr(self, 'playback_state', {}),
            'recording_state': getattr(self, 'recording_state', {}),
            'edge_params': getattr(self, 'edge_params_ref', {}),
            'input_scale_factor': getattr(self, 'input_scale_factor', 1.0)
        })
        
        self.async_inference_manager.start_inference(inference_settings)
        self.inference_manager_started = True
        print("✅ Isolated inference manager active!")

    def _start_inference_when_ready(self):
        """Start inference when both model and components are ready."""
        if hasattr(self, 'inference_started'):
            return  # Already started
            
        if not hasattr(self, 'main_components_initialized') or not self.main_components_initialized:
            return  # Components not ready
            
        if self.parallel_results.get('model') is None:
            return  # Model not ready
            
        if not getattr(self, 'camera_ready', False):
            return  # Camera not ready yet
            
        print("🚀 Starting isolated inference system...")
        
        # Set model reference for compatibility
        self.model = self.parallel_results['model']
        
        # Start isolated inference manager
        self._start_isolated_inference_manager()
        
        # Stop camera preview and switch to inference camera
        self.camera_preview_active = False
        if hasattr(self, 'camera_preview_thread'):
            self.camera_preview_thread.join(timeout=1.0)
        
        self.inference_started = True
        print("✅ Isolated inference system active!")

    def _initialize_text_overlays(self):
        """Initialize text overlay labels."""
        self.overlay_batch = pyglet.graphics.Batch()
        label_color = (200, 200, 200, 200)
        y_pos = self.height - 20
        self.fps_label = pyglet.text.Label("", x=self.width-10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.overlay_batch, color=label_color)
        y_pos -= 20
        self.points_label = pyglet.text.Label("", x=self.width-10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.overlay_batch, color=label_color)
        y_pos -= 20
        self.input_fps_label = pyglet.text.Label("", x=self.width-10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.overlay_batch, color=label_color)
        y_pos -= 20
        self.depth_fps_label = pyglet.text.Label("", x=self.width-10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.overlay_batch, color=label_color)
        y_pos -= 20
        self.latency_label = pyglet.text.Label("", x=self.width-10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.overlay_batch, color=label_color)
        y_pos -= 20
        self.dropped_frames_label = pyglet.text.Label("", x=self.width-10, y=y_pos, anchor_x='right', anchor_y='top', batch=self.overlay_batch, color=label_color)

    def start_inference_thread(self):
        # Check if model is loaded
        if self.model is None:
            if self.parallel_results.get('model_loading', True):
                print("DEBUG: Model still loading, inference will start automatically when ready")
                return
            elif self.parallel_results.get('model_load_error'):
                print(f"DEBUG: Cannot start inference - model load failed: {self.parallel_results['model_load_error']}")
                return
            else:
                print("DEBUG: No model available and not loading")
                return
                
        if self.inference_thread and self.inference_thread.is_alive():
            print("DEBUG: Stopping existing inference thread...")
            self._exit_event.set()
            self.inference_thread.join(timeout=3.0)
            if self.inference_thread.is_alive(): print("Warning: Inference thread did not stop in time.")
            self.inference_thread = None
            try:
                while True: self._data_queue.get_nowait()
            except queue.Empty: pass
            if self.renderer.vertex_list: self.renderer.vertex_list.delete(); self.renderer.vertex_list = None
            self.renderer.current_point_count = 0

        self._exit_event = threading.Event() # Fresh event
        self._update_edge_params() # Ensure params are fresh before thread start
        self.update_playback_state()
        self.update_recording_state()
        if not hasattr(self, 'scale_factor_ref') or self.scale_factor_ref is None: # Should be set by load_settings
            self.scale_factor_ref = [self.input_scale_factor]
        
        self.inference_thread = threading.Thread(
            target=inference_thread_func,
            args=(
                self._data_queue, self._exit_event, self.model, self.device,
                self._inference_interval, self.scale_factor_ref, self.edge_params_ref,
                self.input_mode, self.input_filepath, self.playback_state,
                self.recording_state, self.live_processing_mode, self.screen_capture_monitor_index,
                self._target_inference_fps
            ),
            daemon=True
        )
        self.inference_thread.start()
        print(f"DEBUG: Inference thread started (Mode: {self.input_mode}, File: {self.input_filepath or 'N/A'}).")
        self.status_message = f"Processing {self.input_mode}..."

    def _process_queue_data(self, latest_data):
        """Legacy method - now handled by AsyncDataProcessor."""
        # This method is kept for compatibility but most logic moved to AsyncDataProcessor
        try:
            (vertices_data, colors_data, num_vertices, rgb_np, depth_tensor, edge_viz, 
            smooth_viz, t_cap, cur_frame, tot_frames, rec_count, 
            f_read_dt, d_proc_dt, lat_ms, bias_map, main_coeff_viz) = latest_data

            # Handle status messages
            if isinstance(latest_data, tuple) and isinstance(latest_data[0], str):
                if latest_data[0] == "status": 
                    self.status_message = latest_data[1]
                elif latest_data[0] == "error": 
                    self.status_message = f"ERROR: {latest_data[1]}"
                return None, None, 0

            # Update timing and status
            self.latency_ms = lat_ms
            self.last_capture_timestamp = t_cap
            self.playback_state["current_frame"] = cur_frame
            self.playback_state["total_frames"] = tot_frames
            self.recording_state["frames_saved"] = rec_count
            self.frame_read_delta_t = f_read_dt
            self.depth_process_delta_t = d_proc_dt

            # Update latest data for UI
            self.latest_rgb_frame = rgb_np
            self.latest_edge_map = edge_viz
            self.latest_smoothing_map = smooth_viz
            self.latest_main_screen_coeff_viz_content = main_coeff_viz

            # Process depth tensor for history and visualization
            if depth_tensor is not None:
                self.latest_depth_tensor = depth_tensor.to(self.device)
                self.depth_history.append(self.latest_depth_tensor)
                
                # Generate depth visualization
                try:
                    depth_np = depth_tensor.cpu().numpy()
                    min_val, max_val = np.min(depth_np), np.max(depth_np)
                    if max_val > min_val:
                        depth_scaled = 255 * (np.clip(depth_np, min_val, max_val) - min_val) / (max_val - min_val + 1e-6)
                        depth_u8 = depth_scaled.astype(np.uint8)
                        if depth_u8.size > 0:
                            self.latest_depth_map_viz = cv2.cvtColor(cv2.equalizeHist(depth_u8), cv2.COLOR_GRAY2RGB)
                except Exception as e_depth_viz:
                    print(f"Error processing depth map for viz: {e_depth_viz}")
                    self.latest_depth_map_viz = None

            # Handle bias map
            if bias_map is not None:
                self.depth_bias_map = bias_map.to(self.device)
                self.apply_depth_bias = True
                self.status_message = "Depth bias captured and applied!"
                self._update_edge_params()

            # Store current vertex count for display
            self._current_vertex_count = num_vertices

            return vertices_data, colors_data, num_vertices

        except Exception as e_unpack:
            print(f"ERROR in _process_queue_data: {e_unpack}")
            traceback.print_exc()
            return None, None, 0

    def _update_debug_textures(self):
        """Legacy method - now handled asynchronously."""
        # This is now handled by _update_debug_textures_async
        pass

    def update(self, dt):
        """Main update loop - coordinates isolated renderer with async managers."""
        # FIRST: Initialize main thread components when parallel setup is ready
        self._initialize_main_thread_components_when_ready()
        
        # SECOND: Start isolated inference when both model and components are ready
        self._start_inference_when_ready()
        
        # THIRD: Handle async window manager render updates (if still needed)
        if hasattr(self, 'async_window_manager'):
            self.async_window_manager.process_render_updates()
        
        # Camera motion and input handling (main thread only)
        if self.is_mouse_exclusive:
            self.camera.update_from_input(dt, self.keys, self.is_mouse_exclusive)

        # Camera motion confidence update
        pos_delta = self.camera.position.distance(self.prev_camera_position)
        rot_x_delta = abs(self.camera.rotation_x - self.prev_camera_rotation_x)
        rot_y_delta = abs(self.camera.rotation_y - self.prev_camera_rotation_y) % 360
        if rot_y_delta > 180: 
            rot_y_delta = 360 - rot_y_delta
        
        motion_metric = pos_delta * 50.0 + rot_x_delta * 2.0 + rot_y_delta * 2.0 
        self.camera_motion_confidence = math.exp(-motion_metric * 0.5)
        self.camera_motion_confidence = max(0.0, min(1.0, self.camera_motion_confidence))

        self.prev_camera_position = Vec3(self.camera.position.x, self.camera.position.y, self.camera.position.z)
        self.prev_camera_rotation_x = self.camera.rotation_x
        self.prev_camera_rotation_y = self.camera.rotation_y
        
        # ASYNC: Update edge params for inference manager (non-blocking)
        if hasattr(self, 'async_inference_manager') and hasattr(self, '_update_edge_params'):
            self._update_edge_params()
            # Send updated settings to isolated inference manager
            inference_settings = {
                'edge_params': getattr(self, 'edge_params_ref', {}),
                'camera_motion_confidence': self.camera_motion_confidence,
                'input_scale_factor': getattr(self, 'input_scale_factor', 1.0)
            }
            self.async_inference_manager.update_settings(inference_settings)

        # ASYNC: Process data from isolated managers (all non-blocking)
        if hasattr(self, 'main_components_initialized') and self.main_components_initialized:
            self._process_async_manager_data()
            self._update_async_ui_data()
            # Note: No need for _update_async_renderer_non_blocking - isolated renderer handles itself

    def _process_async_manager_data(self):
        """Process data from all async managers without blocking any threads."""
        # 1. Get inference results from isolated inference manager (non-blocking)
        if hasattr(self, 'async_inference_manager'):
            inference_results = self.async_inference_manager.get_results()
            if inference_results:
                self._process_isolated_inference_results(inference_results)
        
        # 2. Process legacy inference data for compatibility (non-blocking)
        try:
            latest_data = self._data_queue.get_nowait()
            self.data_processor.submit_data(latest_data)
        except queue.Empty:
            pass  # No new data

        # 3. Get processed data from async processor (non-blocking)
        processed_data = self.data_processor.get_processed_data()
        if processed_data:
            self._apply_processed_data_to_managers(processed_data)

    def _process_isolated_inference_results(self, results):
        """Process results from isolated inference manager."""
        try:
            # Handle inference results without blocking
            if isinstance(results, dict):
                # Update UI data through async UI manager
                ui_update = {
                    'inference_fps': results.get('fps', 0.0),
                    'processing_time': results.get('processing_time', 0.0),
                    'frames_processed': results.get('frames_processed', 0)
                }
                if hasattr(self, 'async_ui_manager'):
                    self.async_ui_manager.update_ui_data(ui_update)
            
        except Exception as e:
            print(f"Error processing isolated inference results: {e}")

    def _apply_processed_data_to_managers(self, processed_data):
        """Apply processed data - model output stored in isolated GPU manager."""
        if processed_data is None:
            return
        
        # Handle status messages through UI manager
        if 'status_message' in processed_data:
            if hasattr(self, 'async_ui_manager'):
                self.async_ui_manager.update_ui_data({'status_message': processed_data['status_message']})
            return
            
        # Update application state (main thread)
        self.latest_rgb_frame = processed_data.get('rgb_frame')
        self.latest_depth_map_viz = processed_data.get('depth_viz')
        self.latest_edge_map = processed_data.get('edge_viz')
        self.latest_smoothing_map = processed_data.get('smooth_viz')
        self.latest_main_screen_coeff_viz_content = processed_data.get('coeff_viz')
        
        # Update timing through UI manager
        timing = processed_data.get('timing', (0, 0, 0))
        self.frame_read_delta_t, self.depth_process_delta_t, self.latency_ms = timing
        
        frame_info = processed_data.get('frame_info', (0, 0))
        self.playback_state["current_frame"], self.playback_state["total_frames"] = frame_info
        self.recording_state["frames_saved"] = processed_data.get('recording_count', 0)
        
        # Update UI data through async UI manager
        ui_update = {
            'latency_ms': self.latency_ms,
            'current_frame': self.playback_state.get("current_frame", 0),
            'total_frames': self.playback_state.get("total_frames", 0),
            'frames_saved': self.recording_state.get("frames_saved", 0)
        }
        if hasattr(self, 'async_ui_manager'):
            self.async_ui_manager.update_ui_data(ui_update)
        
        # Handle bias map
        bias_map = processed_data.get('bias_map')
        if bias_map is not None:
            self.depth_bias_map = bias_map.to(self.device)
            self.apply_depth_bias = True
            self.status_message = "Depth bias captured and applied!"
            self._update_edge_params()
        
        # CRITICAL: Model output is already stored in isolated GPU manager
        vertex_count = processed_data.get('vertex_count', 0)
        model_vertices = processed_data.get('model_vertices')
        model_colors = processed_data.get('model_colors')
        
        if vertex_count > 0 and model_vertices is not None and model_colors is not None:
            # Model output is already handled by GPU manager - NO additional processing needed
            # The GPU manager automatically samples model output to viewer's dedicated memory
            print(f"✅ Model output: {vertex_count} vertices (isolated from viewer)")
            
            # Update current count for UI
            self._current_vertex_count = vertex_count
        
        # VIEWER CONTINUES rendering at monitor rate with its dedicated GPU memory
        # NO interference from model processing

    def _update_async_ui_data(self):
        """Update UI data for async UI manager."""
        if not hasattr(self, 'async_ui_manager'):
            return
        
        # Collect current application state for UI from original renderer
        current_points = 0
        if hasattr(self, 'renderer') and self.renderer:
            current_points = getattr(self.renderer, 'current_point_count', 0)
        
        ui_data = {
            'status_message': getattr(self, 'status_message', 'Ready'),
            'vertex_count': current_points,
            'camera_frame': getattr(self, 'latest_rgb_frame', None),
            'render_fps': getattr(self, 'point_cloud_fps', 0.0),
            'input_fps': getattr(self, 'input_fps', 0.0),
            'depth_fps': getattr(self, 'depth_fps', 0.0)
        }
        
        # Send to async UI manager (non-blocking)
        self.async_ui_manager.update_ui_data(ui_data)

    def on_draw(self):
        """Main rendering method - GUARANTEED monitor refresh rate with isolated viewer."""
        # Clear the screen
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        
        # PRIMARY: ISOLATED viewer rendering - NEVER affected by model processing
        if hasattr(self, 'main_components_initialized') and self.main_components_initialized:
            if hasattr(self, 'async_renderer'):
                # Render using DEDICATED viewer GPU memory - ZERO model interference
                self.async_renderer.render_frame()
            else:
                # Fallback rendering while components initialize
                self._render_fallback()
        else:
            # Minimal rendering while components are loading
            self._render_loading_screen()
        
        # SECONDARY: UI rendering - lightweight, uses async data
        self._render_async_ui()
        
        # TERTIARY: Text overlays - show viewer stats (not affected by model processing)
        self._render_isolated_overlay_labels()
        
        # Background texture updates (non-blocking, throttled)
        self._update_debug_textures_async()

    def _render_fallback(self):
        """Fallback rendering when async renderer not ready."""
        if hasattr(self, 'renderer') and self.renderer:
            # Get camera matrices
            projection_matrix, view_matrix = self.get_projection_and_view_matrices()
            
            # Render with simple identity
            latest_rgb_shape = (480, 640, 3)
            render_settings = {}
            self.renderer.render_scene(projection_matrix, view_matrix, render_settings, latest_rgb_shape)

    def _render_loading_screen(self):
        """Render loading screen while components initialize."""
        if hasattr(self, 'latest_rgb_frame') and self.latest_rgb_frame is not None:
            # Show camera preview during loading
            pass

    def _render_async_ui(self):
        """Render UI using data from async UI manager."""
        if hasattr(self, 'ui_manager') and hasattr(self, 'async_ui_manager'):
            # Get UI data from async manager (non-blocking)
            ui_data = self.async_ui_manager.get_ui_data()
            
            # Update UI manager with async data before rendering
            if ui_data:
                # Update status message
                if 'status_message' in ui_data:
                    self.status_message = ui_data['status_message']
                
                # Update other UI-relevant data
                if 'vertex_count' in ui_data:
                    self._current_vertex_count = ui_data['vertex_count']
            
            # Render ImGui with updated data (main thread only)
            self.ui_manager.define_and_render_imgui()

    def _render_isolated_overlay_labels(self):
        """Render text overlays showing VIEWER stats (isolated from model processing)."""
        # Update overlay labels with VIEWER-SPECIFIC data (never affected by model)
        if hasattr(self, 'fps_label') and self.show_fps_overlay:
            # Show VIEWER FPS (should always be monitor refresh rate)
            viewer_fps = 0.0
            if hasattr(self, 'async_renderer'):
                viewer_fps = getattr(self.async_renderer, 'viewer_fps', 0.0)
            self.fps_label.text = f"Viewer FPS: {viewer_fps:.1f}"
            self.fps_label.visible = True
        else:
            if hasattr(self, 'fps_label'):
                self.fps_label.visible = False
        
        if hasattr(self, 'points_label') and self.show_points_overlay:
            # Show VIEWER vertex count (from dedicated GPU memory)
            viewer_count = 0
            model_count = getattr(self, '_current_vertex_count', 0)
            if hasattr(self, 'gpu_manager'):
                with self.gpu_manager.viewer_lock:
                    viewer_count = self.gpu_manager.viewer_vertex_count
            self.points_label.text = f"Viewer: {viewer_count:,} | Model: {model_count:,}"
            self.points_label.visible = True
        else:
            if hasattr(self, 'points_label'):
                self.points_label.visible = False
        
        # Get UI data for other overlays
        ui_data = {}
        if hasattr(self, 'async_ui_manager'):
            ui_data = self.async_ui_manager.get_ui_data()
        
        if hasattr(self, 'input_fps_label') and self.show_input_fps_overlay:
            input_fps = ui_data.get('input_fps', getattr(self, 'input_fps', 0.0))
            self.input_fps_label.text = f"Input FPS: {input_fps:.1f}"
            self.input_fps_label.visible = True
        else:
            if hasattr(self, 'input_fps_label'):
                self.input_fps_label.visible = False
        
        if hasattr(self, 'depth_fps_label') and self.show_depth_fps_overlay:
            depth_fps = ui_data.get('depth_fps', getattr(self, 'depth_fps', 0.0))
            self.depth_fps_label.text = f"Depth FPS: {depth_fps:.1f}"
            self.depth_fps_label.visible = True
        else:
            if hasattr(self, 'depth_fps_label'):
                self.depth_fps_label.visible = False
        
        if hasattr(self, 'latency_label') and self.show_latency_overlay:
            latency_ms = ui_data.get('latency_ms', getattr(self, 'latency_ms', 0.0))
            self.latency_label.text = f"Latency: {latency_ms:.1f} ms"
            self.latency_label.visible = True
        else:
            if hasattr(self, 'latency_label'):
                self.latency_label.visible = False
        
        if hasattr(self, 'dropped_frames_label') and getattr(self, 'show_dropped_frames_overlay', False):
            dropped = 0
            if hasattr(self, 'data_processor'):
                with self.data_processor.stats_lock:
                    dropped = self.data_processor.frames_dropped
            self.dropped_frames_label.text = f"Dropped: {dropped}"
            self.dropped_frames_label.visible = True
        else:
            if hasattr(self, 'dropped_frames_label'):
                self.dropped_frames_label.visible = False

        # Render overlay batch (main thread OpenGL)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        show_any_overlay = (self.show_fps_overlay or self.show_points_overlay or 
                           self.show_input_fps_overlay or self.show_depth_fps_overlay or 
                           self.show_latency_overlay or getattr(self, 'show_dropped_frames_overlay', False))
        
        if hasattr(self, 'overlay_batch') and self.overlay_batch and show_any_overlay:
            self.overlay_batch.draw()
        
        gl.glEnable(gl.GL_DEPTH_TEST)

    def on_close(self):
        """Clean shutdown of all isolated components and resources."""
        print("Window closing, shutting down ISOLATED components...")
        
        # Stop async UI manager
        if hasattr(self, 'async_ui_manager'):
            print("Stopping async UI manager...")
            self.async_ui_manager.stop_ui_thread()
        
        if hasattr(self, 'async_inference_manager'):
            print("Stopping async inference manager...")
            self.async_inference_manager.stop_inference()
        
        # Stop isolated renderer
        if hasattr(self, 'async_renderer'):
            print("Stopping isolated renderer...")
            self.async_renderer.stop()
        
        # Stop camera preview and test pattern
        if hasattr(self, 'camera_preview_active'):
            self.camera_preview_active = False
            if hasattr(self, 'camera_preview_thread'):
                print("Stopping camera preview...")
                self.camera_preview_thread.join(timeout=1.0)
        
        # Stop test pattern thread
        if hasattr(self, 'test_pattern_thread'):
            print("Stopping test pattern...")
            self.test_pattern_thread.join(timeout=1.0)
        
        if hasattr(self, 'preview_cap') and self.preview_cap:
            print("Releasing preview camera...")
            self.preview_cap.release()
        
        # Stop async data processor
        if hasattr(self, 'data_processor') and self.data_processor:
            print("Stopping async data processor...")
            self.data_processor.stop()
        
        # Stop legacy inference thread
        print("Stopping legacy inference thread...")
        if hasattr(self, '_exit_event') and self._exit_event:
            self._exit_event.set()
        if hasattr(self, 'inference_thread') and self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=3.0)
            if self.inference_thread.is_alive():
                print("Warning: Legacy inference thread did not stop in time")
        
        # Wait for any texture update threads to finish
        if hasattr(self, '_texture_update_thread') and self._texture_update_thread.is_alive():
            print("Waiting for texture update thread...")
            self._texture_update_thread.join(timeout=1.0)
        
        # Synchronize ISOLATED CUDA streams before cleanup
        if torch.cuda.is_available() and hasattr(self, 'gpu_manager') and self.gpu_manager:
            print("Synchronizing ISOLATED CUDA streams...")
            try:
                if self.gpu_manager.model_stream:
                    self.gpu_manager.model_stream.synchronize()
                if self.gpu_manager.viewer_stream:
                    self.gpu_manager.viewer_stream.synchronize()
            except Exception as e:
                print(f"Error synchronizing CUDA streams: {e}")
        
        # Clean up renderer and UI
        if hasattr(self, 'renderer') and self.renderer: 
            print("Cleaning up renderer...")
            self.renderer.cleanup()
        
        if hasattr(self, 'ui_manager') and self.ui_manager: 
            print("Shutting down UI manager...")
            self.ui_manager.shutdown()
        
        print("Attempting to close Pyglet window cleanly...")
        super().on_close()
        print("Window cleanup complete.")

    def _update_edge_params(self):
        """Updates the dictionary passed to the inference thread with correct key names and fallbacks."""
        # Helper to safely get attribute or default
        def _get_attr_safe(name, default_key_in_settings):
            return getattr(self, name, DEFAULT_SETTINGS[default_key_in_settings])

        self.edge_params_ref = {
            "enable_point_smoothing": _get_attr_safe("enable_point_smoothing", "enable_point_smoothing"),
            "min_alpha_points": _get_attr_safe("min_alpha_points", "min_alpha_points"),
            "max_alpha_points": _get_attr_safe("max_alpha_points", "max_alpha_points"),
            "enable_edge_aware": _get_attr_safe("enable_edge_aware_smoothing", "enable_edge_aware_smoothing"),
            "depth_threshold1": float(_get_attr_safe("depth_edge_threshold1", "depth_edge_threshold1")),
            "depth_threshold2": float(_get_attr_safe("depth_edge_threshold2", "depth_edge_threshold2")),
            "rgb_threshold1": float(_get_attr_safe("rgb_edge_threshold1", "rgb_edge_threshold1")),
            "rgb_threshold2": float(_get_attr_safe("rgb_edge_threshold2", "rgb_edge_threshold2")),
            "influence": _get_attr_safe("edge_smoothing_influence", "edge_smoothing_influence"),
            "gradient_influence_scale": _get_attr_safe("gradient_influence_scale", "gradient_influence_scale"),
            "enable_sharpening": _get_attr_safe("enable_sharpening", "enable_sharpening"),
            "sharpness": _get_attr_safe("sharpness", "sharpness"),
            "enable_point_thickening": _get_attr_safe("enable_point_thickening", "enable_point_thickening"),
            "thickening_duplicates": int(_get_attr_safe("thickening_duplicates", "thickening_duplicates")),
            "thickening_variance": _get_attr_safe("thickening_variance", "thickening_variance"),
            "thickening_depth_bias": _get_attr_safe("thickening_depth_bias", "thickening_depth_bias"),
            "planar_projection": _get_attr_safe("planar_projection", "planar_projection"),
            "input_camera_fov": _get_attr_safe("input_camera_fov", "input_camera_fov"),
            "render_mode": _get_attr_safe("render_mode", "render_mode"),
            "wavelet_packet_type": _get_attr_safe("wavelet_packet_type", "wavelet_packet_type"),
            "wavelet_packet_window_size": int(_get_attr_safe("wavelet_packet_window_size", "wavelet_packet_window_size")),
            "apply_depth_bias": _get_attr_safe("apply_depth_bias", "apply_depth_bias"),
            "depth_bias_map": self.depth_bias_map, 
            "size_scale_factor": _get_attr_safe("size_scale_factor", "size_scale_factor"),
            "depth_exponent": _get_attr_safe("depth_exponent", "depth_exponent"),
            "min_point_size": _get_attr_safe("min_point_size", "min_point_size"),
            "enable_max_size_clamp": _get_attr_safe("enable_max_size_clamp", "enable_max_size_clamp"),
            "max_point_size": _get_attr_safe("max_point_size", "max_point_size"),
            "input_scale_factor": _get_attr_safe("input_scale_factor", "input_scale_factor"),
            "camera_motion_confidence": self.camera_motion_confidence,
            "live_processing_mode": _get_attr_safe("live_processing_mode", "live_processing_mode"),
        }
        if hasattr(self, 'pending_bias_capture_request') and self.pending_bias_capture_request:
            self.edge_params_ref["trigger_bias_capture"] = self.pending_bias_capture_request
            self.pending_bias_capture_request = None
        else:
            self.edge_params_ref.pop("trigger_bias_capture", None)

    def update_playback_state(self):
        self.playback_state.update({
            "is_playing": getattr(self, 'is_playing', True),
            "speed": getattr(self, "playback_speed", DEFAULT_SETTINGS["playback_speed"]),
            "loop": getattr(self, "loop_video", DEFAULT_SETTINGS["loop_video"])
        })

    def update_recording_state(self):
        self.recording_state.update({
            "is_recording": getattr(self, "is_recording", DEFAULT_SETTINGS["is_recording"]),
            "output_dir": getattr(self, "recording_output_dir", DEFAULT_SETTINGS["recording_output_dir"])
        })

    def load_settings(self, filename="viewer_settings.json"):
        load_settings_from_file(self, filename)
        # Ensure dependent states are updated after loading
        if hasattr(self, 'dmd_time_window'):
            self.depth_history = deque(maxlen=self.dmd_time_window)
        if hasattr(self, '_update_edge_params'):
            self._update_edge_params()
        self.update_playback_state()
        self.update_recording_state()

    def save_settings(self, filename="viewer_settings.json"):
        save_settings_to_file(self, filename)

    def reset_settings(self):
        reset_app_settings(self)
        self.depth_history = deque(maxlen=self.dmd_time_window)
        self._update_edge_params()
        self.update_playback_state()
        self.update_recording_state()

    def _start_camera_preview(self):
        """REMOVED - Camera is now opened asynchronously."""
        pass

    def _initialize_default_point_cloud(self):
        """REMOVED - no more default point cloud visualization."""
        pass

    def _generate_test_pattern(self):
        """Generate an animated test pattern that looks like a camera feed."""
        import numpy as np
        
        # Create a 640x480 test pattern
        width, height = 640, 480
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create animated test pattern based on frame counter
        frame_offset = getattr(self, 'test_pattern_frame', 0)
        
        # Moving gradient bars
        for y in range(height):
            for x in range(width):
                # Moving rainbow pattern
                r = int(128 + 127 * np.sin((x + frame_offset * 2) * 0.02))
                g = int(128 + 127 * np.sin((y + frame_offset * 3) * 0.015))
                b = int(128 + 127 * np.sin((x + y + frame_offset * 4) * 0.01))
                pattern[y, x] = [r, g, b]
        
        # Add "CAMERA STARTING..." text overlay
        try:
            import cv2
            cv2.putText(pattern, "CAMERA LIVE", (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(pattern, f"Frame {frame_offset}", (20, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            pass  # If cv2 not available yet, just show pattern
        
        return pattern

    def _start_instant_test_pattern(self):
        """Start animated test pattern that updates immediately."""
        def test_pattern_worker():
            print("🎭 Test pattern animation starting INSTANTLY...")
            
            while self.camera_preview_active and not self.camera_ready:
                # Update test pattern animation
                self.test_pattern_frame += 1
                if not self.camera_ready:  # Only update if real camera not ready
                    self.latest_rgb_frame = self._generate_test_pattern()
                
                time.sleep(1.0/30.0)  # 30 FPS animation
            
            print("🎭 Test pattern stopped (real camera ready or preview stopped)")
                    
        # Start test pattern thread immediately
        self.test_pattern_thread = threading.Thread(target=test_pattern_worker, daemon=True)
        self.test_pattern_thread.start()
        print("✅ Test pattern animation started INSTANTLY!")

    def _update_debug_textures_async(self):
        """Update debug textures on main thread - OpenGL requires main thread context."""
        # OpenGL operations must happen on the main thread, so we do this synchronously
        # but only if enough time has passed to avoid blocking
        current_time = time.time()
        if not hasattr(self, '_last_texture_update'):
            self._last_texture_update = 0
        
        # Update textures at most 30 FPS to avoid blocking
        if current_time - self._last_texture_update > 1.0/30.0:
            try:
                if self.renderer and self.renderer.debug_textures_initialized:
                    self.renderer.update_all_debug_textures(
                        self.latest_rgb_frame,
                        self.latest_depth_map_viz,
                        self.latest_edge_map,
                        self.latest_smoothing_map,
                        self.latest_main_screen_coeff_viz_content,
                        self.latest_wavelet_map
                    )
                self._last_texture_update = current_time
            except Exception as e:
                # Don't spam errors, just continue
        pass

    def get_projection_and_view_matrices(self):
        view_matrix = self.camera.get_view_matrix()
        aspect = self.width / self.height if self.height > 0 else 1.0
        if self.use_orthographic:
            projection_matrix = Mat4.orthogonal_projection(-self.orthographic_size * aspect, self.orthographic_size * aspect, -self.orthographic_size, self.orthographic_size, 0.001, 10000.0)
        else:
            projection_matrix = Mat4.perspective_projection(aspect, z_near=0.01, z_far=10000.0, fov=60.0)
        return projection_matrix, view_matrix

    def get_camera_matrices(self):
        """Backward compatibility method - calls get_projection_and_view_matrices"""
        return self.get_projection_and_view_matrices()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        if hasattr(self, 'renderer') and self.renderer: 
            self.renderer.on_resize(width, height)

        # Update overlay label positions with safety checks
        if hasattr(self, 'fps_label') and self.fps_label:
        y_pos = height - 20
        self.fps_label.x = width - 10; self.fps_label.y = y_pos; y_pos -= 20
            if hasattr(self, 'points_label') and self.points_label:
        self.points_label.x = width - 10; self.points_label.y = y_pos; y_pos -= 20
            if hasattr(self, 'input_fps_label') and self.input_fps_label:
        self.input_fps_label.x = width - 10; self.input_fps_label.y = y_pos; y_pos -= 20
            if hasattr(self, 'depth_fps_label') and self.depth_fps_label:
        self.depth_fps_label.x = width - 10; self.depth_fps_label.y = y_pos; y_pos -= 20
            if hasattr(self, 'latency_label') and self.latency_label:
        self.latency_label.x = width - 10; self.latency_label.y = y_pos; y_pos -= 20
            if hasattr(self, 'dropped_frames_label') and self.dropped_frames_label:
        self.dropped_frames_label.x = width - 10; self.dropped_frames_label.y = y_pos

    def on_mouse_press(self, x, y, button, modifiers):
        io = imgui.get_io()
        if io.want_capture_mouse:
            if self.is_mouse_exclusive:
                self.set_exclusive_mouse(False)
                self.is_mouse_exclusive = False
            return pyglet.event.EVENT_HANDLED # ImGui handles it, stop propagation

        # ImGui does not want the mouse, app can process it
        if button == pyglet.window.mouse.LEFT:
            if not self.is_mouse_exclusive: # Only enter if not already in it
                self.set_exclusive_mouse(True)
                self.is_mouse_exclusive = True
        # Do not return EVENT_HANDLED here unless this specific app action should consume it exclusively

    def on_mouse_release(self, x, y, button, modifiers):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return pyglet.event.EVENT_HANDLED # ImGui handles it

        # Match original behavior - release exclusive mouse on left button release
        if button == pyglet.window.mouse.LEFT and self.is_mouse_exclusive:
            self.set_exclusive_mouse(False)
            self.is_mouse_exclusive = False

    def on_mouse_motion(self, x, y, dx, dy):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return pyglet.event.EVENT_HANDLED

        if self.is_mouse_exclusive: # Our app's flag for camera control
            self.camera.on_mouse_motion(dx, dy, self.is_mouse_exclusive)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return pyglet.event.EVENT_HANDLED

        if self.is_mouse_exclusive and (buttons & pyglet.window.mouse.LEFT):
            self.camera.on_mouse_motion(dx, dy, self.is_mouse_exclusive) # Use same logic as motion

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return pyglet.event.EVENT_HANDLED
        self.camera.on_mouse_scroll(scroll_y)

    def on_key_press(self, symbol, modifiers):
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return pyglet.event.EVENT_HANDLED

        if symbol == pyglet.window.key.ESCAPE:
            if self.is_mouse_exclusive:
                self.set_exclusive_mouse(False)
                self.is_mouse_exclusive = False
                return pyglet.event.EVENT_HANDLED # ESC consumed for releasing mouse
            else:
                self.close() # Close app if ESC pressed and mouse not captured
                return pyglet.event.EVENT_HANDLED # ESC consumed for closing
        
        self.keys.on_key_press(symbol, modifiers) # Update app's key state handler for camera

    def on_key_release(self, symbol, modifiers):
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return pyglet.event.EVENT_HANDLED
        self.keys.on_key_release(symbol, modifiers)

    def on_text(self, text):
        io = imgui.get_io()
        if io.want_capture_keyboard: # Technically, text input implies ImGui widget is active
            # Event handlers pushed by PygletRenderer for ImGui usually handle on_text.
            # If this method is still reached and ImGui wants keyboard, it's likely already handled.
            # To be safe, let ImGui handle it if it claims to want keyboard focus.
            return pyglet.event.EVENT_HANDLED 
        # No app-specific text input here

    def on_text_motion(self, motion):
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return pyglet.event.EVENT_HANDLED
        # No app-specific text motion here

    def _browse_media_file(self):
        root = tk.Tk(); root.withdraw()
        file_path = filedialog.askopenfilename(title="Select Video or Image File", filetypes=[("Media Files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")] )
        if file_path: self._switch_input_source("File", file_path)
        root.destroy()

    def _browse_glb_sequence(self):
        root = tk.Tk(); root.withdraw()
        dir_path = filedialog.askdirectory(title="Select GLB Sequence Directory")
        if dir_path: self._switch_input_source("GLB Sequence", dir_path)
        root.destroy()

    def _switch_input_source(self, mode, filepath=None, monitor_index=0):
        self.input_mode = mode
        self.input_filepath = filepath if filepath else ""
        if mode == "Screen": self.screen_capture_monitor_index = monitor_index
        self.status_message = f"{mode} selected"
        self.is_playing = True # Default to playing for new source
        self.playback_state["current_frame"] = 0
        self.playback_state["total_frames"] = 0
        self.playback_state["restart"] = False
        self.update_playback_state() # Update the shared dict
        
        # Update async inference manager if available
        if hasattr(self, 'async_inference_manager'):
            inference_settings = {
                'input_mode': self.input_mode,
                'input_filepath': self.input_filepath,
                'screen_capture_monitor_index': monitor_index,
                'playback_state': self.playback_state
            }
            self.async_inference_manager.update_settings(inference_settings)
        else:
            # Fallback to legacy method
        self.start_inference_thread()

    # Legacy method compatibility
    def start_inference_thread(self):
        """Legacy method - now delegates to async inference manager."""
        if hasattr(self, 'async_inference_manager'):
            # Use async inference manager
            inference_settings = getattr(self, 'async_settings', {})
            inference_settings.update({
                'input_mode': getattr(self, 'input_mode', 'Live'),
                'input_filepath': getattr(self, 'input_filepath', ''),
                'live_processing_mode': getattr(self, 'live_processing_mode', True),
                'screen_capture_monitor_index': getattr(self, 'screen_capture_monitor_index', 0),
                'playback_state': getattr(self, 'playback_state', {}),
                'recording_state': getattr(self, 'recording_state', {}),
                'edge_params': getattr(self, 'edge_params_ref', {}),
                'input_scale_factor': getattr(self, 'input_scale_factor', 1.0)
            })
            self.async_inference_manager.start_inference(inference_settings)
        else:
            print("DEBUG: Async inference manager not ready, inference will start when available")

class AsyncInferenceManager:
    """Completely isolated model inference manager - never blocks main thread."""
    
    def __init__(self, model, device, inference_interval, target_fps, data_queue, exit_event):
        self.model = model
        self.device = device
        self.inference_interval = inference_interval
        self.target_fps = target_fps
        self.data_queue = data_queue
        self.exit_event = exit_event
        
        # Inference state - completely isolated
        self.inference_active = False
        self.inference_thread = None
        self.inference_stats = {
            'fps': 0.0,
            'frames_processed': 0,
            'last_frame_time': time.time(),
            'processing_time': 0.0
        }
        
        # Input management - completely async
        self.input_queue = queue.Queue(maxsize=5)  # Small buffer to prevent backup
        self.output_queue = queue.Queue(maxsize=3)  # Output results
        
        # Thread-safe settings
        self.settings_lock = Lock()
        self.current_settings = {}
        
        print("✅ AsyncInferenceManager initialized")
    
    def start_inference(self, settings):
        """Start inference in completely isolated thread."""
        with self.settings_lock:
            self.current_settings = settings.copy()
        
        if self.inference_thread and self.inference_thread.is_alive():
            print("🔄 Restarting inference thread...")
            self.stop_inference()
        
        self.inference_active = True
        self.inference_thread = Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        print("🚀 Isolated inference thread started!")
    
    def stop_inference(self):
        """Stop inference thread gracefully."""
        self.inference_active = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        print("🛑 Inference thread stopped")
    
    def update_settings(self, settings):
        """Update settings non-blocking."""
        with self.settings_lock:
            self.current_settings = settings.copy()
    
    def submit_frame(self, frame_data):
        """Submit frame for processing (non-blocking)."""
        try:
            self.input_queue.put_nowait(frame_data)
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait(frame_data)
            except queue.Empty:
                pass
    
    def get_results(self):
        """Get inference results (non-blocking)."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_stats(self):
        """Get inference statistics."""
        return self.inference_stats.copy()
    
    def _inference_worker(self):
        """Completely isolated inference worker - never blocks anything."""
        print("🤖 Isolated inference worker starting...")
        
        try:
            from unik3d_viewer.inference_logic import inference_thread_func
            
            # Get current settings
            with self.settings_lock:
                settings = self.current_settings.copy()
            
            # Create isolated inference parameters
            scale_factor_ref = [settings.get('input_scale_factor', 1.0)]
            edge_params_ref = settings.get('edge_params', {})
            playback_state = settings.get('playback_state', {})
            recording_state = settings.get('recording_state', {})
            
            # Run inference with isolated data queues
            inference_thread_func(
                self.data_queue, self.exit_event, self.model, self.device,
                self.inference_interval, scale_factor_ref, edge_params_ref,
                settings.get('input_mode', 'Live'), 
                settings.get('input_filepath', ''),
                playback_state, recording_state,
                settings.get('live_processing_mode', True),
                settings.get('screen_capture_monitor_index', 0),
                self.target_fps
            )
            
            except Exception as e:
            print(f"❌ Inference worker error: {e}")
            traceback.print_exc()
        finally:
            print("🤖 Inference worker finished")

class AsyncUIManager:
    """Completely isolated UI/ImGui manager - never blocks inference."""
    
    def __init__(self, window, viewer):
        self.window = window
        self.viewer = viewer
        self.ui_active = True
        
        # UI state - completely isolated from inference
        self.ui_thread = None
        self.ui_data = {
            'status_message': "UI Ready",
            'fps': 0.0,
            'vertex_count': 0,
            'camera_frame': None
        }
        self.ui_lock = Lock()
        
        # UI update queue
        self.ui_update_queue = queue.Queue(maxsize=10)
        
        print("✅ AsyncUIManager initialized")
    
    def start_ui_thread(self):
        """Start UI processing in background thread."""
        self.ui_thread = Thread(target=self._ui_worker, daemon=True)
        self.ui_thread.start()
        print("🎨 Isolated UI thread started!")
    
    def stop_ui_thread(self):
        """Stop UI thread."""
        self.ui_active = False
        if self.ui_thread:
            self.ui_thread.join(timeout=1.0)
        print("🛑 UI thread stopped")
    
    def update_ui_data(self, data):
        """Update UI data (non-blocking)."""
        try:
            self.ui_update_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest update if queue is full
            try:
                self.ui_update_queue.get_nowait()
                self.ui_update_queue.put_nowait(data)
            except queue.Empty:
                pass
    
    def get_ui_data(self):
        """Get current UI data (thread-safe)."""
        with self.ui_lock:
            return self.ui_data.copy()
    
    def _ui_worker(self):
        """UI processing worker - handles UI updates asynchronously."""
        print("🎨 UI worker starting...")
        
        while self.ui_active:
            try:
                # Process UI updates
                try:
                    ui_update = self.ui_update_queue.get(timeout=0.1)
                    with self.ui_lock:
                        self.ui_data.update(ui_update)
                except queue.Empty:
                    continue
                
                # Non-blocking UI processing can happen here
                time.sleep(1.0/60.0)  # 60 FPS UI updates
                
            except Exception as e:
                print(f"❌ UI worker error: {e}")
        
        print("🎨 UI worker finished")

class AsyncWindowManager:
    """Completely isolated window manager - handles OpenGL/rendering separately."""
    
    def __init__(self, window):
        self.window = window
        self.render_active = True
        
        # Rendering state - isolated from inference and UI logic
        self.render_data = {
            'vertices': None,
            'colors': None,
            'vertex_count': 0,
            'camera_matrices': None,
            'render_settings': {}
        }
        self.render_lock = Lock()
        
        # Render update queue
        self.render_queue = queue.Queue(maxsize=5)
        
        print("✅ AsyncWindowManager initialized")
    
    def update_render_data(self, data):
        """Update rendering data (non-blocking)."""
        try:
            self.render_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest render data if queue is full
            try:
                self.render_queue.get_nowait()
                self.render_queue.put_nowait(data)
            except queue.Empty:
                pass
    
    def process_render_updates(self):
        """Process render updates on main thread (OpenGL requirement)."""
        # This MUST run on main thread for OpenGL
        try:
            while True:
                render_update = self.render_queue.get_nowait()
                with self.render_lock:
                    self.render_data.update(render_update)
        except queue.Empty:
            pass  # No updates available
    
    def get_render_data(self):
        """Get current render data (thread-safe)."""
        with self.render_lock:
            return self.render_data.copy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live SLAM viewer using UniK3D.")
    parser.add_argument("--model", type=str, default="unik3d-vitl", help="Name of the UniK3D model.")
    parser.add_argument("--interval", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target FPS for inference.")
    args = parser.parse_args()

    config = pyglet.gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True)
    try:
        window = LiveViewerWindow(model_name=args.model, inference_interval=args.interval,
                                  target_inference_fps=args.target_fps,
                                  width=1280, height=720, caption='UniK3D Live Viewer',
                                  resizable=True, vsync=False, config=config)
    except pyglet.window.NoSuchConfigException:
        print("Warning: Desired GL config not available. Falling back to default.")
        window = LiveViewerWindow(model_name=args.model, inference_interval=args.interval,
                                  target_inference_fps=args.target_fps,
                                  width=1280, height=720, caption='UniK3D Live Viewer',
                                  resizable=True, vsync=False)
    
    try:
        pyglet.app.run()
    except Exception as e_run:
        print(f"Unhandled exception during pyglet.app.run(): {e_run}")
        traceback.print_exc()
    finally:
        print("Application run loop finished or exited via exception.")
        # Ensure cleanup is robustly called, though on_close should handle most
        if hasattr(window, 'renderer') and window.renderer: window.renderer.cleanup()
        if hasattr(window, 'ui_manager') and window.ui_manager: window.ui_manager.shutdown()
        # Note: ImGui context destruction is tricky. If UIManager.shutdown doesn't, 
        # and pyglet doesn't when exiting, it might be left. Usually, pyglet handles it.
        # Forcing imgui.destroy_context() here can cause issues if already destroyed.
        print("Exiting application.") 