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
    """Manages GPU resources and CUDA streams for parallel processing."""
    
    def __init__(self, device):
        self.device = device
        self.inference_stream = None
        self.rendering_stream = None
        self.texture_stream = None
        
        if torch.cuda.is_available():
            # Create separate CUDA streams for true parallelism
            self.inference_stream = torch.cuda.Stream()
            self.rendering_stream = torch.cuda.Stream()
            self.texture_stream = torch.cuda.Stream()
            print("Created CUDA streams: inference, rendering, texture")
        
        # GPU memory pools for efficient allocation
        self.vertex_pool_a = None
        self.vertex_pool_b = None
        self.color_pool_a = None
        self.color_pool_b = None
        self.current_pool = 'a'
        self.pool_lock = Lock()
        
        self._initialize_gpu_pools()
    
    def _initialize_gpu_pools(self):
        """Pre-allocate GPU memory pools for vertex data."""
        if torch.cuda.is_available():
            # Pre-allocate large buffers on GPU - INCREASED SIZE to handle real data
            max_vertices = 500000  # 500K vertices max (was 1M, but let's be more conservative)
            with torch.cuda.stream(self.rendering_stream):
                self.vertex_pool_a = torch.zeros((max_vertices, 3), device=self.device, dtype=torch.float32)
                self.vertex_pool_b = torch.zeros((max_vertices, 3), device=self.device, dtype=torch.float32)
                self.color_pool_a = torch.zeros((max_vertices, 3), device=self.device, dtype=torch.float32)
                self.color_pool_b = torch.zeros((max_vertices, 3), device=self.device, dtype=torch.float32)
            print(f"Initialized GPU memory pools for {max_vertices} vertices")
    
    def get_current_buffers(self):
        """Get current rendering buffers (thread-safe)."""
        with self.pool_lock:
            if self.current_pool == 'a':
                return self.vertex_pool_a, self.color_pool_a
            else:
                return self.vertex_pool_b, self.color_pool_b
    
    def get_update_buffers(self):
        """Get buffers for updating (opposite of current)."""
        with self.pool_lock:
            if self.current_pool == 'a':
                return self.vertex_pool_b, self.color_pool_b
            else:
                return self.vertex_pool_a, self.color_pool_a
    
    def swap_buffers(self):
        """Swap active buffer (thread-safe)."""
        with self.pool_lock:
            self.current_pool = 'b' if self.current_pool == 'a' else 'a'

class AsyncDataProcessor:
    """Handles data processing in separate thread to avoid blocking main thread."""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.data_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory buildup
        self.processed_queue = queue.Queue(maxsize=3)  # Small queue for processed results
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
        """Worker thread for processing data."""
        while not self.exit_event.is_set():
            try:
                # Get data with timeout to allow checking exit event
                data = self.data_queue.get(timeout=0.1)
                
                # Process data using GPU stream
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
        """Process data using GPU streams for TRUE parallelism."""
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
            
            print(f"DEBUG: _process_data_gpu - received {num_vertices} vertices, vertices_data shape: {vertices_data.shape if vertices_data is not None else None}")
            
            # Use texture stream for non-critical processing
            with torch.cuda.stream(self.gpu_manager.texture_stream):
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
            
            # Use rendering stream for vertex data processing - KEEP ON GPU
            vertex_count = 0
            gpu_vertices = None
            gpu_colors = None
            
            if vertices_data is not None and colors_data is not None and num_vertices > 0:
                with torch.cuda.stream(self.gpu_manager.rendering_stream):
                    try:
                        # Convert to GPU tensors and keep them on GPU
                        vertices_tensor = torch.from_numpy(vertices_data.reshape(-1, 3)).to(self.gpu_manager.device)
                        colors_tensor = torch.from_numpy(colors_data.reshape(-1, 3)).to(self.gpu_manager.device)
                        
                        gpu_vertices = vertices_tensor
                        gpu_colors = colors_tensor
                        vertex_count = num_vertices
                        
                        print(f"DEBUG: Created GPU tensors - vertices: {gpu_vertices.shape}, colors: {gpu_colors.shape}, count: {vertex_count}")
                    except Exception as e:
                        print(f"Error creating GPU tensors: {e}")
                        vertex_count = 0
                        gpu_vertices = None
                        gpu_colors = None
            
            result = {
                'vertex_count': vertex_count,
                'gpu_vertices': gpu_vertices,
                'gpu_colors': gpu_colors,
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
            
            print(f"DEBUG: _process_data_gpu returning vertex_count: {vertex_count}")
            return result
            
        except Exception as e:
            print(f"Error processing data in GPU worker: {e}")
            traceback.print_exc()
            return None

class AsyncRenderer:
    """Handles rendering with zero blocking - viewer FPS completely independent of inference."""
    
    def __init__(self, renderer, gpu_manager):
        self.renderer = renderer
        self.gpu_manager = gpu_manager
        self.render_lock = Lock()
        
        # CRITICAL: Pre-allocate CPU vertex buffers for OpenGL (no GPU sync needed)
        max_vertices = 500000
        self.cpu_vertices_buffer_a = np.zeros(max_vertices * 3, dtype=np.float32)
        self.cpu_colors_buffer_a = np.zeros(max_vertices * 3, dtype=np.float32)
        self.cpu_vertices_buffer_b = np.zeros(max_vertices * 3, dtype=np.float32)
        self.cpu_colors_buffer_b = np.zeros(max_vertices * 3, dtype=np.float32)
        self.current_cpu_buffer = 'a'
        
        # Viewer state - completely independent of inference
        self.viewer_vertex_count = 2500  # Start with small default
        self.viewer_ready = False
        self.last_gpu_update_time = 0
        
        # Background GPU-to-CPU worker (completely separate)
        self.gpu_cpu_worker = None
        self.gpu_cpu_queue = queue.Queue(maxsize=2)  # Very small queue
        self.gpu_cpu_active = True
        self._start_gpu_cpu_worker()
        
        # Performance tracking
        self.render_fps = 0.0
        self.last_render_time = time.time()
        self.render_frame_count = 0
        
        print(f"AsyncRenderer initialized with independent CPU buffers for {max_vertices} vertices")
    
    def _start_gpu_cpu_worker(self):
        """Start background worker to transfer GPU data to CPU without blocking viewer."""
        def gpu_cpu_worker():
            print("🔄 GPU-to-CPU worker starting (completely independent)...")
            
            while self.gpu_cpu_active:
                try:
                    # Wait for GPU data update request (non-blocking)
                    gpu_data = self.gpu_cpu_queue.get(timeout=0.1)
                    if gpu_data is None:  # Shutdown signal
                        break
                    
                    gpu_vertices, gpu_colors, vertex_count = gpu_data
                    
                    # BACKGROUND: Convert GPU to CPU (doesn't block viewer)
                    start_time = time.time()
                    if torch.cuda.is_available():
                        with torch.cuda.stream(self.gpu_manager.texture_stream):  # Use texture stream
                            vertices_cpu = gpu_vertices[:vertex_count].cpu().numpy().flatten()
                            colors_cpu = gpu_colors[:vertex_count].cpu().numpy().flatten()
                    else:
                        vertices_cpu = gpu_vertices[:vertex_count].numpy().flatten()
                        colors_cpu = gpu_colors[:vertex_count].numpy().flatten()
                    
                    # Update CPU buffers (atomic swap)
                    with self.render_lock:
                        if self.current_cpu_buffer == 'a':
                            # Update buffer B while viewer uses A
                            buffer_size = min(len(vertices_cpu), len(self.cpu_vertices_buffer_b))
                            self.cpu_vertices_buffer_b[:buffer_size] = vertices_cpu[:buffer_size]
                            color_size = min(len(colors_cpu), len(self.cpu_colors_buffer_b))
                            self.cpu_colors_buffer_b[:color_size] = colors_cpu[:color_size]
                            # Atomic swap
                            self.current_cpu_buffer = 'b'
                        else:
                            # Update buffer A while viewer uses B
                            buffer_size = min(len(vertices_cpu), len(self.cpu_vertices_buffer_a))
                            self.cpu_vertices_buffer_a[:buffer_size] = vertices_cpu[:buffer_size]
                            color_size = min(len(colors_cpu), len(self.cpu_colors_buffer_a))
                            self.cpu_colors_buffer_a[:color_size] = colors_cpu[:color_size]
                            # Atomic swap
                            self.current_cpu_buffer = 'a'
                        
                        self.viewer_vertex_count = vertex_count
                        self.viewer_ready = True
                        self.last_gpu_update_time = time.time()
                    
                    convert_time = time.time() - start_time
                    if convert_time > 0.016:  # Log if > 16ms (slower than 60 FPS)
                        print(f"🐌 GPU-to-CPU conversion took {convert_time:.3f}s for {vertex_count} vertices")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ GPU-to-CPU worker error: {e}")
            
            print("🔄 GPU-to-CPU worker stopped")
        
        self.gpu_cpu_worker = Thread(target=gpu_cpu_worker, daemon=True)
        self.gpu_cpu_worker.start()
        print("✅ GPU-to-CPU worker started (background only)")
    
    def update_vertex_data_gpu_async(self, vertices_tensor, colors_tensor, vertex_count):
        """Submit GPU data for background processing (never blocks viewer)."""
        if vertices_tensor is not None and colors_tensor is not None and vertex_count > 0:
            try:
                # Submit to background worker (non-blocking)
                gpu_data = (vertices_tensor, colors_tensor, vertex_count)
                self.gpu_cpu_queue.put_nowait(gpu_data)
                print(f"🔄 Submitted {vertex_count} vertices to background GPU-to-CPU worker")
            except queue.Full:
                # Drop if queue is full (viewer keeps running smoothly)
                print("⚠️ GPU-to-CPU queue full, dropping frame (viewer unaffected)")
    
    def render_frame(self):
        """Render frame using cached CPU data - NEVER blocks, always runs at refresh rate."""
        current_time = time.time()
        
        # Get current CPU data (atomic read)
        with self.render_lock:
            vertex_count = self.viewer_vertex_count
            ready = self.viewer_ready
            if self.current_cpu_buffer == 'a':
                vertices_data = self.cpu_vertices_buffer_a
                colors_data = self.cpu_colors_buffer_a
            else:
                vertices_data = self.cpu_vertices_buffer_b
                colors_data = self.cpu_colors_buffer_b
            
            # Get latest matrices if available
            view_mat = getattr(self, 'view_matrix', None)
            proj_mat = getattr(self, 'projection_matrix', None)
            settings = getattr(self, 'render_settings', {})
        
        # ALWAYS render something - never wait for GPU
        if ready and vertex_count > 0:
            try:
                # Use cached CPU data directly (zero GPU sync!)
                vertices_to_render = vertices_data[:vertex_count * 3]
                colors_to_render = colors_data[:vertex_count * 3]
                
                # Update OpenGL renderer (main thread only) - pure CPU operation
                if len(vertices_to_render) == vertex_count * 3 and len(colors_to_render) == vertex_count * 3:
                    self.renderer.update_vertex_list(
                        vertices_to_render, colors_to_render, vertex_count, 
                        settings, view_mat  # Use proper matrices when available
                    )
                
            except Exception as e:
                print(f"Error in CPU-only rendering: {e}")
                # Graceful fallback - still render something
                pass
        
        # ALWAYS render the scene regardless of data availability
        if hasattr(self.renderer, 'render_scene'):
            # Use proper matrices if available, fallback to identity
            if proj_mat is not None and view_mat is not None:
                latest_rgb_shape = (480, 640, 3)  # Default shape
                self.renderer.render_scene(proj_mat, view_mat, settings, latest_rgb_shape)
            else:
                # Fallback for initial frames
                identity_matrix = Mat4()
                self.renderer.render_scene(identity_matrix, identity_matrix, {}, (480, 640, 3))
        
        # Update FPS (viewer-only FPS, independent of inference)
        self.render_frame_count += 1
        if current_time - self.last_render_time >= 1.0:
            self.render_fps = self.render_frame_count / (current_time - self.last_render_time)
            self.render_frame_count = 0
            self.last_render_time = current_time
            
            # Log viewer FPS (should be ~refresh rate)
            if self.render_fps > 0:
                print(f"🖥️ VIEWER FPS: {self.render_fps:.1f} (independent of inference)")
    
    def update_render_data(self, vertex_count, render_settings, view_matrix, projection_matrix):
        """Update rendering data (thread-safe) - for camera/view updates only."""
        # Store matrices for scene rendering
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix
        self.render_settings = render_settings.copy()
    
    def stop(self):
        """Stop background GPU-to-CPU worker."""
        self.gpu_cpu_active = False
        try:
            self.gpu_cpu_queue.put_nowait(None)  # Shutdown signal
        except queue.Full:
            pass
        if self.gpu_cpu_worker:
            self.gpu_cpu_worker.join(timeout=1.0)

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
            
        print("🎯 Initializing main thread components with async managers...")
        
        # Initialize core OpenGL components (must be on main thread)
        self.renderer = Renderer(self.width, self.height)
        self.ui_manager = UIManager(self, self)
        self._initialize_text_overlays()
        
        # Initialize GPU components
        self.gpu_manager = GPUResourceManager(self.device)
        self.data_processor = AsyncDataProcessor(self.gpu_manager)
        self.async_renderer = AsyncRenderer(self.renderer, self.gpu_manager)
        
        # Initialize async managers for complete separation
        print("🚀 Initializing isolated async managers...")
        
        # 1. Window Manager (main thread OpenGL operations)
        self.async_window_manager = AsyncWindowManager(self)
        
        # 2. UI Manager (background UI processing)
        self.async_ui_manager = AsyncUIManager(self, self)
        self.async_ui_manager.start_ui_thread()
        
        # 3. Inference Manager (completely isolated model inference)
        # Will be initialized when model is ready
        
        # Start data processing
        self.data_processor.start()
        
        # Clean state - no default visualization
        print("✅ Viewer initialized for max FPS - no default point cloud")
        
        self.main_components_initialized = True
        print("✅ Main thread components with async managers ready!")
        
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
        """Main update loop - coordinates isolated async managers without blocking."""
        # FIRST: Initialize main thread components when parallel setup is ready
        self._initialize_main_thread_components_when_ready()
        
        # SECOND: Start isolated inference when both model and components are ready
        self._start_inference_when_ready()
        
        # THIRD: Handle async window manager render updates (main thread OpenGL requirement)
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
            self._update_async_renderer_non_blocking()

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
        """Apply processed data to async managers without blocking."""
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
        
        # Update render data through window manager (non-blocking)
        vertex_count = processed_data.get('vertex_count', 0)
        gpu_vertices = processed_data.get('gpu_vertices')
        gpu_colors = processed_data.get('gpu_colors')
        
        if vertex_count > 0 and gpu_vertices is not None and gpu_colors is not None:
            # Send render data to async window manager
            render_update = {
                'vertex_count': vertex_count,
                'gpu_vertices': gpu_vertices,
                'gpu_colors': gpu_colors
            }
            if hasattr(self, 'async_window_manager'):
                self.async_window_manager.update_render_data(render_update)
            
            # Update async renderer (non-blocking GPU operations)
            self.async_renderer.update_vertex_data_gpu_async(gpu_vertices, gpu_colors, vertex_count)
            self._current_vertex_count = vertex_count

    def _update_async_ui_data(self):
        """Update UI data for async UI manager."""
        if not hasattr(self, 'async_ui_manager'):
            return
        
        # Collect current application state for UI
        ui_data = {
            'status_message': getattr(self, 'status_message', 'Ready'),
            'vertex_count': getattr(self, '_current_vertex_count', 0),
            'camera_frame': getattr(self, 'latest_rgb_frame', None),
            'render_fps': getattr(self.async_renderer, 'render_fps', 0.0) if hasattr(self, 'async_renderer') else 0.0,
            'input_fps': getattr(self, 'input_fps', 0.0),
            'depth_fps': getattr(self, 'depth_fps', 0.0)
        }
        
        # Send to async UI manager (non-blocking)
        self.async_ui_manager.update_ui_data(ui_data)

    def _update_async_renderer_non_blocking(self):
        """Update async renderer without blocking any threads."""
        if not hasattr(self, 'async_renderer'):
            return
        
        # Prepare render settings (main thread)
        current_render_settings = {key: getattr(self, key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
        
        # Get current camera matrices (main thread)
        projection_matrix, view_matrix = self.get_projection_and_view_matrices()
        
        # Update async renderer matrices for better visual quality (no blocking)
        vertex_count = getattr(self, '_current_vertex_count', 0)
        self.async_renderer.update_render_data(
            vertex_count, current_render_settings, view_matrix, projection_matrix
        )

        # Send render data to window manager (non-blocking)
        if hasattr(self, 'async_window_manager'):
            render_data = {
                'camera_matrices': (projection_matrix, view_matrix),
                'render_settings': current_render_settings,
                'vertex_count': vertex_count
            }
            self.async_window_manager.update_render_data(render_data)

    def on_draw(self):
        """Main rendering method - now runs at monitor refresh rate independent of inference."""
        # Clear the screen
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        
        # If components are ready, do high-FPS async rendering
        if hasattr(self, 'main_components_initialized') and self.main_components_initialized:
            # PRIMARY: ALWAYS render using async renderer - ZERO blocking, runs at refresh rate
            self.async_renderer.render_frame()
            
            # SECONDARY: Lightweight debug rendering (only if needed)
            if any(getattr(self, s, False) for s in DEFAULT_SETTINGS if "debug_show" in s):
                projection_matrix, view_matrix = self.get_projection_and_view_matrices()
                debug_settings = {s: getattr(self, s, DEFAULT_SETTINGS[s]) for s in DEFAULT_SETTINGS if "debug_show" in s or s == "input_camera_fov"}
                latest_rgb_shape = self.latest_rgb_frame.shape if self.latest_rgb_frame is not None else (self.height, self.width)
                
                self.renderer.draw_debug_geometry(
                    projection_matrix, view_matrix, debug_settings, 
                    self.latest_points_for_debug, latest_rgb_shape
                )
            
            # Handle special rendering modes (lightweight check)
            if getattr(self, 'render_mode', 0) == 3:
                self.renderer.render_wavelet_fft_fullscreen()
            
            # UI rendering - lightweight, uses async data
            self._render_async_ui()
            
            # Text overlays - minimal overhead
            self._render_async_overlay_labels()
            
            # Background texture updates (non-blocking, throttled)
            self._update_debug_textures_async()
        else:
            # Minimal rendering while components are loading
            if hasattr(self, 'latest_rgb_frame') and self.latest_rgb_frame is not None:
                # Show instant test pattern
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

    def _render_async_overlay_labels(self):
        """Render text overlays using async UI data."""
        if not hasattr(self, 'async_ui_manager'):
            return
        
        # Get UI data from async manager
        ui_data = self.async_ui_manager.get_ui_data()
        
        # Update overlay labels with async data
        if hasattr(self, 'fps_label') and self.show_fps_overlay:
            render_fps = ui_data.get('render_fps', 0.0)
            self.fps_label.text = f"Render FPS: {render_fps:.1f}"
            self.fps_label.visible = True
        else:
            if hasattr(self, 'fps_label'):
                self.fps_label.visible = False
        
        if hasattr(self, 'points_label') and self.show_points_overlay:
            vertex_count = ui_data.get('vertex_count', 0)
            self.points_label.text = f"Points: {vertex_count}"
            self.points_label.visible = True
        else:
            if hasattr(self, 'points_label'):
                self.points_label.visible = False
        
        if hasattr(self, 'input_fps_label') and self.show_input_fps_overlay:
            input_fps = ui_data.get('input_fps', 0.0)
            self.input_fps_label.text = f"Input FPS: {input_fps:.1f}"
            self.input_fps_label.visible = True
        else:
            if hasattr(self, 'input_fps_label'):
                self.input_fps_label.visible = False
        
        if hasattr(self, 'depth_fps_label') and self.show_depth_fps_overlay:
            depth_fps = ui_data.get('depth_fps', 0.0)
            self.depth_fps_label.text = f"Depth FPS: {depth_fps:.1f}"
            self.depth_fps_label.visible = True
        else:
            if hasattr(self, 'depth_fps_label'):
                self.depth_fps_label.visible = False
        
        if hasattr(self, 'latency_label') and self.show_latency_overlay:
            latency_ms = ui_data.get('latency_ms', 0.0)
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
        """Clean shutdown of all async components and resources."""
        print("Window closing, shutting down isolated async components...")
        
        # Stop async renderer background worker first
        if hasattr(self, 'async_renderer') and self.async_renderer:
            print("Stopping async renderer GPU-to-CPU worker...")
            self.async_renderer.stop()
        
        # Stop async managers
        if hasattr(self, 'async_ui_manager'):
            print("Stopping async UI manager...")
            self.async_ui_manager.stop_ui_thread()
        
        if hasattr(self, 'async_inference_manager'):
            print("Stopping isolated inference manager...")
            self.async_inference_manager.stop_inference()
        
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
        
        # Synchronize CUDA streams before cleanup
        if torch.cuda.is_available() and hasattr(self, 'gpu_manager') and self.gpu_manager:
            print("Synchronizing CUDA streams...")
            try:
                if self.gpu_manager.inference_stream:
                    self.gpu_manager.inference_stream.synchronize()
                if self.gpu_manager.rendering_stream:
                    self.gpu_manager.rendering_stream.synchronize()
                if self.gpu_manager.texture_stream:
                    self.gpu_manager.texture_stream.synchronize()
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