import pyglet
import pyglet.gl as gl
import trimesh
import numpy as np
import os
import glob
import argparse
import math
# Removed threading, time, queue imports
from pyglet.math import Mat4, Vec3
from pyglet.window import key # Import key for KeyStateHandler
import traceback # Import traceback
from collections import deque # For view matrix smoothing

# Simple shaders (Using 'vertices' instead of 'position')
# Expects vec3 floats for both vertices and colors
vertex_source = """#version 150 core
    in vec3 vertices;
    in vec3 colors; // Expects normalized floats (0.0-1.0)

    out vec3 vertex_colors;

    uniform mat4 projection;
    uniform mat4 view;

    void main() {
        gl_Position = projection * view * vec4(vertices, 1.0);
        vertex_colors = colors; // Pass color data through
        gl_PointSize = 2.0; // Fixed size
    }
"""

fragment_source = """#version 150 core
    in vec3 vertex_colors;
    out vec4 final_color;

    void main() {
        // vertex_colors arrive as normalized floats (0.0-1.0)
        final_color = vec4(vertex_colors, 1.0);
    }
"""

class PlayerWindow(pyglet.window.Window):
    def __init__(self, glb_dir, num_frames_to_load=-1, smooth_window=5, *args, **kwargs): # Default to load all, add smoothing window
        super().__init__(*args, **kwargs)
        self.glb_dir = glb_dir
        self.smooth_window = smooth_window # Keep for potential future use, but not used in this version
        # Limit the number of files to process if num_frames_to_load > 0
        all_files = sorted(glob.glob(os.path.join(self.glb_dir, "*.glb")))
        if num_frames_to_load > 0:
            self.glb_files = all_files[:num_frames_to_load]
            print(f"DEBUG: Limiting load to first {len(self.glb_files)} frames.")
        else:
            self.glb_files = all_files
            print(f"DEBUG: Loading all {len(self.glb_files)} frames.")


        if not self.glb_files:
            print(f"Error: No .glb files found in directory: {self.glb_dir}")
            pyglet.app.exit()
            return

        self.current_frame_index = 0 # Start at frame 0
        self.vertex_list = None
        self.playing = True
        self.loop = True
        self.frame_rate = 30.0 # Target playback FPS

        # --- Load frames upfront ---
        self.all_vertices_list = []
        self.all_colors_list = []
        self.frame_counts = []
        self.frame_starts = []
        self.total_vertices = 0
        # Use the original loading method (no smoothing applied during load)
        if not self._load_all_frames_to_ram():
             pyglet.app.exit()
             return

        if self.total_vertices == 0:
             print("Error: No points found in any loaded frames.")
             pyglet.app.exit()
             return

        print(f"Loaded {len(self.glb_files)} frames. Total vertices: {self.total_vertices}")
        print("Concatenating final data...")
        # Concatenate all data into single large numpy arrays
        final_vertices_data = np.concatenate(self.all_vertices_list)
        final_colors_data = np.concatenate(self.all_colors_list)
        # Clear temporary lists to free RAM
        del self.all_vertices_list
        del self.all_colors_list
        print("Data concatenated.")
        # -----------------------------

        # --- Camera State ---
        self.camera_position = Vec3(0.0, 1.0, 5.0) # Start slightly up and back
        self.camera_rotation_x = -10.0 # Initial downward tilt (pitch)
        self.camera_rotation_y = 0.0   # Yaw
        self.world_up_vector = Vec3(0, 1, 0) # Define world up
        self.move_speed = 5.0
        self.fast_move_speed = 15.0
        # --- View Matrix Smoothing ---
        self.view_matrix_history = deque(maxlen=5) # Store last N view matrices
        # ---------------------------

        # --- Input Handlers ---
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.mouse_down = False
        # No need for last_mouse_x/y with exclusive mouse
        # --------------------

        # Batch (for UI), Shader, and SINGLE STATIC Vertex List (NOT in batch)
        self.batch = pyglet.graphics.Batch() # Keep batch for UI elements
        try:
            print("DEBUG: Creating shaders...")
            vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
            frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
            self.shader_program = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
            print(f"DEBUG: Shader program created. Attributes found: {self.shader_program.attributes}")

            # Use format strings matching the shader ('f' for float)
            vert_format = 'f' # Use 'f', pyglet infers count=3 from shader
            color_format = 'f' # Use 'f', pyglet infers count=3 from shader
            print(f"DEBUG: Attempting to create static VertexList with {self.total_vertices} vertices...")

            # Create ONE vertex list with ALL data, usage='static', NOT IN BATCH
            self.vertex_list = self.shader_program.vertex_list(
                self.total_vertices, # Total size for the loaded frames
                gl.GL_POINTS,
                # batch=self.batch, # REMOVED from batch
                usage='static', # Upload once
                vertices=(vert_format, final_vertices_data),
                colors=(color_format, final_colors_data)
            )
            print("DEBUG: Static VertexList created and data uploaded to GPU.")
            # Set initial draw range for the first frame
            self._update_draw_range() # Call helper to set initial range


        except pyglet.graphics.shader.ShaderException as e:
            print(f"FATAL: Shader compilation error: {e}")
            pyglet.app.exit()
            return
        except KeyError as e:
             print(f"FATAL: Error creating VertexList. Invalid format string? Key: {e}")
             pyglet.app.exit()
             return
        except Exception as e:
             print(f"FATAL: Error during pyglet setup: {e}")
             pyglet.app.exit()
             return

        # Schedule frame advance and camera updates separately
        pyglet.clock.schedule_interval(self.advance_frame, 1.0 / self.frame_rate)
        pyglet.clock.schedule_interval(self.update_camera, 1.0 / 60.0) # Update camera frequently

        # Set up OpenGL state
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    def _load_all_frames_to_ram(self):
        """Loads all GLB data into RAM lists at startup (original vertex data)."""
        print("Loading original frame data...")
        current_start_index = 0
        # Iterate only over the limited self.glb_files list
        for i, filepath in enumerate(self.glb_files):
            try:
                scene = trimesh.load(filepath, force='scene')
                pc_geom = None
                for geom in scene.geometry.values():
                    if isinstance(geom, trimesh.PointCloud):
                        pc_geom = geom
                        break

                if pc_geom is not None and len(pc_geom.vertices) > 0:
                    num_vertices = len(pc_geom.vertices)
                    original_vertices = pc_geom.vertices.astype(np.float32)
                    colors_byte = pc_geom.colors[:, :3].astype(np.uint8)

                    # Store flattened float arrays of ORIGINAL vertices
                    vertices = original_vertices.flatten()
                    colors_float = colors_byte.astype(np.float32) / 255.0
                    colors = colors_float.flatten()

                    self.all_vertices_list.append(vertices)
                    self.all_colors_list.append(colors)
                    self.frame_counts.append(num_vertices)
                    self.frame_starts.append(current_start_index)
                    self.total_vertices += num_vertices
                    current_start_index += num_vertices
                else:
                    print(f"Warning: No valid PointCloud geometry found in {filepath}")
                    self.all_vertices_list.append(np.array([], dtype=np.float32))
                    self.all_colors_list.append(np.array([], dtype=np.float32))
                    self.frame_counts.append(0)
                    self.frame_starts.append(current_start_index) # Start index remains same

            except Exception as e:
                print(f"Error loading frame {i} ({filepath}): {e}")
                # Append empty data on error
                self.all_vertices_list.append(np.array([], dtype=np.float32))
                self.all_colors_list.append(np.array([], dtype=np.float32))
                self.frame_counts.append(0)
                self.frame_starts.append(current_start_index)

        return True # Indicate success (even if some frames failed)


    def _update_draw_range(self):
        """Sets the vertex_list start and count for the current_frame_index."""
        if 0 <= self.current_frame_index < len(self.frame_starts):
            start_vertex = self.frame_starts[self.current_frame_index]
            num_vertices = self.frame_counts[self.current_frame_index]
            # print(f"DEBUG: _update_draw_range - Target Frame {self.current_frame_index}, Start: {start_vertex}, Count: {num_vertices}") # Log target
            if num_vertices > 0 and self.vertex_list:
                try:
                    # print(f"DEBUG: _update_draw_range - Before: VL Start={getattr(self.vertex_list, 'start', 'N/A')}, Count={getattr(self.vertex_list, 'count', 'N/A')}")
                    self.vertex_list.start = start_vertex
                    self.vertex_list.count = num_vertices
                    # print(f"DEBUG: _update_draw_range - After: VL Start={self.vertex_list.start}, Count={self.vertex_list.count}") # Log after setting
                except AttributeError as e:
                    print(f"Warning: Failed to set start/count on VertexList: {e}")
                    if self.vertex_list: self.vertex_list.count = 0
            elif self.vertex_list:
                 # print(f"DEBUG: _update_draw_range - Frame {self.current_frame_index} has 0 vertices. Setting count=0.")
                 self.vertex_list.count = 0 # Draw nothing for empty frames
        elif self.vertex_list:
            # print(f"DEBUG: _update_draw_range - Index {self.current_frame_index} out of bounds. Setting count=0.")
            self.vertex_list.count = 0 # Index out of bounds


    def advance_frame(self, dt):
        """Scheduled function to advance frame if playing."""
        if self.playing:
            next_index = self.current_frame_index + 1
            if next_index >= len(self.glb_files): # Use length of loaded files
                if self.loop:
                    next_index = 0
                else:
                    self.playing = False # Stop at the end
                    # print("DEBUG: Playback finished (no loop).")
                    return # Don't advance further
            # print(f"DEBUG: Advancing frame index from {self.current_frame_index} to {next_index}")
            self.current_frame_index = next_index
            self._update_draw_range() # Update draw range after index changes

    def update_camera(self, dt):
        """Scheduled function to handle camera movement based on key states."""
        speed = self.fast_move_speed if self.keys[key.LSHIFT] or self.keys[key.RSHIFT] else self.move_speed
        move_amount = speed * dt

        # Calculate forward and right vectors based on current rotation
        rot_y = -math.radians(self.camera_rotation_y) # Yaw
        rot_x = -math.radians(self.camera_rotation_x) # Pitch

        forward = Vec3(
            math.sin(rot_y) * math.cos(rot_x),
            -math.sin(rot_x),
            -math.cos(rot_y) * math.cos(rot_x)
        ).normalize()

        # Use world up for calculating right vector to prevent roll
        right = self.world_up_vector.cross(forward).normalize()
        # Use world up vector directly for vertical movement
        up = self.world_up_vector

        # Move camera position
        moved = False
        if self.keys[key.W]:
            self.camera_position += forward * move_amount
            moved = True
        if self.keys[key.S]:
            self.camera_position -= forward * move_amount
            moved = True
        # Swap A and D
        if self.keys[key.A]:
            self.camera_position += right * move_amount # Now moves right
            moved = True
        if self.keys[key.D]:
            self.camera_position -= right * move_amount # Now moves left
            moved = True
        # Use world_up_vector for Q/E movement
        if self.keys[key.E]:
            self.camera_position += up * move_amount
            moved = True
        if self.keys[key.Q]:
            self.camera_position -= up * move_amount
            moved = True
        # if moved: # Optional: print position if debugging movement
        #     print(f"DEBUG: Camera Pos: {self.camera_position}")


    def get_camera_matrices(self):
        # Calculate forward vector based on rotation
        rot_y = -math.radians(self.camera_rotation_y)
        rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(
            math.sin(rot_y) * math.cos(rot_x),
            -math.sin(rot_x),
            -math.cos(rot_y) * math.cos(rot_x)
        ).normalize()

        # Target is simply a point in front of the camera
        target = self.camera_position + forward

        # Create view and projection matrices using world up vector
        view = Mat4.look_at(self.camera_position, target, self.world_up_vector)
        projection = Mat4.perspective_projection(self.aspect_ratio, z_near=0.1, z_far=1000.0, fov=60.0) # Increased z_far
        return projection, view

    def on_draw(self):
        self.clear()
        projection, current_view = self.get_camera_matrices()

        # --- Smooth the view matrix ---
        self.view_matrix_history.append(current_view)
        avg_elements = None
        history_len = len(self.view_matrix_history)
        if history_len > 0:
            # Sum the matrices element-wise
            sum_elements = [0.0] * 16
            for mat in self.view_matrix_history:
                for i, val in enumerate(mat):
                    sum_elements[i] += val
            # Divide each element by the count
            avg_elements = tuple(val / history_len for val in sum_elements) # Use tuple
        # -----------------------------

        self.shader_program.use()
        self.shader_program['projection'] = projection
        # Use the averaged elements (tuple) or the current view if averaging failed
        self.shader_program['view'] = avg_elements if avg_elements else current_view

        # --- Draw VertexList directly, relying on its start/count ---
        if self.vertex_list and self.vertex_list.count > 0:
            # print(f"DEBUG: on_draw - Drawing Frame Index: {self.current_frame_index}, VL Start: {self.vertex_list.start}, VL Count: {self.vertex_list.count}") # Optional debug
            try:
                # Draw using internal start/count (no first/count args)
                self.vertex_list.draw(gl.GL_POINTS)
            except Exception as e:
                print(f"Error during vertex_list.draw: {e}")
        # -----------------------------------------------------------

        self.shader_program.stop()

        # Draw UI Text (Still use batch for this)
        status_text = f"Frame: {self.current_frame_index + 1}/{len(self.glb_files)}" # Use length of loaded files
        status_text += " | Playing" if self.playing else " | Paused"
        status_text += " | Loop ON" if self.loop else " | Loop OFF"
        label = pyglet.text.Label(status_text, font_name='Arial', font_size=12,
                                  x=10, y=self.height - 20, anchor_x='left', anchor_y='top',
                                  color=(255, 255, 255, 200), batch=self.batch) # Add label to batch
        self.batch.draw() # Draw the batch (which now only contains the label)


    # --- Input Handlers ---
    def on_resize(self, width, height):
        gl.glViewport(0, 0, max(1, width), max(1, height))

    def on_mouse_press(self, x, y, button, modifiers):
        # print(f"DEBUG: Mouse Press - Button: {button}") # Log press
        if button == pyglet.window.mouse.LEFT:
            # Capture mouse for rotation only when dragging
            self.set_exclusive_mouse(True)
            self.mouse_down = True
            # No need to store last_x/y when exclusive

    def on_mouse_release(self, x, y, button, modifiers):
        # print(f"DEBUG: Mouse Release - Button: {button}") # Log release
        if button == pyglet.window.mouse.LEFT:
            # Release mouse when not dragging
            self.set_exclusive_mouse(False)
            self.mouse_down = False

    # Renamed from on_mouse_drag
    def on_mouse_motion(self, x, y, dx, dy):
        # Use dx, dy directly for rotation when mouse is exclusive (and button is down)
        # print(f"DEBUG: Mouse Motion - dx: {dx}, dy: {dy}, mouse_down: {self.mouse_down}") # Log motion
        if self.mouse_down:
            sensitivity = 0.1 # Adjust sensitivity for exclusive mouse
            # Invert directions
            self.camera_rotation_y -= dx * sensitivity # Inverted Yaw
            self.camera_rotation_y %= 360
            self.camera_rotation_x += dy * sensitivity # Inverted Pitch
            self.camera_rotation_x = max(min(self.camera_rotation_x, 89.9), -89.9)
            # print(f"DEBUG: New Rotation - Yaw: {self.camera_rotation_y:.1f}, Pitch: {self.camera_rotation_x:.1f}") # Log rotation change

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # Zoom by moving camera position along forward vector
        zoom_speed = 0.5 # Adjust zoom speed
        rot_y = -math.radians(self.camera_rotation_y)
        rot_x = -math.radians(self.camera_rotation_x)
        forward = Vec3(
            math.sin(rot_y) * math.cos(rot_x),
            -math.sin(rot_x),
            -math.cos(rot_y) * math.cos(rot_x)
        ).normalize()
        self.camera_position += forward * scroll_y * zoom_speed
        # print(f"DEBUG: Zoom - ScrollY: {scroll_y}, New Pos: {self.camera_position}") # Log zoom


    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.playing = not self.playing
            # print(f"DEBUG: Playback {'started' if self.playing else 'paused'}")
        elif symbol == pyglet.window.key.L:
            self.loop = not self.loop
            # print(f"DEBUG: Loop {'ON' if self.loop else 'OFF'}")
        elif symbol == pyglet.window.key.RIGHT:
            self.playing = False # Pause on manual frame change
            next_index = self.current_frame_index + 1
            if self.loop and next_index >= len(self.glb_files): # Use length of loaded files
                next_index = 0
            if 0 <= next_index < len(self.glb_files):
                 # print(f"DEBUG: Manual frame change {self.current_frame_index} -> {next_index}")
                 self.current_frame_index = next_index
                 self._update_draw_range() # Update draw range
        elif symbol == pyglet.window.key.LEFT:
            self.playing = False # Pause on manual frame change
            next_index = self.current_frame_index - 1
            if self.loop and next_index < 0:
                next_index = len(self.glb_files) - 1 # Use length of loaded files
            if 0 <= next_index < len(self.glb_files):
                 # print(f"DEBUG: Manual frame change {self.current_frame_index} -> {next_index}")
                 self.current_frame_index = next_index
                 self._update_draw_range() # Update draw range
        elif symbol == pyglet.window.key.ESCAPE:
            # Also release mouse if captured
            self.set_exclusive_mouse(False)
            self.close()

    def on_close(self):
        print("Window closing...")
        # No loading thread
        # --- Clean up GPU resources ---
        if self.vertex_list:
            self.vertex_list.delete()
        if self.shader_program:
             self.shader_program.delete()
        # -----------------------------
        super().on_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play a sequence of GLB point cloud files as a 3D video.")
    parser.add_argument("glb_directory", help="Directory containing the sequence of .glb files.")
    parser.add_argument("-n", "--num_frames", type=int, default=-1, help="Number of frames to load upfront (default: -1 for all)") # Default to all
    # Removed --smooth argument as it's not used in this version
    args = parser.parse_args()

    if not os.path.isdir(args.glb_directory):
        print(f"Error: Directory not found: {args.glb_directory}")
    else:
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # Pass num_frames to the window
        window = PlayerWindow(args.glb_directory, num_frames_to_load=args.num_frames, width=1024, height=768, caption='3D Video Player', resizable=True)
        try:
            pyglet.app.run()
        except Exception as e:
            print("\n--- Uncaught Exception ---")
            traceback.print_exc()
            print("------------------------")
        finally:
            print("Exiting application.")