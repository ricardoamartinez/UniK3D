import pyglet
import pyglet.gl as gl
import numpy as np
import math
import traceback
import ctypes

from . import shaders # Import from the new shaders module

class Renderer:
    def __init__(self, window_width, window_height):
        self.width = window_width
        self.height = window_height
        self._aspect_ratio = float(self.width) / self.height if self.height > 0 else 1.0

        self.shader_program = None
        self.debug_shader_program = None
        self.texture_shader_program = None

        self.vertex_list = None
        self.current_point_count = 0

        self.camera_texture = None
        self.depth_texture = None
        self.edge_texture = None
        self.smoothing_texture = None
        self.wavelet_texture = None # For main screen WPT/FFT output
        self.imgui_wavelet_debug_texture = None # For ImGui WPT/FFT debug window
        self.debug_textures_initialized = False
        self._initialize_texture_attrs()

        self.texture_quad_vao = None
        self.texture_quad_vbo = None
        self._default_vao = None # For ImGui safety

        self._setup_shaders()
        self.create_debug_textures(self.width, self.height) # Initial creation
        self._setup_texture_quad()
        self._setup_default_vao()

    def _initialize_texture_attrs(self):
        self.camera_texture = None
        self.depth_texture = None
        self.edge_texture = None
        self.smoothing_texture = None
        self.wavelet_texture = None
        self.imgui_wavelet_debug_texture = None
        self.debug_textures_initialized = False
        self.camera_texture_width, self.camera_texture_height = 0, 0
        self.depth_texture_width, self.depth_texture_height = 0, 0
        self.edge_texture_width, self.edge_texture_height = 0, 0
        self.smoothing_texture_width, self.smoothing_texture_height = 0, 0
        self.wavelet_texture_width, self.wavelet_texture_height = 0, 0
        self.imgui_wavelet_debug_texture_width, self.imgui_wavelet_debug_texture_height = 0, 0

    def _setup_shaders(self):
        try:
            # Main point cloud shader
            vert_shader = pyglet.graphics.shader.Shader(shaders.VERTEX_SHADER_SOURCE, 'vertex')
            frag_shader = pyglet.graphics.shader.Shader(shaders.FRAGMENT_SHADER_SOURCE, 'fragment')
            geom_shader = pyglet.graphics.shader.Shader(shaders.GEOMETRY_SHADER_SOURCE, 'geometry')
            self.shader_program = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader, geom_shader)
            print("DEBUG: Main shader program created in Renderer.")

            # Debug line shader
            debug_vert_shader = pyglet.graphics.shader.Shader(shaders.DEBUG_VERTEX_SHADER_SOURCE, 'vertex')
            debug_frag_shader = pyglet.graphics.shader.Shader(shaders.DEBUG_FRAGMENT_SHADER_SOURCE, 'fragment')
            self.debug_shader_program = pyglet.graphics.shader.ShaderProgram(debug_vert_shader, debug_frag_shader)
            print("DEBUG: Debug shader program created in Renderer.")

            # Texture quad shader (for full-screen WPT/FFT viz)
            texture_vert_shader = pyglet.graphics.shader.Shader(shaders.TEXTURE_VERTEX_SHADER_SOURCE, 'vertex')
            texture_frag_shader = pyglet.graphics.shader.Shader(shaders.TEXTURE_FRAGMENT_SHADER_SOURCE, 'fragment')
            self.texture_shader_program = pyglet.graphics.shader.ShaderProgram(texture_vert_shader, texture_frag_shader)
            print("DEBUG: Texture quad shader program created in Renderer.")

        except pyglet.graphics.shader.ShaderException as e:
            print(f"FATAL: Shader compilation error in Renderer: {e}")
            traceback.print_exc()
            # Consider a more graceful exit or error propagation
            pyglet.app.exit()
        except Exception as e:
            print(f"FATAL: Error during shader setup in Renderer: {e}")
            traceback.print_exc()
            pyglet.app.exit()

    def _setup_texture_quad(self):
        if not self.texture_shader_program:
            print("ERROR: Texture shader program not initialized. Cannot setup texture quad.")
            return
        try:
            quad_vertices = [
                -1, -1,  0, 0,  # pos    # tex
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
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texture_quad_vbo)

            # Access attribute locations more directly
            try:
                pos_attrib_location = self.texture_shader_program.attributes['position']['location']
                tex_coord_attrib_location = self.texture_shader_program.attributes['texCoord_in']['location']
            except KeyError as e:
                print(f"FATAL: Attribute not found in texture_shader_program: {e}")
                traceback.print_exc()
                pyglet.app.exit()
                return

            gl.glEnableVertexAttribArray(pos_attrib_location)
            gl.glVertexAttribPointer(pos_attrib_location, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, ctypes.c_void_p(0))
            
            gl.glEnableVertexAttribArray(tex_coord_attrib_location)
            gl.glVertexAttribPointer(tex_coord_attrib_location, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))

            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            print("DEBUG: Texture quad VAO/VBO created in Renderer.")
        except Exception as e:
            print(f"FATAL: Error during texture quad setup in Renderer: {e}")
            traceback.print_exc()
            pyglet.app.exit()

    def _setup_default_vao(self):
        self._default_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._default_vao)
        gl.glBindVertexArray(self._default_vao) # Bind it once to make it valid
        gl.glBindVertexArray(0) # Unbind
        print("DEBUG: Default VAO created in Renderer.")

    def create_debug_textures(self, width, height):
        """Creates or re-creates textures for debug views."""
        self.delete_debug_textures() # Delete existing first

        width = max(1, width)
        height = max(1, height)

        self.camera_texture = self._create_gl_texture(width, height)
        self.camera_texture_width, self.camera_texture_height = width, height

        self.depth_texture = self._create_gl_texture(width, height)
        self.depth_texture_width, self.depth_texture_height = width, height

        self.edge_texture = self._create_gl_texture(width, height)
        self.edge_texture_width, self.edge_texture_height = width, height

        self.smoothing_texture = self._create_gl_texture(width, height)
        self.smoothing_texture_width, self.smoothing_texture_height = width, height

        self.wavelet_texture = self._create_gl_texture(width, height)
        self.wavelet_texture_width, self.wavelet_texture_height = width, height
        
        self.imgui_wavelet_debug_texture = self._create_gl_texture(width, height)
        self.imgui_wavelet_debug_texture_width, self.imgui_wavelet_debug_texture_height = width, height

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0) # Unbind
        self.debug_textures_initialized = True
        print("DEBUG: Debug textures created/recreated in Renderer.")

    def _create_gl_texture(self, width, height):
        tex_id = gl.GLuint()
        gl.glGenTextures(1, tex_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        return tex_id

    def delete_debug_textures(self):
        textures_to_delete = [
            getattr(self, 'camera_texture', None),
            getattr(self, 'depth_texture', None),
            getattr(self, 'edge_texture', None),
            getattr(self, 'smoothing_texture', None),
            getattr(self, 'wavelet_texture', None),
            getattr(self, 'imgui_wavelet_debug_texture', None)
        ]
        for tex_id in textures_to_delete:
            if tex_id:
                try: gl.glDeleteTextures(1, tex_id)
                except: pass
        self._initialize_texture_attrs()
        self.debug_textures_initialized = False

    def update_vertex_list(self, vertices_data, colors_data, num_vertices, render_settings, view_matrix):
        # print(f"DEBUG: update_vertex_list received {num_vertices} vertices.")
        self.current_point_count = num_vertices
        if vertices_data is None or colors_data is None or num_vertices <= 0:
            if self.vertex_list:
                try: self.vertex_list.delete()
                except Exception: pass
                self.vertex_list = None
            self.current_point_count = 0
            return None, None # Return None for points_for_debug if no vertices

        vertices_for_display = vertices_data
        colors_for_display = colors_data
        
        # Depth Sorting for Gaussian Splats (Render Mode 2)
        if render_settings.get("render_mode") == 2 and view_matrix:
            try:
                vertices_np = vertices_data.reshape((num_vertices, 3))
                colors_np = colors_data.reshape((num_vertices, 3))
                
                vertices_homogeneous = np.hstack((vertices_np, np.ones((num_vertices, 1))))
                view_np = np.array(view_matrix).reshape((4, 4), order='F')
                view_space_points = vertices_homogeneous @ view_np.T
                view_space_depths = view_space_points[:, 2]
                sort_indices = np.argsort(view_space_depths)
                
                vertices_sorted = vertices_np[sort_indices]
                colors_sorted = colors_np[sort_indices]
                vertices_for_display = vertices_sorted.flatten()
                colors_for_display = colors_sorted.flatten()
            except Exception as e_sort:
                print(f"Error during point sorting in Renderer: {e_sort}")
                traceback.print_exc()
                # Fallback to unsorted
                vertices_for_display = vertices_data
                colors_for_display = colors_data

        if np.isnan(vertices_for_display).any() or np.isinf(vertices_for_display).any() or \
           np.isnan(colors_for_display).any() or np.isinf(colors_for_display).any():
            print("ERROR: NaN/Inf in vertex/color data. Skipping vertex list update.")
            if self.vertex_list: self.vertex_list.delete(); self.vertex_list = None
            self.current_point_count = 0
            return None, None

        points_for_debug = None
        if vertices_for_display is not None and num_vertices > 0:
            try:
                points_for_debug = vertices_for_display.reshape((num_vertices, 3))
            except ValueError as e_reshape:
                print(f"Warning: Could not reshape vertices_for_display for debug rays in Renderer: {e_reshape}")
        
        if self.vertex_list: self.vertex_list.delete(); self.vertex_list = None
        try:
            self.vertex_list = self.shader_program.vertex_list(
                num_vertices,
                gl.GL_POINTS,
                vertices=('f', vertices_for_display),
                colors=('f', colors_for_display)
            )
        except Exception as e_create:
             print(f"ERROR creating vertex list in Renderer: {e_create}")
             traceback.print_exc()
             self.vertex_list = None
             self.current_point_count = 0
        return points_for_debug
        
    def render_scene(self, projection_matrix, view_matrix, render_settings, latest_rgb_frame_shape_for_focal):
        # Continue rendering even if no vertex list - this ensures the render loop keeps running
        if not self.shader_program:
            return  # No shader to render with

        self.shader_program.use()
        try:
            # Always set uniforms even if no vertices to draw
            self.shader_program['projection'] = projection_matrix
            self.shader_program['view'] = view_matrix
            self.shader_program['viewportSize'] = (float(self.width), float(self.height))
            self.shader_program['inputScaleFactor'] = render_settings.get("input_scale_factor", 1.0)
            self.shader_program['pointSizeBoost'] = render_settings.get("point_size_boost", 1.0)
            self.shader_program['renderMode'] = render_settings.get("render_mode", 2)
            self.shader_program['falloffFactor'] = render_settings.get("falloff_factor", 1.0)
            self.shader_program['saturation'] = render_settings.get("saturation", 1.0)
            self.shader_program['brightness'] = render_settings.get("brightness", 1.0)
            self.shader_program['contrast'] = render_settings.get("contrast", 1.0)
            self.shader_program['sharpness'] = render_settings.get("sharpness", 1.0) if render_settings.get("enable_sharpening") else 1.0

            input_fov_deg = render_settings.get("input_camera_fov", 60.0)
            fov_rad = math.radians(input_fov_deg)
            input_h = float(latest_rgb_frame_shape_for_focal[0]) if latest_rgb_frame_shape_for_focal else float(self.height)
            input_focal = (input_h * 0.5) / math.tan(fov_rad * 0.5) if math.tan(fov_rad * 0.5) > 1e-6 else 1000.0 # Avoid div by zero
            self.shader_program['inputFocal'] = input_focal
            
            self.shader_program['sizeScaleFactor'] = render_settings.get("size_scale_factor", 0.001)
            self.shader_program['minPointSize'] = render_settings.get("min_point_size", 1.0)
            self.shader_program['enableMaxSizeClamp'] = render_settings.get("enable_max_size_clamp", False)
            self.shader_program['maxPointSize'] = render_settings.get("max_point_size", 50.0)
            self.shader_program['depthExponent'] = render_settings.get("depth_exponent", 2.0)
            self.shader_program['planarProjectionActive'] = render_settings.get("planar_projection", False)

            self.shader_program['debug_show_input_distance'] = render_settings.get("debug_show_input_distance", False)
            self.shader_program['debug_show_raw_diameter'] = render_settings.get("debug_show_raw_diameter", False)
            self.shader_program['debug_show_density_factor'] = render_settings.get("debug_show_density_factor", False)
            self.shader_program['debug_show_final_size'] = render_settings.get("debug_show_final_size", False)

            current_render_mode = render_settings.get("render_mode")
            if current_render_mode == 0 or current_render_mode == 1: # Square or Circle (Opaque)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)
            elif current_render_mode == 2: # Gaussian (Premultiplied Alpha Blend)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_FALSE) # No depth write for transparency
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
            # Mode 3 (Wavelet/FFT) is handled by render_wavelet_fft_fullscreen
            
            # Only draw if we have vertices
            if self.vertex_list and self.current_point_count > 0:
                self.vertex_list.draw(gl.GL_POINTS)

        except Exception as e_render:
             print(f"ERROR during render_scene in Renderer: {e_render}")
             traceback.print_exc()
        finally:
            try: self.shader_program.stop()
            except: pass

    def render_wavelet_fft_fullscreen(self):
        if not self.texture_shader_program or not self.texture_quad_vao or not self.wavelet_texture:
            # print("DEBUG RenderWavelet: Resources MISSING for wavelet quad render in Renderer.")
            return
        try:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_BLEND)
            self.texture_shader_program.use()
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.wavelet_texture)
            self.texture_shader_program['fboTexture'] = 0
            gl.glBindVertexArray(self.texture_quad_vao)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        except Exception as e_render_quad:
            print(f"Error rendering wavelet texture quad in Renderer: {e_render_quad}")
            traceback.print_exc()
        finally:
            gl.glBindVertexArray(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            if self.texture_shader_program: self.texture_shader_program.stop()
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND) 

    def draw_debug_geometry(self, projection_matrix, view_matrix, debug_settings, latest_points_for_debug, latest_rgb_frame_shape_for_frustum):
        if not self.debug_shader_program: return

        batch = pyglet.graphics.Batch()
        debug_elements = []
        
        # Store previous GL state
        prev_line_width = gl.GLfloat(); gl.glGetFloatv(gl.GL_LINE_WIDTH, prev_line_width)
        depth_test_enabled = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        blend_enabled = gl.glIsEnabled(gl.GL_BLEND)
        depth_mask_enabled = gl.GLboolean(); gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK, depth_mask_enabled)

        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_DEPTH_TEST) 
        gl.glDepthMask(gl.GL_FALSE)
        gl.glLineWidth(2.0)

        if debug_settings.get("debug_show_world_axes", True):
            axis_length = 1.0
            v = [0,0,0, axis_length,0,0,  0,0,0, 0,axis_length,0,  0,0,0, 0,0,axis_length]
            c = [1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1]
            debug_elements.append(self.debug_shader_program.vertex_list(6, gl.GL_LINES, batch=batch, position=('f',v), color=('f',c)))

        if debug_settings.get("debug_show_input_frustum", False):
            try:
                fov = math.radians(debug_settings.get("input_camera_fov", 60.0))
                aspect = 1.0
                if latest_rgb_frame_shape_for_frustum and latest_rgb_frame_shape_for_frustum[0] > 0:
                    aspect = latest_rgb_frame_shape_for_frustum[1] / latest_rgb_frame_shape_for_frustum[0]
                near, far = 0.1, 20.0
                hn, wn = 2*near*math.tan(fov/2), 2*near*math.tan(fov/2)*aspect
                hf, wf = 2*far*math.tan(fov/2), 2*far*math.tan(fov/2)*aspect
                corners_model = [
                    ( wn/2,  hn/2, -near), (-wn/2,  hn/2, -near),
                    (-wn/2, -hn/2, -near), ( wn/2, -hn/2, -near),
                    ( wf/2,  hf/2, -far),  (-wf/2,  hf/2, -far),
                    (-wf/2, -hf/2, -far),  ( wf/2, -hf/2, -far) ]
                corners_render = [(p[0], -p[1], -p[2]) for p in corners_model] # Y-down to Y-up
                indices = [0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 0,4,1,5,2,6,3,7]
                v = [coord for i in indices for coord in corners_render[i]]
                c = [1,1,0] * len(indices)
                debug_elements.append(self.debug_shader_program.vertex_list(len(indices), gl.GL_LINES, batch=batch, position=('f',v), color=('f',c)))
            except Exception as e: print(f"Error drawing input frustum: {e}")

        # ... (Viewer Frustum and Input Rays would be similar, using debug_settings and latest_points_for_debug)

        if debug_elements:
            self.debug_shader_program.use()
            self.debug_shader_program['projection'] = projection_matrix
            self.debug_shader_program['view'] = view_matrix
            batch.draw()
            self.debug_shader_program.stop()

        # Restore GL state
        gl.glLineWidth(prev_line_width.value)
        if depth_test_enabled: gl.glEnable(gl.GL_DEPTH_TEST)
        else: gl.glDisable(gl.GL_DEPTH_TEST)
        if blend_enabled: gl.glEnable(gl.GL_BLEND)
        else: gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(depth_mask_enabled.value)

    def on_resize(self, width, height):
        self.width = max(1, width)
        self.height = max(1, height)
        gl.glViewport(0, 0, self.width, self.height)
        self._aspect_ratio = float(self.width) / self.height if self.height > 0 else 1.0
        self.create_debug_textures(self.width, self.height) # Recreate on resize

    def cleanup(self):
        print("Cleaning up Renderer resources...")
        if self.vertex_list: self.vertex_list.delete(); self.vertex_list = None
        if self.shader_program: self.shader_program.delete(); self.shader_program = None
        if self.debug_shader_program: self.debug_shader_program.delete(); self.debug_shader_program = None
        if self.texture_shader_program: self.texture_shader_program.delete(); self.texture_shader_program = None
        if self.texture_quad_vao: gl.glDeleteVertexArrays(1, self.texture_quad_vao); self.texture_quad_vao = None
        if self.texture_quad_vbo: gl.glDeleteBuffers(1, self.texture_quad_vbo); self.texture_quad_vbo = None
        if self._default_vao: gl.glDeleteVertexArrays(1, self._default_vao); self._default_vao = None
        self.delete_debug_textures()
        print("Renderer cleanup complete.") 

    def _upload_texture_data(self, tex_id, current_w_attr_name, current_h_attr_name, data_np, gl_internal_format=gl.GL_RGB8, gl_format=gl.GL_RGB, data_type=gl.GL_UNSIGNED_BYTE):
        current_w = getattr(self, current_w_attr_name)
        current_h = getattr(self, current_h_attr_name)
        if data_np is None or tex_id is None: 
            # If data is None, potentially clear the texture or leave as is based on requirements.
            # For now, if data is None, we do nothing to the texture content.
            return current_w, current_h
        
        if not isinstance(data_np, np.ndarray):
            print(f"Warning: _upload_texture_data received non-numpy data for tex_id {tex_id}")
            return current_w, current_h

        if data_np.ndim < 2 or data_np.size == 0:
            # print(f"Warning: _upload_texture_data received empty or invalid shape data for tex_id {tex_id}. Shape: {data_np.shape}")
            # Optionally clear texture here if this is an error state
            return current_w, current_h

        h_d, w_d = data_np.shape[:2]
        if h_d == 0 or w_d == 0: return current_w, current_h # Invalid dimensions

        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        if w_d != current_w or h_d != current_h:
            # print(f"DEBUG Renderer: Reallocating texture {tex_id} from ({current_w}x{current_h}) to ({w_d}x{h_d})")
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl_internal_format, w_d, h_d, 0, gl_format, data_type, data_np.ctypes.data)
            setattr(self, current_w_attr_name, w_d)
            setattr(self, current_h_attr_name, h_d)
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w_d, h_d, gl_format, data_type, data_np.ctypes.data)
        return getattr(self, current_w_attr_name), getattr(self, current_h_attr_name)

    def update_all_debug_textures(self, latest_rgb_frame, latest_depth_map_viz, latest_edge_map, 
                                  latest_smoothing_map, latest_main_screen_coeff_viz_content, 
                                  latest_wavelet_map):
        if not self.debug_textures_initialized: return
        try:
            # Camera Feed (RGB)
            if self.camera_texture and latest_rgb_frame is not None:
                self.camera_texture_width, self.camera_texture_height = self._upload_texture_data(
                    self.camera_texture, "camera_texture_width", "camera_texture_height", 
                    np.ascontiguousarray(latest_rgb_frame) if latest_rgb_frame is not None else None, gl_format=gl.GL_RGB)

            # Depth Map (BGR from cv2.cvtColor)
            if self.depth_texture and latest_depth_map_viz is not None:
                self.depth_texture_width, self.depth_texture_height = self._upload_texture_data(
                    self.depth_texture, "depth_texture_width", "depth_texture_height", 
                    np.ascontiguousarray(latest_depth_map_viz) if latest_depth_map_viz is not None else None, gl_format=gl.GL_BGR)
            
            # Edge Map (RGB from cv2.cvtColor(COLOR_GRAY2RGB) or similar)
            if self.edge_texture and latest_edge_map is not None:
                self.edge_texture_width, self.edge_texture_height = self._upload_texture_data(
                    self.edge_texture, "edge_texture_width", "edge_texture_height", 
                    np.ascontiguousarray(latest_edge_map) if latest_edge_map is not None else None, gl_format=gl.GL_RGB) # Assuming it became RGB

            # Smoothing Map (Should be RGB for visualization)
            if self.smoothing_texture and latest_smoothing_map is not None:
                self.smoothing_texture_width, self.smoothing_texture_height = self._upload_texture_data(
                    self.smoothing_texture, "smoothing_texture_width", "smoothing_texture_height", 
                    np.ascontiguousarray(latest_smoothing_map) if latest_smoothing_map is not None else None, gl_format=gl.GL_RGB) # Assuming it became RGB

            # Main Screen Wavelet/FFT content (BGR from inference_logic)
            if self.wavelet_texture and latest_main_screen_coeff_viz_content is not None:
                self.wavelet_texture_width, self.wavelet_texture_height = self._upload_texture_data(
                    self.wavelet_texture, "wavelet_texture_width", "wavelet_texture_height", 
                    np.ascontiguousarray(latest_main_screen_coeff_viz_content) if latest_main_screen_coeff_viz_content is not None else None, gl_format=gl.GL_BGR)
            elif self.wavelet_texture: # Clear if no content
                clear_data = np.full((max(1,self.wavelet_texture_height), max(1,self.wavelet_texture_width), 3), [10,0,10], dtype=np.uint8)
                self._upload_texture_data(self.wavelet_texture, "wavelet_texture_width", "wavelet_texture_height", clear_data, gl_format=gl.GL_RGB)

            # ImGui Wavelet Debug Texture (BGR from main_viewer processing)
            if self.imgui_wavelet_debug_texture and latest_wavelet_map is not None:
                self.imgui_wavelet_debug_texture_width, self.imgui_wavelet_debug_texture_height = self._upload_texture_data(
                    self.imgui_wavelet_debug_texture, "imgui_wavelet_debug_texture_width", "imgui_wavelet_debug_texture_height", 
                    np.ascontiguousarray(latest_wavelet_map) if latest_wavelet_map is not None else None, gl_format=gl.GL_BGR)
            elif self.imgui_wavelet_debug_texture:
                clear_data = np.full((max(1,self.imgui_wavelet_debug_texture_height), max(1,self.imgui_wavelet_debug_texture_width), 3), [0,0,20], dtype=np.uint8)
                self._upload_texture_data(self.imgui_wavelet_debug_texture, "imgui_wavelet_debug_texture_width", "imgui_wavelet_debug_texture_height", clear_data, gl_format=gl.GL_RGB)

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        except Exception as e_tex_upload:
            print(f"Error in Renderer.update_all_debug_textures: {e_tex_upload}"); traceback.print_exc() 