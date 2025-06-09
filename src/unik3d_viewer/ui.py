import imgui
from imgui.integrations.pyglet import create_renderer
import pyglet.gl as gl
import os
import mss # For screen capture monitor listing
import tkinter as tk
from tkinter import filedialog
from collections import deque  # Added for wavelet settings

from .gl_utils import hex_to_imvec4 
from .config import DEFAULT_SETTINGS

class UIManager:
    def __init__(self, window_pyglet, app_window_ref):
        self.window_pyglet = window_pyglet
        self.app = app_window_ref # Reference to the LiveViewerWindow instance
        self.imgui_renderer = None
        self._default_vao_ui = None
        self._setup_imgui()
        self._setup_default_vao_for_ui()

    def _setup_imgui(self):
        try:
            imgui.create_context()
            self.imgui_renderer = create_renderer(self.window_pyglet)
            print("DEBUG: ImGui initialized in UIManager.") 
        except Exception as e_imgui:
            print(f"FATAL: Error initializing ImGui in UIManager: {e_imgui}"); pyglet.app.exit(); raise

    def _setup_default_vao_for_ui(self):
        self._default_vao_ui = gl.GLuint(); gl.glGenVertexArrays(1, self._default_vao_ui)
        gl.glBindVertexArray(self._default_vao_ui); gl.glBindVertexArray(0)
        print("DEBUG: Default VAO for UI created.")

    def define_and_render_imgui(self):
        if not self.imgui_renderer: return

        # Style copied from original live_slam_viewer.py
        style = imgui.get_style()
        style.window_rounding = 4.0; style.frame_rounding = 4.0; style.grab_rounding = 4.0
        style.window_padding = (8, 8); style.frame_padding = (6, 4)
        style.item_spacing = (8, 4); style.item_inner_spacing = (4, 4)
        style.window_border_size = 1.0; style.frame_border_size = 0.0
        style.popup_rounding = 4.0; style.popup_border_size = 1.0
        style.window_title_align = (0.5, 0.5)
        
        style.colors[imgui.COLOR_TEXT]                  = hex_to_imvec4("#E0E0E0")
        style.colors[imgui.COLOR_TEXT_DISABLED]         = hex_to_imvec4("#666666")
        style.colors[imgui.COLOR_WINDOW_BACKGROUND]     = hex_to_imvec4("#050505", alpha=0.85)
        style.colors[imgui.COLOR_CHILD_BACKGROUND]      = hex_to_imvec4("#0A0A0A", alpha=0.85)
        style.colors[imgui.COLOR_POPUP_BACKGROUND]      = hex_to_imvec4("#030303", alpha=0.90)
        style.colors[imgui.COLOR_BORDER]                = hex_to_imvec4("#444444")
        style.colors[imgui.COLOR_BORDER_SHADOW]         = hex_to_imvec4("#000000", alpha=0.0)
        style.colors[imgui.COLOR_FRAME_BACKGROUND]      = hex_to_imvec4("#101010", alpha=0.80)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED]= hex_to_imvec4("#181818", alpha=0.85)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE]= hex_to_imvec4("#202020", alpha=0.90)
        style.colors[imgui.COLOR_TITLE_BACKGROUND]      = hex_to_imvec4("#050505", alpha=0.85)
        style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE]= hex_to_imvec4("#151515", alpha=0.90)
        style.colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED]= hex_to_imvec4("#050505", alpha=0.75)
        style.colors[imgui.COLOR_MENUBAR_BACKGROUND]    = hex_to_imvec4("#050505")
        style.colors[imgui.COLOR_SCROLLBAR_BACKGROUND]  = hex_to_imvec4("#000000")
        style.colors[imgui.COLOR_SCROLLBAR_GRAB]        = hex_to_imvec4("#444444")
        style.colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED]= hex_to_imvec4("#666666")
        style.colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = hex_to_imvec4("#888888")
        style.colors[imgui.COLOR_CHECK_MARK]            = hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_SLIDER_GRAB]           = hex_to_imvec4("#BBBBBB")
        style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE]    = hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_BUTTON]                = hex_to_imvec4("#252525")
        style.colors[imgui.COLOR_BUTTON_HOVERED]        = hex_to_imvec4("#353535")
        style.colors[imgui.COLOR_BUTTON_ACTIVE]         = hex_to_imvec4("#454545")
        style.colors[imgui.COLOR_HEADER]                = hex_to_imvec4("#181818")
        style.colors[imgui.COLOR_HEADER_HOVERED]        = hex_to_imvec4("#282828")
        style.colors[imgui.COLOR_HEADER_ACTIVE]         = hex_to_imvec4("#383838")
        style.colors[imgui.COLOR_SEPARATOR]             = hex_to_imvec4("#333333")
        style.colors[imgui.COLOR_SEPARATOR_HOVERED]     = hex_to_imvec4("#555555")
        style.colors[imgui.COLOR_SEPARATOR_ACTIVE]      = hex_to_imvec4("#777777")
        style.colors[imgui.COLOR_RESIZE_GRIP]           = hex_to_imvec4("#444444")
        style.colors[imgui.COLOR_RESIZE_GRIP_HOVERED]   = hex_to_imvec4("#666666")
        style.colors[imgui.COLOR_RESIZE_GRIP_ACTIVE]    = hex_to_imvec4("#888888")
        style.colors[imgui.COLOR_TAB]                   = hex_to_imvec4("#0A0A0A")
        style.colors[imgui.COLOR_TAB_HOVERED]           = hex_to_imvec4("#1A1A1A")
        style.colors[imgui.COLOR_TAB_ACTIVE]            = hex_to_imvec4("#2A2A2A")
        style.colors[imgui.COLOR_TAB_UNFOCUSED]         = hex_to_imvec4("#050505")
        style.colors[imgui.COLOR_TAB_UNFOCUSED_ACTIVE]  = hex_to_imvec4("#151515")
        style.colors[imgui.COLOR_PLOT_LINES]            = hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_PLOT_LINES_HOVERED]    = hex_to_imvec4("#DDDDDD")
        style.colors[imgui.COLOR_PLOT_HISTOGRAM]        = hex_to_imvec4("#BBBBBB")
        style.colors[imgui.COLOR_PLOT_HISTOGRAM_HOVERED]= hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_TABLE_HEADER_BACKGROUND] = hex_to_imvec4("#181818")
        style.colors[imgui.COLOR_TABLE_BORDER_STRONG]   = hex_to_imvec4("#333333")
        style.colors[imgui.COLOR_TABLE_BORDER_LIGHT]    = hex_to_imvec4("#222222")
        style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND]      = hex_to_imvec4("#0A0A0A")
        style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND_ALT]  = hex_to_imvec4("#101010")
        style.colors[imgui.COLOR_TEXT_SELECTED_BACKGROUND]      = hex_to_imvec4("#444444")
        style.colors[imgui.COLOR_DRAG_DROP_TARGET]      = hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_NAV_HIGHLIGHT]         = hex_to_imvec4("#FFFFFF")
        style.colors[imgui.COLOR_NAV_WINDOWING_HIGHLIGHT]= hex_to_imvec4("#888888")
        style.colors[imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND]  = hex_to_imvec4("#000000", alpha=0.2)
        style.colors[imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND]   = hex_to_imvec4("#000000", alpha=0.5)

        imgui.new_frame()

        if getattr(self.app, 'show_screen_share_popup', False):
            imgui.open_popup("Select Screen Monitor")
        main_viewport = imgui.get_main_viewport()
        popup_pos = (main_viewport.work_pos[0] + main_viewport.work_size[0] * 0.5, 
                     main_viewport.work_pos[1] + main_viewport.work_size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], imgui.ALWAYS, 0.5, 0.5)
        imgui.set_next_window_size(400, 0)
        if imgui.begin_popup_modal("Select Screen Monitor", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Select Monitor to Capture:")
            imgui.separator()
            monitors = []
            try:
                with mss.mss() as sct: monitors = sct.monitors
            except Exception as e_mss: imgui.text_colored(f"Error: {e_mss}", 1,0,0,1)
            
            app_temp_monitor_index = getattr(self.app, 'temp_monitor_index', 0)
            if app_temp_monitor_index >= len(monitors): app_temp_monitor_index = 0
            
            # Corrected radio button logic: Use unique labels or manage state carefully if labels are identical.
            # For radio_button, the value argument is what this button represents.
            # It returns True if clicked, and you then set your state to this button's value.
            
            if imgui.radio_button("Entire Desktop (All Monitors)", app_temp_monitor_index == 0):
                setattr(self.app, 'temp_monitor_index', 0)
            
            for i in range(1, len(monitors)):
                 mon = monitors[i]
                 label = f"Monitor {i} ({mon['width']}x{mon['height']} at {mon['left']},{mon['top']})##Monitor{i}" # Unique ID for radio
                 if imgui.radio_button(label, app_temp_monitor_index == i):
                     setattr(self.app, 'temp_monitor_index', i)
            
            imgui.separator()
            if imgui.button("Start Sharing", width=120):
                current_temp_idx = getattr(self.app, 'temp_monitor_index', 0)
                setattr(self.app, 'screen_capture_monitor_index', current_temp_idx)
                setattr(self.app, 'show_screen_share_popup', False)
                if hasattr(self.app, '_switch_input_source') and callable(self.app._switch_input_source):
                    self.app._switch_input_source("Screen", None, getattr(self.app, 'screen_capture_monitor_index',0))
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120): setattr(self.app, 'show_screen_share_popup', False); imgui.close_current_popup()
            imgui.end_popup()

        imgui.set_next_window_position(10, 10, imgui.ONCE)
        app_height = getattr(self.app, 'height', 768) 
        imgui.set_next_window_size(350, app_height - 20, imgui.ONCE)
        imgui.begin("UniK3D Controls")

        status_text = "Status: Idle"; color = (0.5,0.5,0.5,1.0)
        thread_is_running = self.app.inference_thread and self.app.inference_thread.is_alive()
        current_status_message = getattr(self.app, 'status_message', "")
        app_input_mode = getattr(self.app, 'input_mode', DEFAULT_SETTINGS["input_mode"])
        app_input_filepath = getattr(self.app, 'input_filepath', DEFAULT_SETTINGS["input_filepath"])
        app_screen_idx = getattr(self.app, 'screen_capture_monitor_index', DEFAULT_SETTINGS["screen_capture_monitor_index"])

        if thread_is_running:
            if app_input_mode == "Live": status_text = "Status: Live Feed Active"; color = (0.1,1.0,0.1,1.0)
            elif app_input_mode == "File": status_text = f"Status: Proc. File ({os.path.basename(app_input_filepath or 'N/A')})"; color=(0.1,0.6,1.0,1.0)
            elif app_input_mode == "GLB Sequence": status_text = f"Status: GLB Seq. ({os.path.basename(app_input_filepath or 'N/A')})"; color=(1.0,0.6,0.1,1.0)
            elif app_input_mode == "Screen": status_text = f"Status: Screen Share (M{app_screen_idx})"; color=(0.1,1.0,0.6,1.0)
            else: status_text = f"Status: {current_status_message}"
        elif "Error" in current_status_message : color = (1.0,0.1,0.1,1.0); status_text = f"Status: {current_status_message}"
        else: status_text = f"Status: {current_status_message}" 
        imgui.text_colored(status_text, *color); imgui.separator()

        if imgui.begin_tab_bar("MainTabs"):
            if imgui.begin_tab_item("Input/Output")[0]:
                imgui.text("Select Input Source:")
                options = ["Live", "File", "GLB Sequence", "Screen"]
                labels = ["Live Camera", "Media File", "GLB Sequence", "Screen Share"]
                current_idx = options.index(app_input_mode) if app_input_mode in options else 0
                changed, idx = imgui.combo("Input Source", current_idx, labels)
                if changed and idx != current_idx:
                    mode = options[idx]
                    if mode == "Live": self.app._switch_input_source("Live", None)
                    elif mode == "File": self.app._browse_media_file()
                    elif mode == "GLB Sequence": self.app._browse_glb_sequence()
                    elif mode == "Screen": setattr(self.app, 'show_screen_share_popup', True)
                
                if app_input_mode in ["File", "GLB Sequence"]: imgui.text(f"Path: {app_input_filepath or 'None'}")
                elif app_input_mode == "Screen": imgui.text(f"Capturing: Monitor {app_screen_idx if app_screen_idx > 0 else 'All'}")
                
                # Live Processing Mode (Real-time vs Buffered)
                if app_input_mode == "Live":
                    imgui.separator()
                    imgui.text("Live Processing Mode")
                    app_live_mode = getattr(self.app, 'live_processing_mode', DEFAULT_SETTINGS["live_processing_mode"])
                    mode_options = ["Real-time", "Buffered"]
                    current_mode_idx = mode_options.index(app_live_mode) if app_live_mode in mode_options else 0
                    mode_changed, new_mode_idx = imgui.combo("Processing Mode", current_mode_idx, mode_options)
                    if mode_changed:
                        setattr(self.app, 'live_processing_mode', mode_options[new_mode_idx])
                        self.app._update_edge_params()
                    imgui.text("Real-time: Drop frames for low latency")
                    imgui.text("Buffered: Process all frames")
                
                imgui.separator()
                imgui.text("Recording (GLB Sequence)")
                app_is_recording = getattr(self.app, 'is_recording', DEFAULT_SETTINGS["is_recording"])
                rec_changed, new_is_recording = imgui.checkbox("Record Frames", app_is_recording)
                if rec_changed: setattr(self.app, 'is_recording', new_is_recording); self.app.update_recording_state()
                imgui.same_line(); imgui.text(f"Saved: {self.app.recording_state.get('frames_saved',0)}")
                
                app_rec_dir = getattr(self.app, 'recording_output_dir', DEFAULT_SETTINGS["recording_output_dir"])
                dir_changed, new_rec_dir = imgui.input_text("Output Dir", app_rec_dir, 256)
                if dir_changed: setattr(self.app, 'recording_output_dir', new_rec_dir); self.app.update_recording_state()
                if imgui.button("Browse##RecDir"):
                    root = tk.Tk(); root.withdraw()
                    sel_dir = filedialog.askdirectory(title="Select Recording Output Directory", initialdir=os.path.abspath(getattr(self.app, 'recording_output_dir', '.')))
                    if sel_dir: setattr(self.app, 'recording_output_dir', sel_dir); self.app.update_recording_state()
                    root.destroy()
                imgui.end_tab_item()

            show_playback = app_input_mode != "Live" and self.app.playback_state.get("total_frames",0) > 0
            if show_playback and imgui.begin_tab_item("Playback")[0]:
                app_is_playing = getattr(self.app, 'is_playing', True)
                btn_txt = " Pause " if app_is_playing else " Play  "
                if imgui.button(btn_txt): setattr(self.app, 'is_playing', not app_is_playing); self.app.update_playback_state()
                imgui.same_line()
                if imgui.button("Restart"): self.app.playback_state["restart"]=True; setattr(self.app, 'is_playing', True); self.app.update_playback_state()
                imgui.same_line()
                app_loop_video = getattr(self.app, 'loop_video', DEFAULT_SETTINGS["loop_video"])
                loop_ch, new_loop_video = imgui.checkbox("Loop", app_loop_video)
                if loop_ch: setattr(self.app, 'loop_video', new_loop_video); self.app.update_playback_state()
                
                app_playback_speed = getattr(self.app, 'playback_speed', DEFAULT_SETTINGS["playback_speed"])
                speed_ch, new_playback_speed = imgui.slider_float("Speed", app_playback_speed,0.1,4.0,"%.1fx")
                if speed_ch: setattr(self.app, 'playback_speed', new_playback_speed); self.app.update_playback_state()
                cur_f=self.app.playback_state.get("current_frame",0); total_f=self.app.playback_state.get("total_frames",0)
                imgui.text(f"Frame: {cur_f} / {total_f}")
                prog=float(cur_f)/total_f if total_f>0 else 0.0; imgui.progress_bar(prog,(-1,0),f"{cur_f}/{total_f}")
                imgui.end_tab_item()

            if imgui.begin_tab_item("Processing")[0]:
                imgui.text("Temporal Smoothing")
                c,v=imgui.checkbox("Enable##PSE",getattr(self.app,"enable_point_smoothing",DEFAULT_SETTINGS["enable_point_smoothing"])); 
                if c: setattr(self.app, 'enable_point_smoothing', v); self.app._update_edge_params()
                if getattr(self.app,"enable_point_smoothing",DEFAULT_SETTINGS["enable_point_smoothing"]):
                    imgui.indent()
                    c,v=imgui.slider_float("Min Alpha",getattr(self.app,"min_alpha_points",DEFAULT_SETTINGS["min_alpha_points"]),0,1); 
                    if c: setattr(self.app, 'min_alpha_points',v); self.app._update_edge_params()
                    c,v=imgui.slider_float("Max Alpha",getattr(self.app,"max_alpha_points",DEFAULT_SETTINGS["max_alpha_points"]),0,1); 
                    if c: setattr(self.app, 'max_alpha_points',v); self.app._update_edge_params()
                    if imgui.button("Reset##PSR"): setattr(self.app,'min_alpha_points',DEFAULT_SETTINGS["min_alpha_points"]); setattr(self.app,'max_alpha_points',DEFAULT_SETTINGS["max_alpha_points"]); self.app._update_edge_params()
                    imgui.unindent()
                imgui.separator()
                
                imgui.text("Edge-Aware Smoothing")
                c,v=imgui.checkbox("Enable##EASE",getattr(self.app,"enable_edge_aware_smoothing",DEFAULT_SETTINGS["enable_edge_aware_smoothing"])); 
                if c: setattr(self.app,'enable_edge_aware_smoothing',v); self.app._update_edge_params()
                if getattr(self.app,"enable_edge_aware_smoothing",DEFAULT_SETTINGS["enable_edge_aware_smoothing"]):
                    imgui.indent()
                    imgui.columns(2,"edge_thresh_ui",border=False)
                    c,v=imgui.slider_float("DTh1",getattr(self.app,"depth_edge_threshold1",DEFAULT_SETTINGS["depth_edge_threshold1"]),1,255); 
                    if c: setattr(self.app,'depth_edge_threshold1',v); self.app._update_edge_params()
                    c,v=imgui.slider_float("DTh2",getattr(self.app,"depth_edge_threshold2",DEFAULT_SETTINGS["depth_edge_threshold2"]),1,255); 
                    if c: setattr(self.app,'depth_edge_threshold2',v); self.app._update_edge_params()
                    imgui.next_column()
                    c,v=imgui.slider_float("RTh1",getattr(self.app,"rgb_edge_threshold1",DEFAULT_SETTINGS["rgb_edge_threshold1"]),1,255); 
                    if c: setattr(self.app,'rgb_edge_threshold1',v); self.app._update_edge_params()
                    c,v=imgui.slider_float("RTh2",getattr(self.app,"rgb_edge_threshold2",DEFAULT_SETTINGS["rgb_edge_threshold2"]),1,255); 
                    if c: setattr(self.app,'rgb_edge_threshold2',v); self.app._update_edge_params()
                    imgui.columns(1)
                    c,v=imgui.slider_float("EdgeInf",getattr(self.app,"edge_smoothing_influence",DEFAULT_SETTINGS["edge_smoothing_influence"]),0,1); 
                    if c: setattr(self.app,'edge_smoothing_influence',v); self.app._update_edge_params()
                    c,v=imgui.slider_float("GradScale",getattr(self.app,"gradient_influence_scale",DEFAULT_SETTINGS["gradient_influence_scale"]),0,5); 
                    if c: setattr(self.app,'gradient_influence_scale',v); self.app._update_edge_params()
                    if imgui.button("Reset##EASR"): [setattr(self.app,k,DEFAULT_SETTINGS[k]) for k in ["depth_edge_threshold1","depth_edge_threshold2","rgb_edge_threshold1","rgb_edge_threshold2","edge_smoothing_influence","gradient_influence_scale"]]; self.app._update_edge_params()
                    imgui.unindent()
                imgui.separator()

                imgui.text("Image Sharpening")
                c,v=imgui.checkbox("Enable##SharpE",getattr(self.app,"enable_sharpening",DEFAULT_SETTINGS["enable_sharpening"])); 
                if c: setattr(self.app,'enable_sharpening',v); self.app._update_edge_params()
                if getattr(self.app,"enable_sharpening",DEFAULT_SETTINGS["enable_sharpening"]):
                    imgui.indent()
                    c,v=imgui.slider_float("Amount",getattr(self.app,"sharpness",DEFAULT_SETTINGS["sharpness"]),0.1,5.0); 
                    if c: setattr(self.app,'sharpness',v); self.app._update_edge_params()
                    if imgui.button("Reset##SharpR"): setattr(self.app,'sharpness',DEFAULT_SETTINGS["sharpness"]); self.app._update_edge_params()
                    imgui.unindent()
                imgui.separator()

                imgui.text("Point Thickening")
                c,v=imgui.checkbox("Enable##PTEnable",getattr(self.app,"enable_point_thickening",DEFAULT_SETTINGS["enable_point_thickening"])); 
                if c: setattr(self.app,'enable_point_thickening',v); self.app._update_edge_params()
                if getattr(self.app,"enable_point_thickening",DEFAULT_SETTINGS["enable_point_thickening"]):
                    imgui.indent()
                    dup_val = int(getattr(self.app,"thickening_duplicates",DEFAULT_SETTINGS["thickening_duplicates"]))
                    c,v_int=imgui.slider_int("Duplicates", dup_val,0,10); 
                    if c: setattr(self.app,'thickening_duplicates',v_int); self.app._update_edge_params()
                    c,v=imgui.slider_float("Variance",getattr(self.app,"thickening_variance",DEFAULT_SETTINGS["thickening_variance"]),0,0.1,"%.4f"); 
                    if c: setattr(self.app,'thickening_variance',v); self.app._update_edge_params()
                    c,v=imgui.slider_float("Depth Bias",getattr(self.app,"thickening_depth_bias",DEFAULT_SETTINGS["thickening_depth_bias"]),0,0.5,"%.3f"); 
                    if c: setattr(self.app,'thickening_depth_bias',v); self.app._update_edge_params()
                    if imgui.button("Reset##PTR"): [setattr(self.app,k,DEFAULT_SETTINGS[k]) for k in ["thickening_duplicates","thickening_variance","thickening_depth_bias"]]; self.app._update_edge_params()
                    imgui.unindent()
                imgui.separator()
                
                imgui.text("Ray Generation (Input Cam)")
                c,v=imgui.checkbox("Planar Rays",getattr(self.app,"planar_projection",DEFAULT_SETTINGS["planar_projection"])); 
                if c: setattr(self.app,'planar_projection',v); self.app._update_edge_params()
                if getattr(self.app,"planar_projection",DEFAULT_SETTINGS["planar_projection"]):
                    imgui.indent()
                    c,v=imgui.slider_float("FOV (Y)",getattr(self.app,"input_camera_fov",DEFAULT_SETTINGS["input_camera_fov"]),10,120,"%.1f deg"); 
                    if c: setattr(self.app,'input_camera_fov',v); self.app._update_edge_params()
                    if imgui.button("Reset##ICFovR"): setattr(self.app,'input_camera_fov',DEFAULT_SETTINGS["input_camera_fov"]);self.app._update_edge_params()
                    imgui.unindent()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Rendering")[0]:
                imgui.text("Point Style")
                app_render_mode = getattr(self.app, "render_mode", DEFAULT_SETTINGS["render_mode"])
                if imgui.radio_button("Square##RM",app_render_mode==0): setattr(self.app,"render_mode",0); self.app._update_edge_params()
                imgui.same_line(); 
                if imgui.radio_button("Circle##RM",app_render_mode==1): setattr(self.app,"render_mode",1); self.app._update_edge_params()
                imgui.same_line(); 
                if imgui.radio_button("Gaussian##RM",app_render_mode==2): setattr(self.app,"render_mode",2); self.app._update_edge_params()
                imgui.same_line(); 
                if imgui.radio_button("Wavelet/FFT##RM",app_render_mode==3): setattr(self.app,"render_mode",3); self.app._update_edge_params()
                
                if getattr(self.app,"render_mode",DEFAULT_SETTINGS["render_mode"]) == 2:
                    imgui.indent()
                    c,v=imgui.slider_float("Falloff",getattr(self.app,"falloff_factor",DEFAULT_SETTINGS["falloff_factor"]),0.1,50);
                    # For shader-only uniforms like falloff_factor, direct setattr is fine, no _update_edge_params needed if not passed to thread
                    if c: setattr(self.app, 'falloff_factor', v) 
                    if imgui.button("Reset##FalloffR"):setattr(self.app,'falloff_factor',DEFAULT_SETTINGS["falloff_factor"])
                    imgui.unindent()
                if getattr(self.app,"render_mode",DEFAULT_SETTINGS["render_mode"]) == 3:
                    imgui.indent()
                    c,v=imgui.slider_int("Wavelet Win",int(getattr(self.app,"wavelet_packet_window_size",DEFAULT_SETTINGS["wavelet_packet_window_size"])),16,256);
                    if c: setattr(self.app,'wavelet_packet_window_size',v); self.app.depth_history.clear();setattr(self.app,'latest_wavelet_map',None);self.app._update_edge_params()
                    
                    wpt_str = str(getattr(self.app,"wavelet_packet_type",DEFAULT_SETTINGS["wavelet_packet_type"]))
                    c,v_str=imgui.input_text("Wavelet Type", wpt_str, 64);
                    if c: setattr(self.app,'wavelet_packet_type',v_str); self.app.depth_history.clear();setattr(self.app,'latest_wavelet_map',None);self.app._update_edge_params()
                    
                    c,v=imgui.slider_int("FFT Size", int(getattr(self.app, "fft_size", DEFAULT_SETTINGS["fft_size"])), 128, 2048);
                    if c: setattr(self.app,'fft_size',v); self.app.depth_history.clear(); setattr(self.app,'latest_wavelet_map',None); self.app._update_edge_params()
                    c,v=imgui.slider_int("DMD Time Window", int(getattr(self.app, "dmd_time_window", DEFAULT_SETTINGS["dmd_time_window"])), 1, 100);
                    if c: setattr(self.app,'dmd_time_window',v); self.app.depth_history = deque(maxlen=v); setattr(self.app,'latest_wavelet_map',None); self.app._update_edge_params()
                    c,v=imgui.checkbox("CUDA Transform", getattr(self.app, "enable_cuda_transform", DEFAULT_SETTINGS["enable_cuda_transform"]));
                    if c: setattr(self.app,'enable_cuda_transform',v); self.app._update_edge_params()
                    imgui.unindent()
                imgui.separator()
                imgui.text("Point Sizing (Input Cam Rel.)")
                c,v=imgui.slider_float("Base Size Boost",getattr(self.app,"point_size_boost",DEFAULT_SETTINGS["point_size_boost"]),0.1,50);
                if c: setattr(self.app,'point_size_boost',v); self.app._update_edge_params()
                if imgui.button("Reset##PSBR"): setattr(self.app,'point_size_boost',DEFAULT_SETTINGS["point_size_boost"]); self.app._update_edge_params()
                
                c,v=imgui.slider_float("Input Res Scale",getattr(self.app,"input_scale_factor",DEFAULT_SETTINGS["input_scale_factor"]),0.25,4,"%.2f");
                if c: setattr(self.app,'input_scale_factor',v); self.app.scale_factor_ref[0]=v; self.app._update_edge_params()
                if imgui.button("Reset##IRSR"): setattr(self.app,'input_scale_factor',DEFAULT_SETTINGS["input_scale_factor"]); self.app.scale_factor_ref[0]=getattr(self.app,'input_scale_factor'); self.app._update_edge_params()
                
                c,v=imgui.slider_float("Global Size Scale",getattr(self.app,"size_scale_factor",DEFAULT_SETTINGS["size_scale_factor"]),0.0001,10,"%.4f");
                if c: setattr(self.app,'size_scale_factor',v); self.app._update_edge_params()
                
                imgui.text("Point Size Clamping")
                c,v=imgui.slider_float("Min Size (px)", getattr(self.app, "min_point_size", DEFAULT_SETTINGS["min_point_size"]), 0.1, 20.0)
                if c: setattr(self.app,'min_point_size',v); self.app._update_edge_params()
                c_clamp,v_clamp=imgui.checkbox("Enable Max Clamp##PSC", getattr(self.app, "enable_max_size_clamp", DEFAULT_SETTINGS["enable_max_size_clamp"]))
                if c_clamp: setattr(self.app,'enable_max_size_clamp',v_clamp); self.app._update_edge_params()
                if getattr(self.app, "enable_max_size_clamp", DEFAULT_SETTINGS["enable_max_size_clamp"]):
                    imgui.indent()
                    c,v = imgui.slider_float("Max Size (px)", getattr(self.app, "max_point_size", DEFAULT_SETTINGS["max_point_size"]), 1.0, 200.0)
                    if c: setattr(self.app,'max_point_size',v); self.app._update_edge_params()
                    imgui.unindent()
                if imgui.button("Reset##MinMaxPSR"): setattr(self.app,'min_point_size',DEFAULT_SETTINGS["min_point_size"]);setattr(self.app,'enable_max_size_clamp',DEFAULT_SETTINGS["enable_max_size_clamp"]);setattr(self.app,'max_point_size',DEFAULT_SETTINGS["max_point_size"]); self.app._update_edge_params()
                imgui.separator()

                imgui.text("Inv. Square Law")
                c,v=imgui.slider_float("Depth Exp.",getattr(self.app,"depth_exponent",DEFAULT_SETTINGS["depth_exponent"]),-4,4,"%.2f");
                if c: setattr(self.app,'depth_exponent',v); self.app._update_edge_params()
                if imgui.button("Reset##DExpR"): setattr(self.app,'depth_exponent',DEFAULT_SETTINGS["depth_exponent"]); self.app._update_edge_params()
                imgui.separator()

                imgui.text("Color Adjustments")
                c,v=imgui.slider_float("Saturation",getattr(self.app,"saturation",DEFAULT_SETTINGS["saturation"]),0,3);
                if c: setattr(self.app,'saturation',v) 
                if imgui.button("Reset##SatR"):setattr(self.app,'saturation',DEFAULT_SETTINGS["saturation"])
                c,v=imgui.slider_float("Brightness",getattr(self.app,"brightness",DEFAULT_SETTINGS["brightness"]),0,2);
                if c: setattr(self.app,'brightness',v)
                if imgui.button("Reset##BritR"):setattr(self.app,'brightness',DEFAULT_SETTINGS["brightness"])
                c,v=imgui.slider_float("Contrast",getattr(self.app,"contrast",DEFAULT_SETTINGS["contrast"]),0.1,3);
                if c: setattr(self.app,'contrast',v)
                if imgui.button("Reset##ContR"):setattr(self.app,'contrast',DEFAULT_SETTINGS["contrast"])
                imgui.separator()
                
                imgui.text("Viewer Camera Projection")
                c,v=imgui.checkbox("Use Ortho",getattr(self.app,"use_orthographic",DEFAULT_SETTINGS["use_orthographic"]));
                if c: setattr(self.app,'use_orthographic',v)
                if getattr(self.app,"use_orthographic",DEFAULT_SETTINGS["use_orthographic"]):
                    imgui.indent()
                    c,v=imgui.slider_float("Ortho Size",getattr(self.app,"orthographic_size",DEFAULT_SETTINGS["orthographic_size"]),0.1,100,"%.1f");
                    if c: setattr(self.app,'orthographic_size',v)
                    if imgui.button("Reset##OrthoR"):setattr(self.app,'orthographic_size',DEFAULT_SETTINGS["orthographic_size"])
                    imgui.unindent()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Debug")[0]:
                imgui.text("Show Debug Views:")
                _,v=imgui.checkbox("Cam Feed",getattr(self.app,"show_camera_feed",DEFAULT_SETTINGS["show_camera_feed"])); setattr(self.app,'show_camera_feed',v)
                _,v=imgui.checkbox("Depth Map",getattr(self.app,"show_depth_map",DEFAULT_SETTINGS["show_depth_map"])); setattr(self.app,'show_depth_map',v)
                _,v=imgui.checkbox("Edge Map",getattr(self.app,"show_edge_map",DEFAULT_SETTINGS["show_edge_map"])); setattr(self.app,'show_edge_map',v)
                _,v=imgui.checkbox("Smooth Map",getattr(self.app,"show_smoothing_map",DEFAULT_SETTINGS["show_smoothing_map"])); setattr(self.app,'show_smoothing_map',v)
                _,v=imgui.checkbox("Wavelet Map",getattr(self.app,"show_wavelet_map",DEFAULT_SETTINGS["show_wavelet_map"])); setattr(self.app,'show_wavelet_map',v)
                imgui.separator(); imgui.text("Performance Info:")
                imgui.text(f"Points: {self.app.renderer.current_point_count if self.app.renderer else 0}")
                imgui.text(f"Render FPS: {self.app.point_cloud_fps:.1f}")
                imgui.text(f"Input FPS: {self.app.input_fps:.1f}")
                imgui.text(f"Depth FPS: {self.app.depth_fps:.1f}")
                imgui.text(f"Latency: {self.app.latency_ms:.1f} ms")
                imgui.separator(); imgui.text("Show Stats in Overlay:")
                _,v=imgui.checkbox("Render FPS##O",getattr(self.app,"show_fps_overlay",DEFAULT_SETTINGS["show_fps_overlay"])); setattr(self.app,'show_fps_overlay',v)
                _,v=imgui.checkbox("Points##O",getattr(self.app,"show_points_overlay",DEFAULT_SETTINGS["show_points_overlay"])); setattr(self.app,'show_points_overlay',v)
                _,v=imgui.checkbox("Input FPS##O",getattr(self.app,"show_input_fps_overlay",DEFAULT_SETTINGS["show_input_fps_overlay"])); setattr(self.app,'show_input_fps_overlay',v)
                _,v=imgui.checkbox("Depth FPS##O",getattr(self.app,"show_depth_fps_overlay",DEFAULT_SETTINGS["show_depth_fps_overlay"])); setattr(self.app,'show_depth_fps_overlay',v)
                _,v=imgui.checkbox("Latency##O",getattr(self.app,"show_latency_overlay",DEFAULT_SETTINGS["show_latency_overlay"])); setattr(self.app,'show_latency_overlay',v)
                imgui.separator(); imgui.text("Visualize Sizing Calcs:")
                
                attrs_sizing = ["debug_show_input_distance", "debug_show_raw_diameter", "debug_show_density_factor", "debug_show_final_size"]
                labels_sizing = ["Input Dist", "Raw Diam.", "Density Factor", "Final Size"]
                for i, attr_name in enumerate(attrs_sizing):
                    val = getattr(self.app, attr_name, DEFAULT_SETTINGS[attr_name])
                    changed, new_val = imgui.checkbox(labels_sizing[i] + f"##SizingViz{i}", val) # Unique IDs
                    if changed:
                        setattr(self.app, attr_name, new_val)
                        if new_val: 
                            for other_attr in attrs_sizing:
                                if other_attr != attr_name: setattr(self.app, other_attr, False)
                        self.app._update_edge_params()

                imgui.separator(); imgui.text("Visualize Geometry:")
                _,v=imgui.checkbox("World Axes",getattr(self.app,"debug_show_world_axes",DEFAULT_SETTINGS["debug_show_world_axes"])); setattr(self.app,'debug_show_world_axes',v)
                _,v=imgui.checkbox("Input Frustum",getattr(self.app,"debug_show_input_frustum",DEFAULT_SETTINGS["debug_show_input_frustum"])); setattr(self.app,'debug_show_input_frustum',v)
                _,v=imgui.checkbox("Viewer Frustum",getattr(self.app,"debug_show_viewer_frustum",DEFAULT_SETTINGS["debug_show_viewer_frustum"])); setattr(self.app,'debug_show_viewer_frustum',v)
                _,v=imgui.checkbox("Input Rays",getattr(self.app,"debug_show_input_rays",DEFAULT_SETTINGS["debug_show_input_rays"])); setattr(self.app,'debug_show_input_rays',v)
                imgui.separator(); imgui.text("Depth Bias Calibration")
                
                if imgui.button("Capture Bias (Smoothed Ref)"):
                    setattr(self.app, 'pending_bias_capture_request', "smoothed_plane")
                    self.app._update_edge_params()
                    setattr(self.app, 'status_message', "Bias capture (smoothed) requested...")
                imgui.same_line()
                if imgui.button("Capture Bias (Mean Plane Ref)"):
                    setattr(self.app,'pending_bias_capture_request',"mean_plane"); self.app._update_edge_params()
                    setattr(self.app,'status_message',"Bias capture (mean plane) requested...")
                
                current_apply_bias = getattr(self.app, "apply_depth_bias", DEFAULT_SETTINGS["apply_depth_bias"])
                c, new_apply_bias = imgui.checkbox("Apply Depth Bias", current_apply_bias)
                if c: setattr(self.app,'apply_depth_bias',new_apply_bias); self.app._update_edge_params()
                
                if getattr(self.app, 'depth_bias_map', None) is not None: imgui.same_line(); imgui.text_colored("Bias Captured",0,1,0,1)
                elif getattr(self.app, "latest_depth_tensor_for_calib", None) is not None: imgui.same_line(); imgui.text_colored("Raw Depth Stored",1,1,0,1)
                else: imgui.same_line(); imgui.text_colored("Bias Not Capt.",0.7,0.7,0.7,1)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Settings")[0]:
                if imgui.button("Load Settings"): self.app.load_settings()
                imgui.same_line(); 
                if imgui.button("Save Settings"): self.app.save_settings()
                imgui.same_line(); 
                if imgui.button("Reset All Defaults"): self.app.reset_settings()
                imgui.text(f"Settings File: viewer_settings.json")
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end() 

        # Debug View Windows
        if getattr(self.app,"show_camera_feed",False) and self.app.renderer and self.app.renderer.camera_texture and self.app.latest_rgb_frame is not None:
            imgui.set_next_window_size(320,240,imgui.ONCE); imgui.set_next_window_position(10, getattr(self.app,'height',768)-250 if getattr(self.app,'height',768) > 250 else 10, imgui.ONCE)
            is_open, new_show_val = imgui.begin("Camera Feed", getattr(self.app,"show_camera_feed",False), flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS )
            if is_open: 
                aw = imgui.get_content_region_available()[0]; oh,ow = self.app.latest_rgb_frame.shape[:2]; ar=float(ow)/oh if oh>0 else 1.0
                imgui.image(self.app.renderer.camera_texture, aw, aw/ar if ar > 0 else aw)
            imgui.end()
            if not is_open: setattr(self.app,'show_camera_feed', False)
        
        if getattr(self.app,"show_depth_map",False) and self.app.renderer and self.app.renderer.depth_texture and self.app.latest_depth_map_viz is not None:
            imgui.set_next_window_size(320,240,imgui.ONCE); imgui.set_next_window_position(340, getattr(self.app,'height',768)-250 if getattr(self.app,'height',768) > 250 else 10, imgui.ONCE)
            is_open, new_show_val = imgui.begin("Depth Map", getattr(self.app,"show_depth_map",False), flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS)
            if is_open:
                aw = imgui.get_content_region_available()[0]; oh,ow = self.app.latest_depth_map_viz.shape[:2]; ar=float(ow)/oh if oh > 0 else 1.0
                imgui.image(self.app.renderer.depth_texture, aw, aw/ar if ar > 0 else aw)
            imgui.end()
            if not is_open: setattr(self.app,'show_depth_map', False)

        if getattr(self.app,"show_edge_map",False) and self.app.renderer and self.app.renderer.edge_texture and self.app.latest_edge_map is not None:
            imgui.set_next_window_size(320,240,imgui.ONCE); imgui.set_next_window_position(10, getattr(self.app,'height',768)-500 if getattr(self.app,'height',768) > 500 else 260, imgui.ONCE)
            is_open, new_show_val = imgui.begin("Edge Map", getattr(self.app,"show_edge_map",False), flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS)
            if is_open:
                aw = imgui.get_content_region_available()[0]; oh,ow = self.app.latest_edge_map.shape[:2]; ar=float(ow)/oh if oh > 0 else 1.0
                imgui.image(self.app.renderer.edge_texture, aw, aw/ar if ar > 0 else aw)
            imgui.end()
            if not is_open: setattr(self.app,'show_edge_map', False)

        if getattr(self.app,"show_smoothing_map",False) and self.app.renderer and self.app.renderer.smoothing_texture and self.app.latest_smoothing_map is not None:
            imgui.set_next_window_size(320,240,imgui.ONCE); imgui.set_next_window_position(340, getattr(self.app,'height',768)-500 if getattr(self.app,'height',768) > 500 else 260, imgui.ONCE)
            is_open, new_show_val = imgui.begin("Smoothing Alpha", getattr(self.app,"show_smoothing_map",False), flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS)
            if is_open:
                aw = imgui.get_content_region_available()[0]; oh,ow = self.app.latest_smoothing_map.shape[:2]; ar=float(ow)/oh if oh>0 else 1.0
                imgui.image(self.app.renderer.smoothing_texture, aw, aw/ar if ar > 0 else aw)
            imgui.end()
            if not is_open: setattr(self.app,'show_smoothing_map', False)

        if getattr(self.app,"show_wavelet_map",False) and self.app.renderer and self.app.renderer.imgui_wavelet_debug_texture and self.app.latest_wavelet_map is not None:
            imgui.set_next_window_size(320,240,imgui.ONCE); imgui.set_next_window_position(670, getattr(self.app,'height',768)-250 if getattr(self.app,'height',768) > 250 else 10, imgui.ONCE)
            is_open, new_show_val = imgui.begin("Wavelet Debug", getattr(self.app,"show_wavelet_map",False), flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS)
            if is_open:
                tex_id = self.app.renderer.imgui_wavelet_debug_texture
                tex_w = getattr(self.app.renderer, "imgui_wavelet_debug_texture_width", 1)
                tex_h = getattr(self.app.renderer, "imgui_wavelet_debug_texture_height", 1)
                aw = imgui.get_content_region_available()[0]; ar = float(tex_w)/tex_h if tex_h > 0 else 1.0
                imgui.image(tex_id, aw, aw/ar if ar > 0 else aw)
            imgui.end()
            if not is_open: setattr(self.app,'show_wavelet_map', False)
        
        if getattr(self.app, "render_mode", DEFAULT_SETTINGS["render_mode"]) == 3:
             setattr(self.app, "show_wavelet_map", True)

        gl.glEnable(gl.GL_BLEND); gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)
        try: gl.glBindVertexArray(self._default_vao_ui)
        except Exception: pass
        imgui.render()
        try: self.imgui_renderer.render(imgui.get_draw_data())
        except Exception as e_render_imgui: print(f"Warning: ImGui renderer error: {e_render_imgui}")
        gl.glEnable(gl.GL_DEPTH_TEST)

    def shutdown(self):
        if self.imgui_renderer: self.imgui_renderer.shutdown(); self.imgui_renderer = None
        if self._default_vao_ui: 
            try: 
                gl.glDeleteVertexArrays(1, self._default_vao_ui)
                self._default_vao_ui = None
            except: 
                pass
        print("UIManager shutdown.") 