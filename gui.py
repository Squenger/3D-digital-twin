import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import trimesh
from PIL import Image, ImageTk
import subprocess
import sys
import math


import customtkinter as ctk

from main import TifTo3D
from pathlib import Path
ctk.set_default_color_theme("blue")

class TifTo3D_GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("3D Digital Twin - Reconstruction via MICRO CT")
        self.geometry("800x750")
        self.minsize(800, 750)
        
        # Grid setup
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.tab_recon = self.tabview.add("Reconstruction")
        self.tab_visu = self.tabview.add("Visualization")
        
        self.setup_reconstruction_tab()
        self.setup_visualisation_tab()

    def setup_reconstruction_tab(self):
        self.tab_recon.grid_columnconfigure(1, weight=1)
        
        # --- Paths Section ---
        self.frame_paths = ctk.CTkFrame(self.tab_recon)
        self.frame_paths.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.frame_paths.grid_columnconfigure(1, weight=1)
        
        # Batch Checkbox
        self.batch_var = ctk.BooleanVar(value=False)
        self.chk_batch = ctk.CTkCheckBox(self.frame_paths, text="Multiple reconstruction", variable=self.batch_var, command=self.toggle_batch_mode)
        self.chk_batch.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Input folder
        self.lbl_input = ctk.CTkLabel(self.frame_paths, text="Input folder:")
        self.lbl_input.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.entry_input = ctk.CTkEntry(self.frame_paths)
        self.entry_input.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.entry_input.insert(0, '/Volumes/KINGSTON/DATA_MICRO_CT/sample 3/Sample 3 15um/S3_15um_Original')
        
        self.btn_input = ctk.CTkButton(self.frame_paths, text="Browse", width=100, command=self.browse_input)
        self.btn_input.grid(row=1, column=2, padx=10, pady=5)
        
        # Output folder / file
        self.lbl_output = ctk.CTkLabel(self.frame_paths, text="Output file:")
        self.lbl_output.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.entry_output = ctk.CTkEntry(self.frame_paths)
        self.entry_output.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.entry_output.insert(0, './sortie/modele3(20).stl')
        
        self.btn_output = ctk.CTkButton(self.frame_paths, text="Browse", width=100, command=self.browse_output)
        self.btn_output.grid(row=2, column=2, padx=10, pady=5)

        # --- Parameters Section ---
        self.frame_params = ctk.CTkFrame(self.tab_recon)
        self.frame_params.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        params_labels = [
            ("Layer thickness (mm)", "0.015", "layer_thickness"),
            ("Voxel size (mm)", "0.015", "voxel_size"),
            ("Threshold (Auto if empty)", "", "threshold"),
            ("Component threshold (faces)", "1000", "component_threshold"),
            ("Window size (pixels)", "400", "window_size"),
            ("Std Dev Z (vertical)", "28.0", "std_dev_z"),
            ("Std Dev Y (frontal)", "21.0", "std_dev_y"),
            ("Std Dev X (sagittal)", "21.0", "std_dev_x"),
        ]
        
        self.entries = {}
        for i, (label_text, default_val, key) in enumerate(params_labels):
            lbl = ctk.CTkLabel(self.frame_params, text=label_text)
            lbl.grid(row=i, column=0, padx=10, pady=5, sticky="w")
            ent = ctk.CTkEntry(self.frame_params, width=100)
            ent.insert(0, default_val)
            ent.grid(row=i, column=1, padx=10, pady=5, sticky="e")
            self.entries[key] = ent
            
        self.btn_manual_var = ctk.CTkButton(self.frame_params, text="Manual variance limit...", fg_color="gray", command=self.open_variance_selector)
        self.btn_manual_var.grid(row=len(params_labels), column=0, columnspan=2, pady=10)
            
        # --- Options Section ---
        self.frame_options = ctk.CTkFrame(self.tab_recon)
        self.frame_options.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.fast_test_var = ctk.BooleanVar(value=False)
        self.manual_crop_var = ctk.BooleanVar(value=False)
        self.solid_var = ctk.BooleanVar(value=False)
        self.visualize_var = ctk.BooleanVar(value=False)
        self.post_process_var = ctk.BooleanVar(value=True)
        self.gen_porosity_mesh_var = ctk.BooleanVar(value=False)
        self.save_slices_var = ctk.BooleanVar(value=False)
        
        ctk.CTkCheckBox(self.frame_options, text="Fast test (500 slices)", variable=self.fast_test_var).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self.frame_options, text="Manual Cropping", variable=self.manual_crop_var).grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self.frame_options, text="Solid model (Beta)", variable=self.solid_var, command=self.on_solid_toggle).grid(row=2, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self.frame_options, text="Open 3D view at the end", variable=self.visualize_var).grid(row=3, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self.frame_options, text="Estimate Porosity & Extract Slices", variable=self.post_process_var).grid(row=4, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self.frame_options, text="Generate 3D porosity model", variable=self.gen_porosity_mesh_var).grid(row=5, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self.frame_options, text="Save Cross Slices to Disk", variable=self.save_slices_var).grid(row=6, column=0, padx=10, pady=10, sticky="w")
        
        # --- Execution Section ---
        self.frame_run = ctk.CTkFrame(self.tab_recon)
        self.frame_run.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.frame_run.grid_columnconfigure(0, weight=1)
        self.frame_run.grid_columnconfigure(1, weight=1)
        
        self._cancel_event = threading.Event()
        
        self.btn_start = ctk.CTkButton(self.frame_run, text="Start reconstruction", command=self.start_reconstruction_thread)
        self.btn_start.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="e")
        
        self.btn_cancel = ctk.CTkButton(
            self.frame_run, text="Cancel", fg_color="#c0392b", hover_color="#922b21",
            state="disabled", command=self.cancel_reconstruction
        )
        self.btn_cancel.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="w")
        
        self.status_label = ctk.CTkLabel(self.frame_run, text="Ready.")
        self.status_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0,5))
        
        self.progress_bar = ctk.CTkProgressBar(self.frame_run)
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=20, pady=(0,10), sticky="ew")
        self.progress_bar.set(0)
        
    def setup_visualisation_tab(self):
        self.tab_visu.grid_columnconfigure(0, weight=1)
        self.tab_visu.grid_rowconfigure(0, weight=1)
        
        # Visualization sub-tabs
        self.vis_tabview = ctk.CTkTabview(self.tab_visu)
        self.vis_tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.tab_v3d = self.vis_tabview.add("3D Model")
        self.tab_vslices = self.vis_tabview.add("Cross-sections")
        self.tab_vgraphs = self.vis_tabview.add("Graphs")
        
        self._setup_vis_3d()
        self._setup_vis_slices()
        self._setup_vis_graphs()
        
    def _setup_vis_3d(self):
        self.tab_v3d.grid_columnconfigure(0, weight=1)
        
        lbl = ctk.CTkLabel(self.tab_v3d, text="Interactive 3D model visualization", font=ctk.CTkFont(size=14, weight="bold"))
        lbl.grid(row=0, column=0, pady=20)
        
        self.lbl_current_3d = ctk.CTkLabel(self.tab_v3d, text="No model loaded.")
        self.lbl_current_3d.grid(row=1, column=0, pady=(0, 10))
        
        btn_view = ctk.CTkButton(self.tab_v3d, text="Manually load a (.stl) file...", command=self.load_and_view_model)
        btn_view.grid(row=2, column=0, pady=10)
        
        self.btn_view_current = ctk.CTkButton(self.tab_v3d, text="Open current model", state="disabled", fg_color="green", hover_color="darkgreen")
        self.btn_view_current.grid(row=3, column=0, pady=10)
        
        self.btn_view_porosity = ctk.CTkButton(self.tab_v3d, text="View porosity model", state="disabled", fg_color="orange", hover_color="darkorange")
        self.btn_view_porosity.grid(row=4, column=0, pady=10)
        
    def _setup_vis_slices(self):
        self.tab_vslices.grid_columnconfigure(0, weight=1)
        self.tab_vslices.grid_rowconfigure(1, weight=1)
        
        frame_controls = ctk.CTkFrame(self.tab_vslices)
        frame_controls.grid(row=0, column=0, sticky="ew", pady=5)
        
        btn_load = ctk.CTkButton(frame_controls, text="Load image folder...", command=self.load_slice_folder)
        btn_load.pack(side="left", padx=10, pady=5)
        
        # Axis Selector
        self.axis_var = ctk.StringVar(value="Z_Axial")
        self.seg_axis = ctk.CTkSegmentedButton(frame_controls, values=["Z_Axial", "Y_Frontal", "X_Sagittal"], variable=self.axis_var, command=self.change_axis)
        self.seg_axis.pack(side="left", padx=10)
        
        self.btn_prev_slice = ctk.CTkButton(frame_controls, text="<< Previous", width=80, command=self.prev_slice)
        self.btn_prev_slice.pack(side="left", padx=5)
        
        self.lbl_slice_info = ctk.CTkLabel(frame_controls, text="0 / 0")
        self.lbl_slice_info.pack(side="left", padx=5)
        
        self.btn_next_slice = ctk.CTkButton(frame_controls, text="Next >>", width=80, command=self.next_slice)
        self.btn_next_slice.pack(side="left", padx=5)
        
        # Ruler UI Controls
        self.ruler_var = ctk.BooleanVar(value=False)
        self.btn_ruler = ctk.CTkButton(frame_controls, text="Ruler: OFF", width=100, fg_color="gray", command=self.toggle_ruler)
        self.btn_ruler.pack(side="left", padx=10)
        
        self.lbl_measure = ctk.CTkLabel(frame_controls, text="Dist: -- mm", text_color="cyan", font=ctk.CTkFont(weight="bold"))
        self.lbl_measure.pack(side="left", padx=5)
        
        self.slice_canvas = tk.Canvas(self.tab_vslices, bg="#2b2b2b", width=600, height=600, highlightthickness=0)
        self.slice_canvas.grid(row=1, column=0, sticky="nsew", pady=10)
        
        self.slice_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.slice_canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.slice_canvas.bind("<Button-5>", self.on_mouse_wheel)
        
        self.slice_canvas.bind("<ButtonPress-1>", self.on_ruler_press)
        self.slice_canvas.bind("<B1-Motion>", self.on_ruler_drag)
        self.slice_canvas.bind("<ButtonRelease-1>", self.on_ruler_release)
        
        self.tab_vslices.bind("<MouseWheel>", self.on_mouse_wheel)
        self.tab_vslices.bind("<Button-4>", self.on_mouse_wheel)
        self.tab_vslices.bind("<Button-5>", self.on_mouse_wheel)
        
        self.slice_files = []
        self.slice_dict = {"Z_Axial": [], "Y_Frontal": [], "X_Sagittal": []}
        self.current_slice_index = 0
        
        self.slice_resolutions = {"Z_Axial": (1.0, 1.0), "Y_Frontal": (1.0, 1.0), "X_Sagittal": (1.0, 1.0)}
        self.ruler_line = None
        self.ruler_start = (0, 0)
        self.img_scale = 1.0
        self.canvas_image_tk = None

    def _setup_vis_graphs(self):
        self.tab_vgraphs.grid_columnconfigure(0, weight=1)
        self.tab_vgraphs.grid_rowconfigure(1, weight=1)
        
        frame_controls = ctk.CTkFrame(self.tab_vgraphs)
        frame_controls.grid(row=0, column=0, sticky="ew", pady=5)
        
        btn_load = ctk.CTkButton(frame_controls, text="Open a graph (.png, .jpg)...", command=self.load_graph)
        btn_load.pack(side="left", padx=10, pady=5)
        
        self.lbl_stats = ctk.CTkLabel(frame_controls, text="", font=ctk.CTkFont(weight="bold", size=14))
        self.lbl_stats.pack(side="right", padx=20)
        
        self.graph_image_label = ctk.CTkLabel(self.tab_vgraphs, text="No graph loaded")
        self.graph_image_label.grid(row=1, column=0, sticky="nsew", pady=10)

    # --- Slices Viewer Logic ---
    def load_slice_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
            files = sorted([f for f in Path(folder).iterdir() if f.suffix.lower() in valid_exts and f.is_file()])
            if files:
                self.slice_files = files
                self.current_slice_index = 0
                
                try:
                    v_sz = float(self.entries["voxel_size"].get() or "0.015")
                    l_tk = float(self.entries["layer_thickness"].get() or "0.015")
                except ValueError:
                    v_sz = l_tk = 1.0
                    
                self.slice_resolutions = {
                    "Z_Axial": (v_sz, v_sz),
                    "Y_Frontal": (v_sz, l_tk),
                    "X_Sagittal": (v_sz, l_tk)
                }
                
                self.show_current_slice()
            else:
                self.slice_canvas.delete("all")
                self.slice_canvas.create_text(300, 300, text="No valid image found in this folder", fill="white")
                self.lbl_slice_info.configure(text="0 / 0")

    def change_axis(self, value):
        self.slice_files = self.slice_dict.get(value, [])
        self.current_slice_index = 0
        if self.slice_files:
            self.show_current_slice()
        else:
            self.slice_canvas.delete("all")
            self.slice_canvas.create_text(300, 300, text="No image for this axis", fill="white")
            self.lbl_slice_info.configure(text="0 / 0")

    def show_current_slice(self):
        if not self.slice_files: return
        try:
            path = self.slice_files[self.current_slice_index]
            img = Image.open(path)
            orig_w, orig_h = img.size
            
            # Resize for display (max 600x600 approx.)
            img.thumbnail((600, 600))
            new_w, new_h = img.size
            self.img_scale = orig_w / new_w if new_w > 0 else 1.0
            
            self.canvas_image_tk = ImageTk.PhotoImage(img)
            self.slice_canvas.delete("all")
            self.slice_canvas.create_image(300, 300, image=self.canvas_image_tk, anchor="center")
            self.lbl_slice_info.configure(text=f"{self.current_slice_index + 1} / {len(self.slice_files)}")
            
            self.ruler_line = None
            self.lbl_measure.configure(text="Dist: -- mm")
        except Exception as e:
            self.slice_canvas.delete("all")
            self.slice_canvas.create_text(300, 300, text=f"Loading error: {e}", fill="red")

    def toggle_ruler(self):
        val = not self.ruler_var.get()
        self.ruler_var.set(val)
        if val:
            self.btn_ruler.configure(text="Ruler: ON", fg_color="green")
            self.slice_canvas.configure(cursor="crosshair")
        else:
            self.btn_ruler.configure(text="Ruler: OFF", fg_color="gray")
            self.slice_canvas.configure(cursor="arrow")
            if self.ruler_line:
                self.slice_canvas.delete(self.ruler_line)
                self.ruler_line = None
            self.lbl_measure.configure(text="Dist: -- mm")

    def on_ruler_press(self, event):
        if not self.ruler_var.get() or not self.slice_files: return
        self.ruler_start = (event.x, event.y)
        if self.ruler_line:
            self.slice_canvas.delete(self.ruler_line)
        self.ruler_line = self.slice_canvas.create_line(event.x, event.y, event.x, event.y, fill="cyan", width=2)
        self.lbl_measure.configure(text="Dist: 0.000 mm")

    def on_ruler_drag(self, event):
        if not self.ruler_var.get() or not self.ruler_line: return
        self.slice_canvas.coords(self.ruler_line, self.ruler_start[0], self.ruler_start[1], event.x, event.y)
        self._update_measurement((event.x, event.y))

    def on_ruler_release(self, event):
        if not self.ruler_var.get() or not self.ruler_line: return
        self._update_measurement((event.x, event.y))

    def _update_measurement(self, end_pos):
        dx = end_pos[0] - self.ruler_start[0]
        dy = end_pos[1] - self.ruler_start[1]
        
        dx_orig = dx * self.img_scale
        dy_orig = dy * self.img_scale
        
        axis = self.axis_var.get()
        res_x, res_y = self.slice_resolutions.get(axis, (1.0, 1.0))
        
        real_dx = dx_orig * res_x
        real_dy = dy_orig * res_y
        real_dist = math.sqrt(real_dx**2 + real_dy**2)
        
        self.lbl_measure.configure(text=f"Dist: {real_dist:.3f} mm")

    def next_slice(self):
        if self.slice_files and self.current_slice_index < len(self.slice_files) - 1:
            self.current_slice_index += 1
            self.show_current_slice()

    def prev_slice(self):
        if self.slice_files and self.current_slice_index > 0:
            self.current_slice_index -= 1
            self.show_current_slice()
            
    def on_mouse_wheel(self, event):
        if not self.slice_files: return
        
        # Determine direction based on OS
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            self.prev_slice()
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            self.next_slice()

    # --- Graphs Viewer Logic ---
    def load_graph(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path:
            try:
                img = Image.open(path)
                img.thumbnail((700, 700))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self.graph_image_label.configure(image=ctk_img, text="")
            except Exception as e:
                self.graph_image_label.configure(text=f"Loading error: {e}")
        
    def toggle_batch_mode(self):
        if self.batch_var.get():
            self.lbl_input.configure(text="Parent folder:")
            self.lbl_output.configure(text="Output folder:")
            self.entry_output.delete(0, 'end')
        else:
            self.lbl_input.configure(text="Input folder:")
            self.lbl_output.configure(text="Output file:")
            self.entry_output.delete(0, 'end')
            
    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_input.delete(0, 'end')
            self.entry_input.insert(0, folder)
            
    def browse_output(self):
        if self.batch_var.get():
            path = filedialog.askdirectory()
        else:
            if self.solid_var.get():
                path = filedialog.asksaveasfilename(defaultextension=".step", filetypes=[("STEP CAD", "*.step"), ("All files", "*.*")])
            else:
                path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("Stereolithography", "*.stl"), ("All files", "*.*")])
        
        if path:
            self.entry_output.delete(0, 'end')
            self.entry_output.insert(0, path)
            
    def on_solid_toggle(self):
        """Automatically suggest .step extension when solid mode is enabled."""
        current_path = self.entry_output.get().strip()
        if not current_path: return
        
        path = Path(current_path)
        if self.solid_var.get():
            if path.suffix.lower() == ".stl":
                new_path = path.with_suffix(".step")
                self.entry_output.delete(0, 'end')
                self.entry_output.insert(0, str(new_path))
        else:
            if path.suffix.lower() == ".step":
                new_path = path.with_suffix(".stl")
                self.entry_output.delete(0, 'end')
                self.entry_output.insert(0, str(new_path))

    # --- Interactive Features ---
    def interactive_crop_callback(self, slices):
        self._crop_bounds = None
        self._crop_event = threading.Event()
        
        def _show_crop():
            from interactive_windows import ManualCropWindow
            def _on_confirm(bounds):
                self._crop_bounds = bounds
                self._crop_event.set()
            # Avoid duplicate windows
            ManualCropWindow(self, slices, _on_confirm)
            
        self.after(0, _show_crop)
        self._crop_event.wait()
        return self._crop_bounds

    def open_variance_selector(self):
        self.btn_manual_var.configure(state="disabled")
        in_path = self.entry_input.get().strip()
        if not in_path:
            messagebox.showerror("Error", "Input folder required.")
            self.btn_manual_var.configure(state="normal")
            return
            
        def _load_and_select():
            try:
                from tif_loader import TifLoader
                loader = TifLoader(Path(in_path))
                slices = loader.load(fast_test=True)
                
                def _show_popup():
                    from interactive_windows import VarianceSelectorWindow
                    current = {
                        "Z": float(self.entries["std_dev_z"].get() or "28.0"),
                        "Y": float(self.entries["std_dev_y"].get() or "28.0"),
                        "X": float(self.entries["std_dev_x"].get() or "28.0"),
                    }
                    
                    def on_confirm(result_dict):
                        for axis, key in [("Z", "std_dev_z"), ("Y", "std_dev_y"), ("X", "std_dev_x")]:
                            self.entries[key].delete(0, 'end')
                            self.entries[key].insert(0, f"{result_dict[axis]:.2f}")
                        self.btn_manual_var.configure(state="normal")
                        
                    VarianceSelectorWindow(self, slices, current, on_confirm)
                    
                self.after(0, _show_popup)
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self.btn_manual_var.configure(state="normal"))
                
        threading.Thread(target=_load_and_select).start()

    def load_and_view_model(self):
        path = filedialog.askopenfilename(filetypes=[("STL files", "*.stl")])
        if path:
            try:
                # Use subprocess to isolate Trimesh/Pyglet from Tkinter to avoid macOS segfaults
                subprocess.Popen([
                    sys.executable, 
                    "-c", 
                    "import sys, trimesh; trimesh.load_mesh(sys.argv[1]).show()", 
                    str(path)
                ])
            except Exception as e:
                messagebox.showerror("Error", f"Could not load the model: {e}")

    def update_progress(self, current, total, message=None):
        pct = current / total if total > 0 else 0
        def _update():
            self.progress_bar.set(pct)
            if message:
                self.status_label.configure(text=message)
        self.after(0, _update)

    def start_reconstruction_thread(self):
        self._cancel_event.clear()
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        t = threading.Thread(target=self.run_reconstruction, daemon=True)
        t.start()
        
    def cancel_reconstruction(self):
        self._cancel_event.set()
        self.btn_cancel.configure(state="disabled")
        self.update_progress(0, 1, "Cancelling... Please wait.")
        
    def _safe_show_mesh(self, path):
        try:
            # Use subprocess to isolate Trimesh/Pyglet from Tkinter to avoid macOS segfaults
            subprocess.Popen([
                sys.executable, 
                "-c", 
                "import sys, trimesh; trimesh.load_mesh(sys.argv[1]).show()", 
                str(path)
            ])
        except Exception as e:
            messagebox.showerror("Error", f"Could not display the model: {e}")

    def run_reconstruction(self):
        try:
            self.update_progress(0, 1, "Initializing...")   
            
            in_path = self.entry_input.get().strip()
            out_path = self.entry_output.get().strip()
            
            if not in_path or not out_path:
                raise ValueError("Please specify the input and output paths.")
                
            in_pt = Path(in_path)
            out_pt = Path(out_path)
            
            # Parameters
            def get_float(k, default):
                val = self.entries[k].get().strip()
                return float(val) if val else default
                
            def get_int(k, default):
                val = self.entries[k].get().strip()
                return int(val) if val else default
                
            layer_t = get_float("layer_thickness", 0.015)
            voxel_s = get_float("voxel_size", 0.015)
            thresh_val = self.entries["threshold"].get().strip()
            thresh = int(thresh_val) if thresh_val else None
            comp_thresh = get_int("component_threshold", 1000)
            win_size = get_int("window_size", 400)
            std_dev_z = get_float("std_dev_z", 28.0)
            std_dev_y = get_float("std_dev_y", 28.0)
            std_dev_x = get_float("std_dev_x", 28.0)
            
            fast = self.fast_test_var.get()
            solid = self.solid_var.get()
            visu = self.visualize_var.get()
            post = self.post_process_var.get()
            gen_por = self.gen_porosity_mesh_var.get()
            save_sl = self.save_slices_var.get()
            
            is_batch = self.batch_var.get()
            
            if is_batch:
                samples = [d for d in in_pt.iterdir() if d.is_dir()]
                if not samples:
                    raise ValueError(f"No sample folder found in {in_pt}.")
                
                out_pt.mkdir(parents=True, exist_ok=True)
                
                for i, sample_dir in enumerate(samples, 1):
                    if self._cancel_event.is_set():
                        self.update_progress(0, 1, "Cancelled.")
                        return
                        
                    sample_name = sample_dir.name
                    cur_out = out_pt / f"{sample_name}.stl"
                    
                    self.update_progress(i-1, len(samples), f"[{i}/{len(samples)}] Processing: {sample_name}...")
                    
                    # Wrapper progress for batch
                    def batch_progress(c, t, msg=None):
                        batch_pct = (i - 1 + (c / t)) / len(samples)
                        if msg:
                            full_msg = f"[{i}/{len(samples)}] {sample_name}: {msg}"
                        else:
                            full_msg = None
                        self.update_progress(batch_pct, 1, full_msg)

                    pipeline = TifTo3D(
                        input_folder=sample_dir,
                        output_path=cur_out,
                        threshold=thresh,
                        layer_thickness=layer_t,
                        voxel_size=voxel_s,
                        visualize=False,  
                        solid=solid,
                        fast_test=fast,
                        component_threshold=comp_thresh,
                        window_size=win_size,
                        std_dev_z=std_dev_z,
                        std_dev_y=std_dev_y,
                        std_dev_x=std_dev_x,
                        post_process=post,
                        generate_porosity_mesh=gen_por,
                        save_slices=save_sl,
                        progress_callback=batch_progress,
                        interactive_crop_callback=self.interactive_crop_callback if self.manual_crop_var.get() else None
                    )
                    pipeline.execute()
                    
                self.update_progress(1, 1, "Batch processing complete.")
                messagebox.showinfo("Success", "Batch processing successfully completed.")
            else:
                pipeline = TifTo3D(
                    input_folder=in_pt,
                    output_path=out_pt,
                    threshold=thresh,
                    layer_thickness=layer_t,
                    voxel_size=voxel_s,
                    visualize=False,  
                    solid=solid,
                    fast_test=fast,
                    component_threshold=comp_thresh,
                    window_size=win_size,
                    std_dev_z=std_dev_z,
                    std_dev_y=std_dev_y,
                    std_dev_x=std_dev_x,
                    post_process=post,
                    generate_porosity_mesh=gen_por,
                    save_slices=save_sl,
                    progress_callback=self.update_progress,
                    interactive_crop_callback=self.interactive_crop_callback if self.manual_crop_var.get() else None
                )
                pipeline.execute()
                self.update_progress(1, 1, "Reconstruction complete.")
                messagebox.showinfo("Success", "Reconstruction successfully completed.")
                
                # Update 3D Model tab
                def prepare_3d_vis():
                    self.lbl_current_3d.configure(text=f"Current model generated: {out_pt.name}")
                    self.btn_view_current.configure(state="normal", command=lambda: self._safe_show_mesh(out_pt))
                    
                    if hasattr(pipeline, 'porosity_model_path') and pipeline.porosity_model_path:
                        self.btn_view_porosity.configure(state="normal", command=lambda: self._safe_show_mesh(pipeline.porosity_model_path))
                    else:
                        self.btn_view_porosity.configure(state="disabled")
                    
                    if hasattr(pipeline, 'generated_slices') and pipeline.generated_slices:
                        if isinstance(pipeline.generated_slices, dict):
                            self.slice_dict = pipeline.generated_slices
                            if hasattr(pipeline, 'slice_resolutions'):
                                self.slice_resolutions = pipeline.slice_resolutions
                            cur_ax = self.axis_var.get()
                            self.slice_files = self.slice_dict.get(cur_ax, [])
                        else:
                            self.slice_files = pipeline.generated_slices
                            
                        self.current_slice_index = 0
                        self.tabview.set("Visualization")
                        self.vis_tabview.set("Cross-sections")
                        if self.slice_files:
                            self.show_current_slice()
                        
                    if hasattr(pipeline, 'porosity_rate'):
                        self.lbl_stats.configure(text=f"Calculated global porosity: {pipeline.porosity_rate:.2f} %", text_color="orange")
                        
                    if hasattr(pipeline, 'generated_graphs') and len(pipeline.generated_graphs) > 0:
                        try:
                            img = Image.open(pipeline.generated_graphs[0])
                            img.thumbnail((700, 700))
                            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                            self.graph_image_label.configure(image=ctk_img, text="")
                            self.vis_tabview.set("Graphs")  # Switch to graphs to show it
                        except Exception as e:
                            print(f"[GUI Error loading graph] {e}")
                            
                    elif visu and not (hasattr(pipeline, 'generated_slices') and pipeline.generated_slices):
                        self.tabview.set("Visualization")
                        self.vis_tabview.set("3D Model")
                        self._safe_show_mesh(out_pt)
                        
                self.after(0, prepare_3d_vis)
                
        except Exception as e:
            if not self._cancel_event.is_set():
                self.update_progress(0, 1, f"Error: {e}")
                messagebox.showerror("Error", str(e))
        finally:
            def _reset():
                self.btn_start.configure(state="normal")
                self.btn_cancel.configure(state="disabled")
                if self._cancel_event.is_set():
                    self.update_progress(0, 1, "Cancelled. Ready to restart.")
                    self.progress_bar.set(0)
            self.after(0, _reset)

if __name__ == "__main__":
    app = TifTo3D_GUI()
    app.mainloop()
