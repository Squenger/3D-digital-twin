import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

class ManualCropWindow(ctk.CTkToplevel):
    def __init__(self, master, slices, on_confirm_callback):
        super().__init__(master)
        self.title("Manual Cropping (X, Y, Z)")
        self.geometry("900x700")
        # Ensure the popup stays on top and grabs focus
        self.transient(master)
        self.grab_set()
        
        self.on_confirm = on_confirm_callback
        
        self.depth = len(slices)
        self.height, self.width = slices[0].shape
        
        self.bounds = {
            "Z": [0, self.depth - 1],
            "Y": [0, self.height - 1],
            "X": [0, self.width - 1]
        }
        
        lbl = ctk.CTkLabel(self, text="Computing projections, please wait...", font=("Arial", 16))
        lbl.pack(expand=True)
        self.update()
        
        # Precompute projections
        vol = np.stack(slices, axis=0) # [Z, Y, X]
        self.proj_z = np.mean(vol, axis=0) # Shape: [Y, X]
        self.proj_y = np.mean(vol, axis=1) # Shape: [Z, X]
        self.proj_x = np.mean(vol, axis=2) # Shape: [Z, Y]
        del vol
        
        lbl.destroy()
        self.setup_ui()
        
    def setup_ui(self):
        lbl_inst = ctk.CTkLabel(self, text="Drag your mouse to highlight the region to keep (green highlight).", font=("Arial", 14, "bold"))
        lbl_inst.pack(pady=10)
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
        
        tab_z = self.tabview.add("Crop Z")
        tab_y = self.tabview.add("Crop Y")
        tab_x = self.tabview.add("Crop X")
        
        self.selectors = [] 
        
        # Z Crop: Use X projection [Z, Y]. Vertical axis is Z.
        self.create_plot(tab_z, self.proj_x, "Z", "vertical", f"Z Axis (0 to {self.depth})")
        # Y Crop: Use Z projection [Y, X]. Vertical axis is Y.
        self.create_plot(tab_y, self.proj_z, "Y", "vertical", f"Y Axis (0 to {self.height})")
        # X Crop: Use Z projection [Y, X]. Horizontal axis is X.
        self.create_plot(tab_x, self.proj_z, "X", "horizontal", f"X Axis (0 to {self.width})")
        
        btn_confirm = ctk.CTkButton(self, text="Confirm Crop", command=self.confirm)
        btn_confirm.pack(pady=10)
        
    def create_plot(self, parent, image, axis_name, direction, title):
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray', aspect='auto')
        ax.set_title(title)
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        def onselect(vmin, vmax):
            self.bounds[axis_name] = [int(vmin), int(vmax)]
            
        span = SpanSelector(
            ax, onselect, direction, useblit=True,
            interactive=True, drag_from_anywhere=True
        )
        self.selectors.append(span)
        
    def confirm(self):
        # Default bounds if not selected
        z1, z2 = self.bounds["Z"]
        y1, y2 = self.bounds["Y"]
        x1, x2 = self.bounds["X"]
        
        # Return standard format [z_start, z_end, y_start, y_end, x_start, x_end]
        result = [min(z1, z2), max(z1, z2), min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)]
        self.on_confirm(result)
        self.destroy()


class VarianceSelectorWindow(ctk.CTkToplevel):
    def __init__(self, master, slices, current_thresholds, on_confirm_callback):
        """
        current_thresholds: dict with keys 'Z', 'Y', 'X' and float values,
                            OR a single float (backward compat).
        on_confirm_callback: called with a dict {'Z': float, 'Y': float, 'X': float}.
        """
        super().__init__(master)
        self.title("Manual Variance Threshold — Per Axis")
        self.geometry("900x720")
        self.transient(master)
        self.grab_set()
        
        self.on_confirm = on_confirm_callback
        # Support both old single-float and new dict format
        if isinstance(current_thresholds, dict):
            self.thresholds = {k: float(v) for k, v in current_thresholds.items()}
        else:
            t = float(current_thresholds)
            self.thresholds = {"Z": t, "Y": t, "X": t}
        
        lbl = ctk.CTkLabel(self, text="Computing variance profiles, please wait...", font=("Arial", 16))
        lbl.pack(expand=True)
        self.update()
        
        self.compute_variances(slices)
        
        lbl.destroy()
        self.setup_ui()
        
    def compute_variances(self, slices):
        depth = len(slices)
        h, w = slices[0].shape
        
        z_mid = depth // 2
        z_margin = max(1, min(10, depth // 10))
        center_slices = np.stack(slices[max(0, z_mid-z_margin):min(depth, z_mid+z_margin+1)], axis=0)
        center_img = np.mean(center_slices, axis=0)
        
        y_mid, x_mid = h // 2, w // 2
        wy, wx = min(h, 400) // 2, min(w, 400) // 2
        step = max(1, 400 // 20)
        
        # Z variance: std inside center window for each Z slice
        self.var_z_idx = list(range(depth))
        self.var_z = []
        for i in range(depth):
            win = slices[i][y_mid-wy:y_mid+wy, x_mid-wx:x_mid+wx]
            val_pix = win[win > 5]
            self.var_z.append(float(np.std(val_pix)) if len(val_pix) > 0 else 0.0)
                
        # Y variance: scanning rows of the MIP center image
        self.var_y_idx = list(range(0, h, step))
        self.var_y = []
        for i in self.var_y_idx:
            win = center_img[i:i+step, x_mid-wx:x_mid+wx]
            val_pix = win[win > 5]
            self.var_y.append(float(np.std(val_pix)) if len(val_pix) > 0 else 0.0)
            
        # X variance: scanning columns of the MIP center image
        self.var_x_idx = list(range(0, w, step))
        self.var_x = []
        for i in self.var_x_idx:
            win = center_img[y_mid-wy:y_mid+wy, i:i+step]
            val_pix = win[win > 5]
            self.var_x.append(float(np.std(val_pix)) if len(val_pix) > 0 else 0.0)

    def setup_ui(self):
        lbl_inst = ctk.CTkLabel(
            self,
            text="Click on a graph to place the threshold line. Each axis is independent.",
            font=("Arial", 13, "bold")
        )
        lbl_inst.pack(pady=(10, 5))
        
        tabview = ctk.CTkTabview(self)
        tabview.pack(fill="both", expand=True, padx=10, pady=5)
        
        for axis_name, x_vals, y_vals in [
            ("Z", self.var_z_idx, self.var_z),
            ("Y", self.var_y_idx, self.var_y),
            ("X", self.var_x_idx, self.var_x),
        ]:
            tab = tabview.add(f"Axis {axis_name}")
            self._create_axis_tab(tab, axis_name, x_vals, y_vals)
        
        btn_confirm = ctk.CTkButton(self, text="Confirm All Thresholds", command=self.confirm)
        btn_confirm.pack(pady=10)

    def _create_axis_tab(self, parent, axis_name, x_vals, y_vals):
        thresh = self.thresholds[axis_name]

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x_vals, y_vals, color="steelblue", label=f"{axis_name} Std Dev")
        hline = ax.axhline(thresh, color='red', linestyle='--', linewidth=1.5, label=f"Threshold ({thresh:.1f})")
        ax.set_title(f"Variance Profile — Axis {axis_name}")
        ax.set_xlabel("Slice / Position index")
        ax.set_ylabel("Standard deviation")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        lbl = ctk.CTkLabel(parent, text=f"Threshold {axis_name}: {thresh:.2f}", font=("Arial", 13, "bold"), text_color="orange")
        lbl.pack(pady=4)

        def onclick(event, a=axis_name, h=hline, c=canvas, l=lbl):
            if event.inaxes == ax and event.ydata is not None:
                self.thresholds[a] = float(event.ydata)
                h.set_ydata([event.ydata, event.ydata])
                c.draw()
                l.configure(text=f"Threshold {a}: {event.ydata:.2f}")

        fig.canvas.mpl_connect('button_press_event', onclick)

    def confirm(self):
        self.on_confirm(dict(self.thresholds))
        self.destroy()
