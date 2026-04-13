import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

class ManualCropWindow(ctk.CTkToplevel):
    """
    A popup window allowing manual selection of X, Y, and Z bounds globally
    based on the geometric projections of the volume.
    """
    def __init__(self, master, slices, on_confirm_callback):
        """
        Initializes the cropping interface tool.

        Parameters
        ----------
        master : ctk.CTk
            The main parent window.
        slices : list of np.ndarray
            The loaded raw grayscale images forming the volume grid.
        on_confirm_callback : callable
            A function that takes a list of 6 integers `[z1, z2, y1, y2, x1, x2]`
            representing the new architectural boundary.
        """
        super().__init__(master)
        self.title("Manual Cropping Domain (X, Y, Z)")
        self.geometry("900x700")
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
        
        lbl = ctk.CTkLabel(self, text="Computing spatial projections, please wait...", font=("Arial", 16))
        lbl.pack(expand=True)
        self.update()
        
        # Precompute volumetric projections
        vol = np.stack(slices, axis=0) # [Z, Y, X]
        self.proj_z = np.mean(vol, axis=0) # Shape: [Y, X]
        self.proj_y = np.mean(vol, axis=1) # Shape: [Z, X]
        self.proj_x = np.mean(vol, axis=2) # Shape: [Z, Y]
        del vol
        
        lbl.destroy()
        self.setup_ui()
        
    def setup_ui(self):
        lbl_inst = ctk.CTkLabel(self, text="Drag your cursor to explicitly isolate the target matter (green highlight).", font=("Arial", 14, "bold"))
        lbl_inst.pack(pady=10)
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
        
        tab_z = self.tabview.add("Crop Z")
        tab_y = self.tabview.add("Crop Y")
        tab_x = self.tabview.add("Crop X")
        
        self.selectors = [] 
        
        # View setup
        self.create_plot(tab_z, self.proj_x, "Z", "vertical", f"Z Dimensional Origin (0 to {self.depth})")
        self.create_plot(tab_y, self.proj_z, "Y", "vertical", f"Y Dimensional Origin (0 to {self.height})")
        self.create_plot(tab_x, self.proj_z, "X", "horizontal", f"X Dimensional Origin (0 to {self.width})")
        
        btn_confirm = ctk.CTkButton(self, text="Confirm Physical Crop", command=self.confirm)
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
        z1, z2 = self.bounds["Z"]
        y1, y2 = self.bounds["Y"]
        x1, x2 = self.bounds["X"]
        
        result = [min(z1, z2), max(z1, z2), min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)]
        self.on_confirm(result)
        self.destroy()


class VarianceSelectorWindow(ctk.CTkToplevel):
    """
    An interface tool presenting visual profiles of the physical variance 
    (standard deviation) computed along structural grid lines. This helps researchers mathematically
    pinpoint optimal threshold cutoffs for empty space.
    """
    def __init__(self, master, slices, current_thresholds, on_confirm_callback):
        """
        Initializes the variance selection tool.

        Parameters
        ----------
        master : ctk.CTk
            The main parent window.
        slices : list of np.ndarray
            The dataset of image geometries.
        current_thresholds : dict or float
            The currently operational limits structured as `{"Z": float, "Y": float, "X": float}` 
        on_confirm_callback : callable
            A function receiving the new thresholds dict object upon user confirmation.
        """
        super().__init__(master)
        self.title("Limit Control — Explicit Variance Analysis (Per Axis)")
        self.geometry("900x720")
        self.transient(master)
        self.grab_set()
        
        self.on_confirm = on_confirm_callback
        
        if isinstance(current_thresholds, dict):
            self.thresholds = {k: float(v) for k, v in current_thresholds.items()}
        else:
            t = float(current_thresholds)
            self.thresholds = {"Z": t, "Y": t, "X": t}
        
        lbl = ctk.CTkLabel(self, text="Computing standard deviation maps, please wait...", font=("Arial", 16))
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
        
        self.var_z_idx = list(range(depth))
        self.var_z = []
        for i in range(depth):
            win = slices[i][y_mid-wy:y_mid+wy, x_mid-wx:x_mid+wx]
            val_pix = win[win > 5]
            self.var_z.append(float(np.std(val_pix)) if len(val_pix) > 0 else 0.0)
                
        self.var_y_idx = list(range(0, h, step))
        self.var_y = []
        for i in self.var_y_idx:
            win = center_img[i:i+step, x_mid-wx:x_mid+wx]
            val_pix = win[win > 5]
            self.var_y.append(float(np.std(val_pix)) if len(val_pix) > 0 else 0.0)
            
        self.var_x_idx = list(range(0, w, step))
        self.var_x = []
        for i in self.var_x_idx:
            win = center_img[y_mid-wy:y_mid+wy, i:i+step]
            val_pix = win[win > 5]
            self.var_x.append(float(np.std(val_pix)) if len(val_pix) > 0 else 0.0)

    def setup_ui(self):
        lbl_inst = ctk.CTkLabel(
            self,
            text="Explicitly click on a plot profile to assign the minimum deviation cutoff. Axes are mutually independent.",
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
        
        btn_confirm = ctk.CTkButton(self, text="Confirm Theoretical Parameters", command=self.confirm)
        btn_confirm.pack(pady=10)

    def _create_axis_tab(self, parent, axis_name, x_vals, y_vals):
        thresh = self.thresholds[axis_name]

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x_vals, y_vals, color="steelblue", label=f"{axis_name} Deviation Profile")
        hline = ax.axhline(thresh, color='red', linestyle='--', linewidth=1.5, label=f"Boundary Cutoff ({thresh:.1f})")
        ax.set_title(f"Empirical Variance Chart — Domain {axis_name}")
        ax.set_xlabel("Spatio-Temporal Index")
        ax.set_ylabel("Standard Deviation Intensity")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        lbl = ctk.CTkLabel(parent, text=f"Limit Parameter {axis_name}: {thresh:.2f}", font=("Arial", 13, "bold"), text_color="orange")
        lbl.pack(pady=4)

        def onclick(event, a=axis_name, h=hline, c=canvas, l=lbl):
            if event.inaxes == ax and event.ydata is not None:
                self.thresholds[a] = float(event.ydata)
                h.set_ydata([event.ydata, event.ydata])
                c.draw()
                l.configure(text=f"Limit Parameter {a}: {event.ydata:.2f}")

        fig.canvas.mpl_connect('button_press_event', onclick)

    def confirm(self):
        self.on_confirm(dict(self.thresholds))
        self.destroy()
