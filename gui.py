#!/usr/bin/env python3
"""
GUI application for video frame cleanup with drag-and-drop support.
"""

import os
import sys
import threading
from pathlib import Path
from queue import Queue

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    print("Error: tkinter not found. Install with: python -m pip install tk")
    sys.exit(1)

from remove_letterbox import process_video


class VideoCleanerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Cleaner")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.input_video = None
        self.output_video = None
        self.processing = False
        self.queue = Queue()
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        self._create_widgets()
        self._start_queue_check()
    
    def _create_widgets(self):
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10, padx=10, fill='x')
        title_label = ttk.Label(title_frame, text="Video Frame Cleaner", font=("Arial", 16, "bold"))
        title_label.pack()
        
        # Input section
        input_frame = ttk.LabelFrame(self.root, text="Input Video", padding=10)
        input_frame.pack(pady=5, padx=10, fill='x')
        
        self.input_label = ttk.Label(input_frame, text="No video selected", foreground="gray")
        self.input_label.pack(side='left', fill='x', expand=True, padx=5)
        
        ttk.Button(input_frame, text="Browse", command=self._select_input).pack(side='right', padx=5)
        ttk.Button(input_frame, text="Clear", command=self._clear_input).pack(side='right', padx=2)
        
        # Output section
        output_frame = ttk.LabelFrame(self.root, text="Output Video", padding=10)
        output_frame.pack(pady=5, padx=10, fill='x')
        
        self.output_label = ttk.Label(output_frame, text="No output path set", foreground="gray")
        self.output_label.pack(side='left', fill='x', expand=True, padx=5)
        
        ttk.Button(output_frame, text="Browse", command=self._select_output).pack(side='right', padx=5)
        ttk.Button(output_frame, text="Auto", command=self._set_auto_output).pack(side='right', padx=2)
        
        # Parameters section
        params_frame = ttk.LabelFrame(self.root, text="Processing Parameters", padding=10)
        params_frame.pack(pady=5, padx=10, fill='both', expand=True)
        
        # Mode selection
        mode_frame = ttk.Frame(params_frame)
        mode_frame.pack(fill='x', pady=5)
        ttk.Label(mode_frame, text="Mode:").pack(side='left', padx=5)
        self.mode_var = tk.StringVar(value="blur")
        ttk.Radiobutton(mode_frame, text="Blur Background", variable=self.mode_var, value="blur").pack(side='left', padx=5)
        ttk.Radiobutton(mode_frame, text="Crop", variable=self.mode_var, value="crop").pack(side='left', padx=5)
        
        # Variance threshold
        var_frame = ttk.Frame(params_frame)
        var_frame.pack(fill='x', pady=5)
        ttk.Label(var_frame, text="Variance Threshold (lower=more sensitive):").pack(side='left', padx=5)
        self.variance_var = tk.DoubleVar(value=18.0)
        self.variance_scale = ttk.Scale(var_frame, from_=5.0, to=50.0, variable=self.variance_var, orient='horizontal')
        self.variance_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.variance_label = ttk.Label(var_frame, text="18.0", width=5)
        self.variance_label.pack(side='left', padx=5)
        self.variance_scale.configure(command=lambda v: self.variance_label.configure(text=f"{float(v):.1f}"))
        
        # Brightness cutoff
        bright_frame = ttk.Frame(params_frame)
        bright_frame.pack(fill='x', pady=5)
        ttk.Label(bright_frame, text="Brightness Cutoff (lower=darker bars):").pack(side='left', padx=5)
        self.brightness_var = tk.DoubleVar(value=220.0)
        self.brightness_scale = ttk.Scale(bright_frame, from_=50.0, to=255.0, variable=self.brightness_var, orient='horizontal')
        self.brightness_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.brightness_label = ttk.Label(bright_frame, text="220.0", width=5)
        self.brightness_label.pack(side='left', padx=5)
        self.brightness_scale.configure(command=lambda v: self.brightness_label.configure(text=f"{float(v):.0f}"))
        
        # Edge sensitivity
        edge_frame = ttk.Frame(params_frame)
        edge_frame.pack(fill='x', pady=5)
        ttk.Label(edge_frame, text="Edge Sensitivity (colored borders):").pack(side='left', padx=5)
        self.edge_var = tk.DoubleVar(value=0.3)
        self.edge_scale = ttk.Scale(edge_frame, from_=0.1, to=0.8, variable=self.edge_var, orient='horizontal')
        self.edge_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.edge_label = ttk.Label(edge_frame, text="0.3", width=5)
        self.edge_label.pack(side='left', padx=5)
        self.edge_scale.configure(command=lambda v: self.edge_label.configure(text=f"{float(v):.2f}"))
        
        # Blur size
        blur_frame = ttk.Frame(params_frame)
        blur_frame.pack(fill='x', pady=5)
        ttk.Label(blur_frame, text="Blur Size (blur mode only):").pack(side='left', padx=5)
        self.blur_var = tk.IntVar(value=31)
        self.blur_scale = ttk.Scale(blur_frame, from_=5, to=99, variable=self.blur_var, orient='horizontal')
        self.blur_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.blur_label = ttk.Label(blur_frame, text="31", width=5)
        self.blur_label.pack(side='left', padx=5)
        self.blur_scale.configure(command=lambda v: self.blur_label.configure(text=f"{int(float(v))}"))
        
        # Sample every
        sample_frame = ttk.Frame(params_frame)
        sample_frame.pack(fill='x', pady=5)
        ttk.Label(sample_frame, text="Sample Every N Frames:").pack(side='left', padx=5)
        self.sample_var = tk.IntVar(value=10)
        sample_spinbox = ttk.Spinbox(sample_frame, from_=1, to=100, textvariable=self.sample_var, width=5)
        sample_spinbox.pack(side='left', padx=5)
        
        # Spike filter window
        spike_frame = ttk.Frame(params_frame)
        spike_frame.pack(fill='x', pady=5)
        ttk.Label(spike_frame, text="Spike Filter Window:").pack(side='left', padx=5)
        self.spike_var = tk.IntVar(value=5)
        spike_spinbox = ttk.Spinbox(spike_frame, from_=1, to=20, textvariable=self.spike_var, width=5)
        spike_spinbox.pack(side='left', padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.pack(pady=5, padx=10, fill='x')
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(fill='x', padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10, fill='x', padx=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Video", command=self._process_video)
        self.process_button.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(button_frame, text="Open Output Folder", command=self._open_output_folder).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side='left', padx=5)
    
    def _select_input(self):
        file = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        if file:
            self.input_video = file
            self.input_label.configure(text=os.path.basename(file), foreground="black")
            self._set_auto_output()
    
    def _clear_input(self):
        self.input_video = None
        self.input_label.configure(text="No video selected", foreground="gray")
    
    def _select_output(self):
        file = filedialog.asksaveasfilename(
            title="Save cleaned video as",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if file:
            self.output_video = file
            self.output_label.configure(text=os.path.basename(file), foreground="black")
    
    def _set_auto_output(self):
        if self.input_video:
            base, ext = os.path.splitext(self.input_video)
            self.output_video = f"{base}_cleaned.mp4"
            self.output_label.configure(text=os.path.basename(self.output_video), foreground="black")
    
    def _open_output_folder(self):
        if self.output_video and os.path.exists(os.path.dirname(self.output_video)):
            folder = os.path.dirname(self.output_video)
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
    
    def _process_video(self):
        if not self.input_video:
            messagebox.showerror("Error", "Please select an input video")
            return
        
        if not self.output_video:
            messagebox.showerror("Error", "Please set an output path")
            return
        
        self.processing = True
        self.process_button.configure(state='disabled')
        self.progress_label.configure(text="Processing...")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self._process_thread)
        thread.daemon = True
        thread.start()
    
    def _process_thread(self):
        try:
            process_video(
                self.input_video,
                self.output_video,
                mode=self.mode_var.get(),
                variance_threshold=self.variance_var.get(),
                brightness_cutoff=self.brightness_var.get(),
                sample_every=self.sample_var.get(),
                blur_size=self.blur_var.get(),
                edge_sensitivity=self.edge_var.get(),
                spike_filter_window=self.spike_var.get(),
            )
            self.queue.put(("success", "Processing complete!"))
        except Exception as e:
            self.queue.put(("error", str(e)))
    
    def _start_queue_check(self):
        try:
            while not self.queue.empty():
                msg_type, msg = self.queue.get_nowait()
                if msg_type == "success":
                    self.progress_label.configure(text=msg)
                    self.progress_var.set(100)
                    messagebox.showinfo("Success", msg)
                    self.processing = False
                    self.process_button.configure(state='normal')
                elif msg_type == "error":
                    self.progress_label.configure(text=f"Error: {msg}")
                    messagebox.showerror("Error", msg)
                    self.processing = False
                    self.process_button.configure(state='normal')
        except:
            pass
        
        self.root.after(500, self._start_queue_check)


def main():
    root = tk.Tk()
    app = VideoCleanerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
