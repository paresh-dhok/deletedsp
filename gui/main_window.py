import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pyaudio
from config import GUIConfig


class MainWindow:
    def __init__(self, audio_processor):
        self.audio_processor = audio_processor
        self.config = GUIConfig()

        self.root = tk.Tk()
        self.root.title(self.config.WINDOW_TITLE)
        self.root.geometry(f"{self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}")

        self.is_processing = False
        self.is_recording = False
        self.update_job = None

        self.selected_input_device = None
        self.speech_prob_history = []
        self.processing_time_history = []

        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        device_frame = ttk.LabelFrame(main_frame, text="Audio Device Selection", padding="10")
        device_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, padx=5, pady=5)

        device_list = []
        device_indices = []
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            default_input = p.get_default_input_device_info()['index']
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_name = f"{i}: {info['name']}"
                    if i == default_input:
                        device_name += " (Default)"
                    device_list.append(device_name)
                    device_indices.append(i)
            p.terminate()
        except Exception as e:
            # Be permissive here: PyAudio may raise different exceptions depending on
            # platform and installation. Fall back to a placeholder device list.
            device_list = ["Unable to detect devices"]
            device_indices = [None]
            print(f"Error detecting audio devices: {e}")

        self.device_var = tk.StringVar(value=device_list[0] if device_list else "")
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var,
                                         values=device_list, state="readonly", width=40)
        self.device_combo.grid(row=0, column=1, padx=5, pady=5)
        self.device_indices = device_indices

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.calibrate_btn = ttk.Button(control_frame, text="Calibrate Noise", command=self._calibrate_noise)
        self.calibrate_btn.grid(row=0, column=0, padx=5, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start Processing", command=self._start_processing)
        self.start_btn.grid(row=0, column=1, padx=5, pady=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Processing", command=self._stop_processing, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=5, pady=5)

        self.record_btn = ttk.Button(control_frame, text="Start Recording", command=self._toggle_recording, state=tk.DISABLED)
        self.record_btn.grid(row=0, column=3, padx=5, pady=5)

        self.bypass_btn = ttk.Button(control_frame, text="Toggle Bypass", command=self._toggle_bypass)
        self.bypass_btn.grid(row=0, column=4, padx=5, pady=5)

        settings_frame = ttk.LabelFrame(main_frame, text="Processing Settings", padding="10")
        settings_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.spectral_var = tk.BooleanVar(value=True)
        spectral_check = ttk.Checkbutton(settings_frame, text="Spectral Subtraction",
                                        variable=self.spectral_var,
                                        command=self._update_settings)
        spectral_check.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

        self.wiener_var = tk.BooleanVar(value=True)
        wiener_check = ttk.Checkbutton(settings_frame, text="Wiener Filter",
                                      variable=self.wiener_var,
                                      command=self._update_settings)
        wiener_check.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        self.adaptive_var = tk.BooleanVar(value=True)
        adaptive_check = ttk.Checkbutton(settings_frame, text="Adaptive Noise Estimation",
                                        variable=self.adaptive_var,
                                        command=self._update_settings)
        adaptive_check.grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)

        viz_frame = ttk.LabelFrame(main_frame, text="Real-Time Visualization", padding="10")
        viz_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)

        self.ax1.set_title("Speech Probability")
        self.ax1.set_xlabel("Time")
        self.ax1.set_ylabel("Probability")
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_title("Processing Time")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Time (ms)")
        self.ax2.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_label = ttk.Label(status_frame, text="Status: Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.stats_label = ttk.Label(status_frame, text="Frames: 0 | Speech: 0 | Noise: 0")
        self.stats_label.pack(side=tk.LEFT, padx=20)

        self.recording_label = ttk.Label(status_frame, text="", foreground="red")
        self.recording_label.pack(side=tk.LEFT, padx=20)


    def _setup_layout(self):
        """Compatibility shim for older code that expects a separate layout setup step.

        The widget creation already configures grid/pack and places widgets. Keep this
        method so `__init__` can call it without raising AttributeError.
        """
        # Intentionally no-op; layout handled in _create_widgets
        return


    def _get_selected_input_device(self):
        selected_index = self.device_combo.current()
        if selected_index >= 0 and selected_index < len(self.device_indices):
            return self.device_indices[selected_index]
        return None

    def _calibrate_noise(self):
        self.status_label.config(text="Status: Calibrating... Please remain silent", foreground="orange")
        self.root.update()

        try:
            input_device = self._get_selected_input_device()
            if not self.is_processing:
                self.audio_processor.audio_capture.start(input_device_index=input_device)

            self.audio_processor.calibrate_noise(duration_seconds=2.0)

            if not self.is_processing:
                self.audio_processor.audio_capture.stop()

            self.status_label.config(text="Status: Calibration complete", foreground="green")
            messagebox.showinfo("Calibration", "Noise profile calibrated successfully!")

        except (OSError, tk.TclError) as e:
            # PyAudio-related errors typically surface as OSError on many platforms.
            self.status_label.config(text="Status: Calibration failed", foreground="red")
            messagebox.showerror("Error", f"Calibration failed: {str(e)}")

    def _start_processing(self):
        try:
            input_device = self._get_selected_input_device()
            # AudioProcessor.start_processing expects parameter name `input_device`
            # (not `input_device_index`). Pass using the correct name.
            self.audio_processor.start_processing(input_device=input_device)
            self.is_processing = True

            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL)
            self.calibrate_btn.config(state=tk.DISABLED)
            self.device_combo.config(state=tk.DISABLED)

            self.status_label.config(text="Status: Processing", foreground="green")

            self._update_display()

        except (OSError, tk.TclError) as e:
            # Catch OSError which PyAudio commonly raises for device errors.
            messagebox.showerror("Error", f"Failed to start processing: {str(e)}")
            self.status_label.config(text="Status: Error", foreground="red")

    def _stop_processing(self):
        if self.is_recording:
            self._toggle_recording()

        self.audio_processor.stop_processing()
        self.is_processing = False

        if self.update_job:
            self.root.after_cancel(self.update_job)
            self.update_job = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        self.calibrate_btn.config(state=tk.NORMAL)
        self.device_combo.config(state="readonly")

        self.status_label.config(text="Status: Stopped", foreground="orange")
        self.recording_label.config(text="")

    def _toggle_recording(self):
        if not self.is_processing:
            messagebox.showwarning("Warning", "Please start processing first")
            return

        if self.is_recording:
            filepath = self.audio_processor.stop_recording()
            self.is_recording = False
            self.record_btn.config(text="Start Recording")
            self.recording_label.config(text="")

            if filepath:
                messagebox.showinfo("Recording Saved", f"Saved: {filepath}")
        else:
            self.audio_processor.start_recording()
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.recording_label.config(text="REC", foreground="red")

    def _toggle_bypass(self):
        bypass_state = self.audio_processor.toggle_bypass()
        if bypass_state:
            self.status_label.config(text="Status: Bypass Mode (No Processing)", foreground="blue")
        else:
            self.status_label.config(text="Status: Processing", foreground="green")

    def _update_settings(self):
        self.audio_processor.use_spectral_subtraction = self.spectral_var.get()
        self.audio_processor.use_wiener_filter = self.wiener_var.get()
        self.audio_processor.use_adaptive_noise = self.adaptive_var.get()

    def _update_display(self):
        if not self.is_processing:
            return

        try:
            stats = self.audio_processor.get_stats()

            self.stats_label.config(
                text=f"Frames: {stats['frames_processed']} | "
                     f"Speech: {stats['speech_frames']} | "
                     f"Noise: {stats['noise_frames']}"
            )

            # Ensure we append a valid numeric value for plotting (no None/NaN)
            try:
                prob = float(stats.get('current_speech_prob', 0.0))
            except Exception:
                prob = 0.0
            # Clamp to [0,1]
            if prob != prob:  # NaN check
                prob = 0.0
            prob = max(0.0, min(1.0, prob))
            self.speech_prob_history.append(prob)
            if len(self.speech_prob_history) > self.config.PLOT_HISTORY:
                self.speech_prob_history.pop(0)

            self.processing_time_history.append(stats['processing_time_ms'])
            if len(self.processing_time_history) > self.config.PLOT_HISTORY:
                self.processing_time_history.pop(0)

            self.ax1.clear()
            self.ax1.set_title("Speech Probability")
            self.ax1.set_xlabel("Frame")
            self.ax1.set_ylabel("Probability")
            self.ax1.set_ylim(0, 1)
            self.ax1.grid(True, alpha=0.3)
            if len(self.speech_prob_history) > 0:
                # for small numbers of points, draw markers so the user can see activity
                if len(self.speech_prob_history) <= 2:
                    self.ax1.plot(self.speech_prob_history, 'bo-', linewidth=2, markersize=6)
                else:
                    self.ax1.plot(self.speech_prob_history, 'b-', linewidth=2)
                self.ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

            self.ax2.clear()
            self.ax2.set_title("Processing Time")
            self.ax2.set_xlabel("Frame")
            self.ax2.set_ylabel("Time (ms)")
            self.ax2.grid(True, alpha=0.3)
            if len(self.processing_time_history) > 0:
                self.ax2.plot(self.processing_time_history, 'g-', linewidth=2)

            self.canvas.draw()

        except (tk.TclError, RuntimeError) as e:
            print(f"Error updating display: {e}")

        self.update_job = self.root.after(self.config.UPDATE_INTERVAL, self._update_display)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self):
        if self.is_recording:
            self._toggle_recording()

        if self.is_processing:
            self._stop_processing()

        self.audio_processor.cleanup()
        self.root.destroy()
