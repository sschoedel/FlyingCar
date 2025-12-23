"""
Base UI components for controller tuning.

Provides common functionality shared across all controller tuners:
- Plots (thrust, gimbal angles)
- Mode toggle (tune/play)
- Test scenario inputs
- History/duration slider
- Data logging
"""

import copy
import tkinter as tk
from tkinter import ttk
from collections import deque
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class BaseTunerPanel:
    """
    Base class for controller-specific tuner panels.

    Subclasses should override create_controls() to add their specific sliders.
    """

    def __init__(self, parent, weights, on_weights_changed):
        """
        Args:
            parent: Parent tkinter frame
            weights: Weights dict from JSON
            on_weights_changed: Callback(weights) when sliders change
        """
        self.parent = parent
        self.weights = weights
        self.on_weights_changed = on_weights_changed
        self.frame = ttk.Frame(parent)

    def create_controls(self):
        """Create controller-specific controls. Override in subclass."""
        pass

    def show(self):
        """Show this tuner panel."""
        self.frame.pack(fill='x', pady=5)

    def hide(self):
        """Hide this tuner panel."""
        self.frame.pack_forget()

    def update_weights(self, weights):
        """Update weights reference. Override if panel needs to refresh."""
        self.weights = weights

    def _create_slider(self, parent, label, from_, to, initial, resolution=1, command=None):
        """Helper to create a labeled slider with value display."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2)

        ttk.Label(frame, text=label, width=8).pack(side='left')

        var = tk.DoubleVar(value=initial)
        scale = ttk.Scale(frame, from_=from_, to=to, variable=var, orient='horizontal',
                          command=lambda v: self._on_scale_change(var, value_label, resolution, command))
        scale.pack(side='left', fill='x', expand=True, padx=5)

        value_label = ttk.Label(frame, text=f"{initial:.{self._get_decimals(resolution)}f}", width=8)
        value_label.pack(side='left')

        return var, frame

    def _get_decimals(self, resolution):
        """Get number of decimal places from resolution."""
        if resolution >= 1:
            return 0
        elif resolution >= 0.1:
            return 1
        elif resolution >= 0.01:
            return 2
        else:
            return 3

    def _on_scale_change(self, var, value_label, resolution, command):
        """Handle scale change, snap to resolution."""
        val = var.get()
        snapped = round(val / resolution) * resolution
        decimals = self._get_decimals(resolution)
        value_label.config(text=f"{snapped:.{decimals}f}")
        if command:
            command(snapped)

    def _update_weight(self, category, subcategory, key, value):
        """Update a weight value and notify callback."""
        if subcategory:
            if category not in self.weights:
                self.weights[category] = {}
            if subcategory not in self.weights[category]:
                self.weights[category][subcategory] = {}
            self.weights[category][subcategory][key] = value
        else:
            if category not in self.weights:
                self.weights[category] = {}
            self.weights[category][key] = value
        self.on_weights_changed(self.weights)


class BaseControllerUI:
    """
    Base UI class with common functionality for all controllers.

    Provides:
    - Mode toggle (tune/play)
    - Test scenario inputs
    - Plots (thrust, gimbal angles)
    - History/duration slider
    - Data logging
    """

    def __init__(self, root, weights, on_reset_sim, on_save, on_new_target):
        """
        Args:
            root: Tkinter root or parent frame
            weights: Initial weights dict from JSON
            on_reset_sim: Callback() to reset simulation
            on_save: Callback(weights) to save weights to file
            on_new_target: Callback() to set new random target (play mode)
        """
        self.root = root
        self.weights = copy.deepcopy(weights)
        self.on_reset_sim = on_reset_sim
        self.on_save = on_save
        self.on_new_target = on_new_target

        # Mode: "tune" or "play"
        self.mode = "tune"

        # Data buffers for plotting
        self.time_window = 5.0
        self.dt = 0.001  # Simulation timestep
        self.times = deque()
        self.thrusts = [deque() for _ in range(4)]
        self.gimbal_angles = [deque() for _ in range(8)]
        self.gimbal_cmd_angles = [deque() for _ in range(8)]
        self._update_buffer_size()
        self.update_interval = 50
        self.step_count = 0

        # Auto-pause after reset
        self.plot_paused = False
        self.pause_after_reset = 0.5  # seconds

        # State logging for tune mode
        self.logged_data = {
            'time': [],
            'position': [],
            'orientation': [],
            'gimbal_pos': [],
            'linear_vel': [],
            'angular_vel': [],
            'gimbal_vel': [],
            'thrust_cmd': [],
            'gimbal_cmd': [],
            'target_pos': [],
            'target_yaw': [],
        }

        # Track tune-mode widgets for visibility toggle
        self.tune_mode_widgets = []

        # Main frame for this UI
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill='both', expand=True)

    def create_common_controls(self, controls_frame):
        """Create common controls (scenario, actions). Call from subclass."""
        scenario = self.weights.get("test_scenario", {})

        # Actions frame
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions")
        actions_frame.pack(side='left', fill='both', padx=5)
        ttk.Button(actions_frame, text="Reset Sim", command=self._on_reset_click).pack(fill='x', padx=5, pady=2)
        ttk.Button(actions_frame, text="Save JSON", command=self._on_save_click).pack(fill='x', padx=5, pady=2)
        ttk.Button(actions_frame, text="Save Data", command=self._on_save_data_click).pack(fill='x', padx=5, pady=2)

        return actions_frame

    def create_scenario_controls(self, parent):
        """Create test scenario controls."""
        scenario = self.weights.get("test_scenario", {})
        start_pos = scenario.get("start_pos", [0, 0, 2])
        target_pos = scenario.get("target_pos", [2, 1, 3])

        scenario_frame = ttk.LabelFrame(parent, text="Test Scenario")
        scenario_frame.pack(side='left', fill='both', expand=True, padx=5)

        # Start position
        start_frame = ttk.Frame(scenario_frame)
        start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(start_frame, text="Start:").pack(side='left')
        self.start_x, _ = self._create_entry(start_frame, "X", start_pos[0],
                                              lambda v: self._update_scenario("start_pos", 0, v))
        self.start_y, _ = self._create_entry(start_frame, "Y", start_pos[1],
                                              lambda v: self._update_scenario("start_pos", 1, v))
        self.start_z, _ = self._create_entry(start_frame, "Z", start_pos[2],
                                              lambda v: self._update_scenario("start_pos", 2, v))
        self.start_yaw, _ = self._create_slider(scenario_frame, "Start Yaw", -180, 180,
                                                 scenario.get("start_yaw", 0), 1,
                                                 lambda v: self._update_scenario("start_yaw", None, v))

        # Target position
        target_frame = ttk.Frame(scenario_frame)
        target_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(target_frame, text="Target:").pack(side='left')
        self.target_x, _ = self._create_entry(target_frame, "X", target_pos[0],
                                               lambda v: self._update_scenario("target_pos", 0, v))
        self.target_y, _ = self._create_entry(target_frame, "Y", target_pos[1],
                                               lambda v: self._update_scenario("target_pos", 1, v))
        self.target_z, _ = self._create_entry(target_frame, "Z", target_pos[2],
                                               lambda v: self._update_scenario("target_pos", 2, v))
        self.target_yaw, _ = self._create_slider(scenario_frame, "Target Yaw", -180, 180,
                                                  scenario.get("target_yaw", 0), 1,
                                                  lambda v: self._update_scenario("target_yaw", None, v))

        return scenario_frame

    def create_mode_slider(self, parent):
        """Create the appropriate slider for current mode."""
        self.slider_frame = ttk.Frame(parent)
        self.slider_frame.pack(fill='x', padx=10, pady=5)
        self._recreate_mode_slider()

    def _recreate_mode_slider(self):
        """Recreate slider for current mode."""
        for widget in self.slider_frame.winfo_children():
            widget.destroy()

        if self.mode == "tune":
            self.slider_var, self.slider_widget = self._create_slider(
                self.slider_frame, "Duration (s)", 0.01, 2.0, self.pause_after_reset, 0.01,
                self._on_duration_change)
        else:
            self.slider_var, self.slider_widget = self._create_slider(
                self.slider_frame, "History (s)", 1, 60, self.time_window, 0.5,
                self._on_history_change)

    def create_plots(self, parent):
        """Create matplotlib plots embedded in tkinter."""
        self.fig = Figure(figsize=(8, 4))

        # Thrust plot
        self.ax_thrust = self.fig.add_subplot(2, 1, 1)
        self.ax_thrust.set_ylabel('Thrust (N)')
        self.ax_thrust.set_title('Thruster Control Inputs')
        self.ax_thrust.grid(True, alpha=0.3)
        self.ax_thrust.set_ylim(0, 1100)
        self.ax_thrust.tick_params(labelbottom=False)

        self.thrust_labels = ["FR", "FL", "RR", "RL"]
        self.colors = ['red', 'green', 'blue', 'orange']
        self.thrust_lines = []
        for label, color in zip(self.thrust_labels, self.colors):
            line, = self.ax_thrust.plot([], [], label=label, color=color, linewidth=1.5)
            self.thrust_lines.append(line)
        self.ax_thrust.legend(loc='upper right')

        # Gimbal angles plot
        self.ax_gimbal_plot = self.fig.add_subplot(2, 1, 2)
        self.ax_gimbal_plot.set_xlabel('Time (s)')
        self.ax_gimbal_plot.set_ylabel('Angle (deg)')
        self.ax_gimbal_plot.set_title('Gimbal Joint Angles')
        self.ax_gimbal_plot.grid(True, alpha=0.3)
        self.ax_gimbal_plot.set_ylim(-75, 75)

        self.gimbal_labels = ["FR_pitch", "FR_roll", "FL_pitch", "FL_roll",
                              "RR_pitch", "RR_roll", "RL_pitch", "RL_roll"]
        self.gimbal_lines = []
        for i, label in enumerate(self.gimbal_labels):
            color = self.colors[i // 2]
            line, = self.ax_gimbal_plot.plot([], [], label=label, color=color,
                                              linewidth=1.5, linestyle='-')
            self.gimbal_lines.append(line)

        self.gimbal_cmd_lines = []
        for i, label in enumerate(self.gimbal_labels):
            color = self.colors[i // 2]
            line, = self.ax_gimbal_plot.plot([], [], color=color,
                                              linewidth=1.0, linestyle='--', alpha=0.7)
            self.gimbal_cmd_lines.append(line)
        self.ax_gimbal_plot.legend(loc='upper right', ncol=4, fontsize=8)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)

    def _create_slider(self, parent, label, from_, to, initial, resolution=1, command=None):
        """Helper to create a labeled slider with value display."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2)

        ttk.Label(frame, text=label, width=12).pack(side='left')

        var = tk.DoubleVar(value=initial)
        scale = ttk.Scale(frame, from_=from_, to=to, variable=var, orient='horizontal',
                          command=lambda v: self._on_scale_change(var, value_label, resolution, command))
        scale.pack(side='left', fill='x', expand=True, padx=5)

        value_label = ttk.Label(frame, text=f"{initial:.{self._get_decimals(resolution)}f}", width=8)
        value_label.pack(side='left')

        return var, frame

    def _create_entry(self, parent, label, initial, command=None):
        """Helper to create a labeled entry field."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2)

        ttk.Label(frame, text=label, width=8).pack(side='left')

        var = tk.StringVar(value=f"{initial:.1f}")
        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.pack(side='left', padx=5)
        entry.bind('<Return>', lambda e: command(var.get()) if command else None)
        entry.bind('<FocusOut>', lambda e: command(var.get()) if command else None)

        return var, frame

    def _get_decimals(self, resolution):
        """Get number of decimal places from resolution."""
        if resolution >= 1:
            return 0
        elif resolution >= 0.1:
            return 1
        elif resolution >= 0.01:
            return 2
        else:
            return 3

    def _on_scale_change(self, var, value_label, resolution, command):
        """Handle scale change, snap to resolution."""
        val = var.get()
        snapped = round(val / resolution) * resolution
        decimals = self._get_decimals(resolution)
        value_label.config(text=f"{snapped:.{decimals}f}")
        if command:
            command(snapped)

    def _update_scenario(self, key, index, value):
        """Update a scenario value."""
        try:
            val = float(value)
        except ValueError:
            return
        if "test_scenario" not in self.weights:
            self.weights["test_scenario"] = {}
        if index is not None:
            if key not in self.weights["test_scenario"]:
                self.weights["test_scenario"][key] = [0, 0, 0]
            self.weights["test_scenario"][key][index] = val
        else:
            self.weights["test_scenario"][key] = val

    def _on_duration_change(self, value):
        """Handle duration slider change (tune mode)."""
        self.pause_after_reset = value
        self.time_window = value + 0.1
        self._update_buffer_size()

    def _on_history_change(self, value):
        """Handle history slider change (play mode)."""
        self.time_window = value
        self._update_buffer_size()

    def _update_buffer_size(self):
        """Resize data buffers to match current time window."""
        max_points = int(self.time_window / self.dt) + 100

        def resize_deque(old_deque, new_max):
            return deque(old_deque, maxlen=new_max)

        self.times = resize_deque(self.times, max_points)
        self.thrusts = [resize_deque(d, max_points) for d in self.thrusts]
        self.gimbal_angles = [resize_deque(d, max_points) for d in self.gimbal_angles]
        self.gimbal_cmd_angles = [resize_deque(d, max_points) for d in self.gimbal_cmd_angles]

    def _on_reset_click(self):
        """Handle reset button click."""
        self.clear_plot()
        self.on_reset_sim()

    def _on_save_click(self):
        """Handle save button click."""
        self.on_save(self.weights)
        print("Weights saved to lqr_weights.json")

    def _on_save_data_click(self):
        """Save logged state data to npz file."""
        if not self.logged_data['time']:
            print("No data to save. Run a test first.")
            return

        data_to_save = {}
        for key, value in self.logged_data.items():
            if value:
                data_to_save[key] = np.array(value)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_data_{timestamp}.npz"

        np.savez(filename, **data_to_save)
        print(f"Data saved to {filename} ({len(self.logged_data['time'])} samples)")

    def set_mode(self, mode):
        """Set the current mode (tune/play)."""
        self.mode = mode
        self._recreate_mode_slider()
        self._update_buffer_size()
        self.clear_plot()

    def get_scenario(self):
        """Get current test scenario settings."""
        return self.weights.get("test_scenario", {})

    def clear_plot(self):
        """Clear all plot data."""
        self.times.clear()
        for t in self.thrusts:
            t.clear()
        for g in self.gimbal_angles:
            g.clear()
        for g in self.gimbal_cmd_angles:
            g.clear()
        for line in self.thrust_lines:
            line.set_data([], [])
        for line in self.gimbal_lines:
            line.set_data([], [])
        for line in self.gimbal_cmd_lines:
            line.set_data([], [])
        self.ax_thrust.set_xlim(0, self.time_window)
        self.ax_gimbal_plot.set_xlim(0, self.time_window)
        self.step_count = 0
        self.plot_paused = False

        for key in self.logged_data:
            self.logged_data[key] = []

        self.canvas.draw()

    def update_plot(self, t, ctrl, gimbal_angles, gimbal_cmd, qpos=None, qvel=None, target_pos=None, target_yaw=None):
        """Add new data point and update plot periodically."""
        if self.plot_paused:
            return

        self.times.append(t)
        for i in range(4):
            self.thrusts[i].append(ctrl[i])
        for i in range(8):
            self.gimbal_angles[i].append(np.degrees(gimbal_angles[i]))
            self.gimbal_cmd_angles[i].append(np.degrees(gimbal_cmd[i]))

        if self.mode == "tune" and qpos is not None and qvel is not None:
            self.logged_data['time'].append(t)
            self.logged_data['position'].append(qpos[0:3].copy())
            self.logged_data['orientation'].append(qpos[3:7].copy())
            self.logged_data['gimbal_pos'].append(qpos[7:15].copy())
            self.logged_data['linear_vel'].append(qvel[0:3].copy())
            self.logged_data['angular_vel'].append(qvel[3:6].copy())
            self.logged_data['gimbal_vel'].append(qvel[6:14].copy())
            self.logged_data['thrust_cmd'].append(ctrl[0:4].copy())
            self.logged_data['gimbal_cmd'].append(ctrl[4:12].copy())
            if target_pos is not None:
                self.logged_data['target_pos'].append(target_pos.copy())
            if target_yaw is not None:
                self.logged_data['target_yaw'].append(target_yaw)

        if self.mode == "tune" and t >= self.pause_after_reset:
            self.plot_paused = True
            self._redraw_plot()
            return

        self.step_count += 1
        if self.step_count % self.update_interval == 0:
            self._redraw_plot()

    def _redraw_plot(self):
        """Redraw both plots."""
        if len(self.times) < 2:
            return

        times_arr = np.array(self.times)

        for i, line in enumerate(self.thrust_lines):
            line.set_data(times_arr, np.array(self.thrusts[i]))

        for i, line in enumerate(self.gimbal_lines):
            line.set_data(times_arr, np.array(self.gimbal_angles[i]))

        for i, line in enumerate(self.gimbal_cmd_lines):
            line.set_data(times_arr, np.array(self.gimbal_cmd_angles[i]))

        t_max = times_arr[-1]
        t_min = max(0, t_max - self.time_window)
        self.ax_thrust.set_xlim(t_min, t_max + 0.1)
        self.ax_gimbal_plot.set_xlim(t_min, t_max + 0.1)

        self.canvas.draw_idle()

    def update(self):
        """Process tkinter events (call from main loop)."""
        pass  # Handled by root.update() in main

    def close(self):
        """Clean up resources."""
        pass
