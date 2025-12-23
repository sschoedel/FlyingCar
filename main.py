import mujoco
import mujoco.viewer
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import copy

from LQR import FlyingCarLQR, load_lqr_weights, save_lqr_weights


def add_thrust_arrow(viewer, site_id, thrust, max_thrust, data):
    """Add an arrow showing thrust magnitude at a thruster site."""
    if thrust < 0.1:
        return

    # Get site position and orientation
    site_pos = data.site_xpos[site_id].copy()
    site_mat = data.site_xmat[site_id].reshape(3, 3)

    # Arrow points in -z direction of site (opposite thrust direction), scaled by magnitude
    scale = thrust / max_thrust * 2.0  # Max arrow length 1.0m
    arrow_dir = site_mat @ np.array([0, 0, -1])  # Local -z in world frame

    # Arrow starts at site and points in thrust direction
    arrow_start = site_pos
    arrow_end = site_pos + arrow_dir * scale

    # Add arrow to viewer
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([1.0, 0.5, 0.0, 0.8]),  # Orange color
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.02,  # Arrow width
        arrow_start,
        arrow_end,
    )
    viewer.user_scn.ngeom += 1


def add_target_marker(viewer, target_pos, target_yaw):
    """Add a sphere marker at the target position with an arrow showing yaw."""
    # Sphere at target position
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.1, 0, 0]),  # size
        target_pos,
        np.eye(3).flatten(),
        np.array([0.0, 1.0, 0.0, 0.5]),  # Green, semi-transparent
    )
    viewer.user_scn.ngeom += 1

    # Arrow showing yaw direction
    arrow_length = 0.5
    arrow_dir = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
    arrow_start = target_pos
    arrow_end = target_pos + arrow_dir * arrow_length

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([0.0, 1.0, 0.0, 0.8]),  # Green color
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.03,  # Arrow width
        arrow_start,
        arrow_end,
    )
    viewer.user_scn.ngeom += 1


def random_target():
    """Generate a random target position and yaw orientation."""
    pos = np.array([
        np.random.uniform(-5, 5),   # x
        np.random.uniform(-5, 5),   # y
        np.random.uniform(1, 5),    # z (keep above ground)
    ])
    yaw = np.random.uniform(-np.pi, np.pi)  # Random yaw angle
    return pos, yaw


def yaw_to_quat(yaw):
    """Convert yaw angle to quaternion (w, x, y, z)."""
    return np.array([
        np.cos(yaw / 2),
        0,
        0,
        np.sin(yaw / 2)
    ])


class LQRTunerUI:
    """Tkinter-based UI for tuning LQR weights and test scenarios with real-time visualization."""

    def __init__(self, weights, on_weights_changed, on_reset_sim, on_save, on_new_target):
        """
        Args:
            weights: Initial weights dict from JSON
            on_weights_changed: Callback(weights) when sliders change
            on_reset_sim: Callback() to reset simulation
            on_save: Callback(weights) to save weights to file
            on_new_target: Callback() to set new random target (play mode)
        """
        self.weights = copy.deepcopy(weights)
        self.on_weights_changed = on_weights_changed
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

        # Create tkinter window
        self.root = tk.Tk()
        self.root.title("LQR Tuner")
        self.root.geometry("1200x900")

        # Track tune-mode widgets for visibility toggle
        self.tune_mode_widgets = []

        # Create UI
        self._create_controls()
        self._create_plots()

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

    def _create_controls(self):
        """Create all control widgets."""
        q = self.weights.get("Q", {})
        r = self.weights.get("R", {})
        scenario = self.weights.get("test_scenario", {})

        # Top frame for mode toggle
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)

        self.mode_btn = ttk.Button(top_frame, text="Switch to Play Mode",
                                    command=self._on_mode_toggle)
        self.mode_btn.pack(side='left', padx=5)

        # Main controls frame
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(fill='x', padx=10, pady=5)

        # Row 1: Q Position, Q Orientation, R Control, Actions
        row1 = ttk.Frame(self.controls_frame)
        row1.pack(fill='x', pady=5)
        self.tune_mode_widgets.append(row1)

        # Q: Position
        q_pos_frame = ttk.LabelFrame(row1, text="Q: Position")
        q_pos_frame.pack(side='left', fill='both', expand=True, padx=5)
        pos = q.get("position", {})
        self.q_pos_x, _ = self._create_slider(q_pos_frame, "x", 1, 500, pos.get("x", 100), 1,
                                               lambda v: self._update_weight("Q", "position", "x", v))
        self.q_pos_y, _ = self._create_slider(q_pos_frame, "y", 1, 500, pos.get("y", 100), 1,
                                               lambda v: self._update_weight("Q", "position", "y", v))
        self.q_pos_z, _ = self._create_slider(q_pos_frame, "z", 1, 500, pos.get("z", 200), 1,
                                               lambda v: self._update_weight("Q", "position", "z", v))

        # Q: Orientation
        q_ori_frame = ttk.LabelFrame(row1, text="Q: Orientation")
        q_ori_frame.pack(side='left', fill='both', expand=True, padx=5)
        ori = q.get("orientation", {})
        self.q_ori_roll, _ = self._create_slider(q_ori_frame, "roll", 1, 200, ori.get("roll", 50), 1,
                                                  lambda v: self._update_weight("Q", "orientation", "roll", v))
        self.q_ori_pitch, _ = self._create_slider(q_ori_frame, "pitch", 1, 200, ori.get("pitch", 50), 1,
                                                   lambda v: self._update_weight("Q", "orientation", "pitch", v))
        self.q_ori_yaw, _ = self._create_slider(q_ori_frame, "yaw", 1, 200, ori.get("yaw", 50), 1,
                                                 lambda v: self._update_weight("Q", "orientation", "yaw", v))

        # R: Control Cost
        r_frame = ttk.LabelFrame(row1, text="R: Control Cost")
        r_frame.pack(side='left', fill='both', expand=True, padx=5)
        self.r_thrust, _ = self._create_slider(r_frame, "thrust", 0.001, 1.0, r.get("thrust", 0.01), 0.001,
                                                lambda v: self._update_weight("R", None, "thrust", v))
        self.r_gimbal, _ = self._create_slider(r_frame, "gimbal", 0.01, 10.0, r.get("gimbal", 0.1), 0.01,
                                                lambda v: self._update_weight("R", None, "gimbal", v))

        # Actions
        actions_frame = ttk.LabelFrame(row1, text="Actions")
        actions_frame.pack(side='left', fill='both', padx=5)
        ttk.Button(actions_frame, text="Reset Sim", command=self._on_reset_click).pack(fill='x', padx=5, pady=2)
        ttk.Button(actions_frame, text="Save JSON", command=self._on_save_click).pack(fill='x', padx=5, pady=2)

        # Row 2: Q Velocity, Q Angular Vel, Q Gimbal, Test Scenario
        row2 = ttk.Frame(self.controls_frame)
        row2.pack(fill='x', pady=5)
        self.tune_mode_widgets.append(row2)

        # Q: Velocity
        q_vel_frame = ttk.LabelFrame(row2, text="Q: Velocity")
        q_vel_frame.pack(side='left', fill='both', expand=True, padx=5)
        vel = q.get("velocity", {})
        self.q_vel_vx, _ = self._create_slider(q_vel_frame, "vx", 0.1, 100, vel.get("vx", 10), 0.1,
                                                lambda v: self._update_weight("Q", "velocity", "vx", v))
        self.q_vel_vy, _ = self._create_slider(q_vel_frame, "vy", 0.1, 100, vel.get("vy", 10), 0.1,
                                                lambda v: self._update_weight("Q", "velocity", "vy", v))
        self.q_vel_vz, _ = self._create_slider(q_vel_frame, "vz", 0.1, 100, vel.get("vz", 10), 0.1,
                                                lambda v: self._update_weight("Q", "velocity", "vz", v))

        # Q: Angular Velocity
        q_ang_frame = ttk.LabelFrame(row2, text="Q: Angular Velocity")
        q_ang_frame.pack(side='left', fill='both', expand=True, padx=5)
        ang = q.get("angular_velocity", {})
        self.q_ang_wx, _ = self._create_slider(q_ang_frame, "wx", 0.1, 50, ang.get("wx", 5), 0.1,
                                                lambda v: self._update_weight("Q", "angular_velocity", "wx", v))
        self.q_ang_wy, _ = self._create_slider(q_ang_frame, "wy", 0.1, 50, ang.get("wy", 5), 0.1,
                                                lambda v: self._update_weight("Q", "angular_velocity", "wy", v))
        self.q_ang_wz, _ = self._create_slider(q_ang_frame, "wz", 0.1, 50, ang.get("wz", 5), 0.1,
                                                lambda v: self._update_weight("Q", "angular_velocity", "wz", v))

        # Q: Gimbal
        q_gimbal_frame = ttk.LabelFrame(row2, text="Q: Gimbal")
        q_gimbal_frame.pack(side='left', fill='both', expand=True, padx=5)
        self.q_gimbal_pos, _ = self._create_slider(q_gimbal_frame, "pos", 0.01, 10.0, q.get("gimbal_pos", 1.0), 0.01,
                                                    lambda v: self._update_weight("Q", None, "gimbal_pos", v))
        self.q_gimbal_vel, _ = self._create_slider(q_gimbal_frame, "vel", 0.01, 5.0, q.get("gimbal_vel", 0.1), 0.01,
                                                    lambda v: self._update_weight("Q", None, "gimbal_vel", v))

        # Test Scenario
        scenario_frame = ttk.LabelFrame(row2, text="Test Scenario")
        scenario_frame.pack(side='left', fill='both', expand=True, padx=5)
        start_pos = scenario.get("start_pos", [0, 0, 2])
        target_pos = scenario.get("target_pos", [2, 1, 3])

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

        # History/duration slider (always visible, behavior changes by mode)
        self.slider_frame = ttk.Frame(self.root)
        self.slider_frame.pack(fill='x', padx=10, pady=5)
        self._create_mode_slider()

    def _create_mode_slider(self):
        """Create the appropriate slider for current mode."""
        # Clear existing slider
        for widget in self.slider_frame.winfo_children():
            widget.destroy()

        if self.mode == "tune":
            # Tune mode: duration before pause
            self.slider_var, self.slider_widget = self._create_slider(
                self.slider_frame, "Duration (s)", 0.01, 2.0, self.pause_after_reset, 0.01,
                self._on_duration_change)
        else:
            # Play mode: history length
            self.slider_var, self.slider_widget = self._create_slider(
                self.slider_frame, "History (s)", 1, 60, self.time_window, 0.5,
                self._on_history_change)

    def _on_duration_change(self, value):
        """Handle duration slider change (tune mode)."""
        self.pause_after_reset = value
        self.time_window = value + 0.1  # Slightly larger window to show full duration
        self._update_buffer_size()

    def _on_history_change(self, value):
        """Handle history slider change (play mode)."""
        self.time_window = value
        self._update_buffer_size()

    def _create_plots(self):
        """Create matplotlib plots embedded in tkinter."""
        # Create matplotlib figure - size will be determined by tkinter layout
        self.fig = Figure(figsize=(8, 4))

        # Thrust plot
        self.ax_thrust = self.fig.add_subplot(2, 1, 1)
        self.ax_thrust.set_ylabel('Thrust (N)')
        self.ax_thrust.set_title('Thruster Control Inputs')
        self.ax_thrust.grid(True, alpha=0.3)
        self.ax_thrust.set_ylim(0, 1100)
        self.ax_thrust.tick_params(labelbottom=False)

        # Create thrust plot lines
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

        # Create gimbal plot lines
        self.gimbal_labels = ["FR_pitch", "FR_roll", "FL_pitch", "FL_roll",
                              "RR_pitch", "RR_roll", "RL_pitch", "RL_roll"]
        self.gimbal_lines = []
        for i, label in enumerate(self.gimbal_labels):
            color = self.colors[i // 2]
            line, = self.ax_gimbal_plot.plot([], [], label=label, color=color,
                                              linewidth=1.5, linestyle='-')
            self.gimbal_lines.append(line)

        # Commanded angles (dashed)
        self.gimbal_cmd_lines = []
        for i, label in enumerate(self.gimbal_labels):
            color = self.colors[i // 2]
            line, = self.ax_gimbal_plot.plot([], [], color=color,
                                              linewidth=1.0, linestyle='--', alpha=0.7)
            self.gimbal_cmd_lines.append(line)
        self.ax_gimbal_plot.legend(loc='upper right', ncol=4, fontsize=8)

        self.fig.tight_layout()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)

    def _update_buffer_size(self):
        """Resize data buffers to match current time window."""
        max_points = int(self.time_window / self.dt) + 100  # Add small buffer

        # Resize deques while preserving recent data
        def resize_deque(old_deque, new_max):
            new_deque = deque(old_deque, maxlen=new_max)
            return new_deque

        self.times = resize_deque(self.times, max_points)
        self.thrusts = [resize_deque(d, max_points) for d in self.thrusts]
        self.gimbal_angles = [resize_deque(d, max_points) for d in self.gimbal_angles]
        self.gimbal_cmd_angles = [resize_deque(d, max_points) for d in self.gimbal_cmd_angles]

    def _update_weight(self, category, subcategory, key, value):
        """Update a weight value and notify callback."""
        if subcategory:
            self.weights[category][subcategory][key] = value
        else:
            self.weights[category][key] = value
        self.on_weights_changed(self.weights)

    def _update_scenario(self, key, index, value):
        """Update a scenario value."""
        try:
            val = float(value)
        except ValueError:
            return
        if index is not None:
            self.weights["test_scenario"][key][index] = val
        else:
            self.weights["test_scenario"][key] = val

    def _on_reset_click(self):
        """Handle reset button click."""
        self.clear_plot()
        self.on_reset_sim()

    def _on_save_click(self):
        """Handle save button click."""
        self.on_save(self.weights)
        print("Weights saved to lqr_weights.json")

    def _on_mode_toggle(self):
        """Toggle between tune and play modes."""
        if self.mode == "tune":
            self.mode = "play"
            self.mode_btn.config(text='Switch to Tune Mode')
            self.time_window = 5.0  # Reset to default play mode history
            self.on_new_target()
        else:
            self.mode = "tune"
            self.mode_btn.config(text='Switch to Play Mode')
            self.time_window = self.pause_after_reset + 0.1  # Match duration
            self.on_reset_sim()

        self._update_mode_visibility()
        self._create_mode_slider()  # Recreate slider for new mode
        self._update_buffer_size()
        self.clear_plot()
        print(f"Switched to {self.mode} mode")

    def _update_mode_visibility(self):
        """Show/hide UI elements based on current mode."""
        for widget in self.tune_mode_widgets:
            if self.mode == "tune":
                widget.pack(fill='x', pady=5)
            else:
                widget.pack_forget()

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
        self.canvas.draw()

    def update_plot(self, t, ctrl, gimbal_angles, gimbal_cmd):
        """Add new data point and update plot periodically."""
        if self.plot_paused:
            return

        self.times.append(t)
        for i in range(4):
            self.thrusts[i].append(ctrl[i])
        for i in range(8):
            self.gimbal_angles[i].append(np.degrees(gimbal_angles[i]))
            self.gimbal_cmd_angles[i].append(np.degrees(gimbal_cmd[i]))

        # Check if we should pause (only in tune mode)
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
        self.root.update()

    def close(self):
        """Close the UI window."""
        self.root.destroy()


class ThrustPlotter:
    """Real-time plotter for thrust control inputs using matplotlib."""

    def __init__(self, time_window=5.0, update_interval=50, dt=0.001):
        self.time_window = time_window
        self.update_interval = update_interval
        self.dt = dt  # Simulation timestep
        self.step_count = 0

        # Calculate max points based on time window and timestep
        self._update_max_points()

        # Data buffers
        self.times = deque(maxlen=self.max_points)
        self.thrusts = [deque(maxlen=self.max_points) for _ in range(4)]

        # Setup matplotlib interactive mode
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.canvas.manager.set_window_title('Thrust Inputs')

        # Make room for slider
        self.fig.subplots_adjust(bottom=0.2)

        # Create lines
        self.labels = ["FR", "FL", "RR", "RL"]
        self.colors = ['red', 'green', 'blue', 'orange']
        self.lines = []
        for i, (label, color) in enumerate(zip(self.labels, self.colors)):
            line, = self.ax.plot([], [], label=label, color=color, linewidth=1.5)
            self.lines.append(line)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Thrust (N)')
        self.ax.set_title('Thruster Control Inputs')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 550)

        # Add slider for time window
        from matplotlib.widgets import Slider
        slider_ax = self.fig.add_axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(
            slider_ax, 'History (s)',
            valmin=1.0, valmax=30.0, valinit=time_window, valstep=0.5
        )
        self.slider.on_changed(self._on_slider_change)

        self.fig.show()

    def _update_max_points(self):
        """Calculate max points based on time window."""
        self.max_points = int(self.time_window / self.dt) + 1

    def _on_slider_change(self, val):
        """Handle slider value change."""
        self.time_window = val
        self._update_max_points()

        # Resize deques - keep most recent data that fits
        old_times = list(self.times)
        old_thrusts = [list(t) for t in self.thrusts]

        self.times = deque(maxlen=self.max_points)
        self.thrusts = [deque(maxlen=self.max_points) for _ in range(4)]

        # Copy back data (deque will automatically keep only max_points)
        for t in old_times:
            self.times.append(t)
        for i, thrust_data in enumerate(old_thrusts):
            for val in thrust_data:
                self.thrusts[i].append(val)

    def update(self, t, ctrl):
        """Add new data point and update plot periodically."""
        self.times.append(t)
        for i in range(4):
            self.thrusts[i].append(ctrl[i])

        self.step_count += 1
        if self.step_count % self.update_interval == 0:
            self._redraw()

    def _redraw(self):
        """Redraw the plot with current data."""
        if len(self.times) < 2:
            return

        times_arr = np.array(self.times)
        for i, line in enumerate(self.lines):
            line.set_data(times_arr, np.array(self.thrusts[i]))

        # Sliding time window
        t_max = times_arr[-1]
        t_min = max(0, t_max - self.time_window)
        self.ax.set_xlim(t_min, t_max + 0.1)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)


def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("flying_car.xml")
    data = mujoco.MjData(model)

    # Get site IDs for thrusters
    thruster_sites = [
        model.site("thruster_fr").id,
        model.site("thruster_fl").id,
        model.site("thruster_rr").id,
        model.site("thruster_rl").id,
    ]

    # Get max thrust from actuator control range
    max_thrust = model.actuator_ctrlrange[0, 1]

    # Load LQR weights from JSON
    weights = load_lqr_weights("lqr_weights.json")

    # Initialize LQR controller with weights
    controller = FlyingCarLQR(model, weights)

    # Get initial scenario from weights
    scenario = weights.get("test_scenario", {})
    start_pos = np.array(scenario.get("start_pos", [0, 0, 2]))
    start_yaw = np.radians(scenario.get("start_yaw", 0))
    target_pos = np.array(scenario.get("target_pos", [2, 1, 3]))
    target_yaw = np.radians(scenario.get("target_yaw", 0))

    # Flags to trigger actions from main thread (needed because key_callback runs in different thread)
    reset_requested = False
    new_target_requested = False

    def reset_simulation():
        """Reset simulation to start state from current scenario."""
        nonlocal reset_requested
        reset_requested = True

    def on_weights_changed(new_weights):
        """Callback when LQR weights are changed via UI."""
        controller.update_weights(new_weights)
        print("LQR weights updated, controller recomputed")

    def on_save(weights_to_save):
        """Callback to save weights to JSON."""
        save_lqr_weights(weights_to_save, "lqr_weights.json")

    def on_new_target():
        """Set a new random target position and yaw (play mode)."""
        nonlocal target_pos, target_yaw
        target_pos, target_yaw = random_target()
        print(f"New target: pos={target_pos}, yaw={np.degrees(target_yaw):.1f}°")

    # Setup tuner UI
    tuner = LQRTunerUI(
        weights=weights,
        on_weights_changed=on_weights_changed,
        on_reset_sim=reset_simulation,
        on_save=on_save,
        on_new_target=on_new_target
    )

    def apply_reset(viewer):
        """Apply simulation reset using current scenario from tuner."""
        nonlocal target_pos, target_yaw, reset_requested

        scenario = tuner.get_scenario()
        start_pos = np.array(scenario.get("start_pos", [0, 0, 2]))
        start_yaw_deg = scenario.get("start_yaw", 0)
        target_pos = np.array(scenario.get("target_pos", [2, 1, 3]))
        target_yaw = np.radians(scenario.get("target_yaw", 0))

        # Reset MuJoCo state
        mujoco.mj_resetData(model, data)

        # Set start position
        data.qpos[0:3] = start_pos
        data.qpos[3:7] = yaw_to_quat(np.radians(start_yaw_deg))
        data.qpos[7:] = 0  # Gimbal joints at zero

        # Zero velocity
        data.qvel[:] = 0

        # Set hover control
        data.ctrl[:] = controller.hover_ctrl

        # Clear plot data
        tuner.clear_plot()

        reset_requested = False
        print(f"Simulation reset: start={start_pos}, target={target_pos}")

    # Keyboard callback for manual controls
    # Note: This runs in a different thread, so we only set flags here
    # and let the main loop handle tkinter operations
    def key_callback(keycode):
        nonlocal reset_requested, new_target_requested
        if keycode == ord(' '):  # Spacebar
            if tuner.mode == "tune":
                # Tune mode: reset simulation
                reset_requested = True
            else:
                # Play mode: set new random target
                new_target_requested = True
        elif keycode == ord('r'):  # R key - always reset
            reset_requested = True

    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set initial camera to view the grid
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 12.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -30.0
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        # Apply initial reset to set start state
        apply_reset(viewer)

        start_time = time.time()

        try:
            while viewer.is_running():
                # Check for reset request
                if reset_requested:
                    apply_reset(viewer)
                    start_time = time.time()

                # Check for new target request (play mode)
                if new_target_requested:
                    on_new_target()
                    new_target_requested = False

                # Compute control input
                target_quat = yaw_to_quat(target_yaw)
                data.ctrl[:] = controller.compute_control(data, target_pos, target_quat)

                # Step simulation
                mujoco.mj_step(model, data)

                # Update plots in tuner UI
                # Gimbal joint angles are qpos[7:15] (8 joints: pitch/roll for each thruster)
                # Commanded gimbal angles are ctrl[4:12]
                gimbal_angles = data.qpos[7:15]
                gimbal_cmd = data.ctrl[4:12]
                tuner.update_plot(data.time, data.ctrl[:4], gimbal_angles, gimbal_cmd)

                # Process tkinter events
                tuner.update()

                # Clear user geometry and add visualizations
                viewer.user_scn.ngeom = 0
                for i, site_id in enumerate(thruster_sites):
                    add_thrust_arrow(viewer, site_id, data.ctrl[i], max_thrust, data)
                add_target_marker(viewer, target_pos, target_yaw)

                # Wait until next timestep
                elapsed = time.time() - start_time
                sleep_time = data.time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Sync viewer
                viewer.sync()
        finally:
            tuner.close()


if __name__ == "__main__":
    main()
