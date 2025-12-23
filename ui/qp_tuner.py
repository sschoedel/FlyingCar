"""
QP-specific tuner panel.

Provides sliders for Q and R weight matrices plus QP-specific parameters like horizon.
"""

import tkinter as tk
from tkinter import ttk

from .base_ui import BaseTunerPanel


class QPTunerPanel(BaseTunerPanel):
    """
    QP-specific tuner panel with Q/R weight sliders and horizon control.
    """

    def __init__(self, parent, weights, on_weights_changed, on_horizon_changed=None):
        """
        Args:
            parent: Parent tkinter frame
            weights: Weights dict from JSON
            on_weights_changed: Callback(weights) when sliders change
            on_horizon_changed: Callback(horizon) when horizon slider changes
        """
        self.on_horizon_changed = on_horizon_changed
        self.horizon = 10  # Default horizon
        super().__init__(parent, weights, on_weights_changed)
        self.create_controls()

    def create_controls(self):
        """Create QP-specific controls (Q/R weights + horizon)."""
        q = self.weights.get("Q", {})
        r = self.weights.get("R", {})

        # Row 0: QP-specific parameters
        row0 = ttk.Frame(self.frame)
        row0.pack(fill='x', pady=5)

        qp_frame = ttk.LabelFrame(row0, text="QP Parameters")
        qp_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.horizon_var, _ = self._create_slider(qp_frame, "Horizon", 1, 50, self.horizon, 1,
                                                   self._on_horizon_change)

        # Row 1: Q Position, Q Orientation, R Control
        row1 = ttk.Frame(self.frame)
        row1.pack(fill='x', pady=5)

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

        # Row 2: Q Velocity, Q Angular Vel, Q Gimbal
        row2 = ttk.Frame(self.frame)
        row2.pack(fill='x', pady=5)

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

    def _on_horizon_change(self, value):
        """Handle horizon slider change."""
        self.horizon = int(value)
        if self.on_horizon_changed:
            self.on_horizon_changed(self.horizon)

    def get_horizon(self):
        """Get current horizon value."""
        return self.horizon
