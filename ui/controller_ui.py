"""
Main controller UI that manages controller and mode switching.

Provides:
- Controller selection dropdown (LQR, QP, etc.)
- Mode toggle (tune/play)
- Controller-specific tuner panels
- Common plots and scenario controls
"""

import copy
import tkinter as tk
from tkinter import ttk

from .base_ui import BaseControllerUI
from .lqr_tuner import LQRTunerPanel
from .qp_tuner import QPTunerPanel


class ControllerUI:
    """
    Main UI for controller selection, tuning, and visualization.

    Manages switching between different controllers and their tuner panels,
    while maintaining common functionality like plots and mode switching.
    """

    # Registry of available controllers and their tuner panels
    CONTROLLERS = {
        'LQR': {
            'tuner_class': LQRTunerPanel,
            'description': 'Linear Quadratic Regulator (linearized at hover)',
        },
        'QP': {
            'tuner_class': QPTunerPanel,
            'description': 'Quadratic Programming (re-linearizes each step)',
        },
    }

    def __init__(self, weights, on_controller_changed, on_weights_changed,
                 on_reset_sim, on_save, on_new_target):
        """
        Args:
            weights: Initial weights dict from JSON
            on_controller_changed: Callback(controller_type, horizon) when controller changes
            on_weights_changed: Callback(weights) when tuner sliders change
            on_reset_sim: Callback() to reset simulation
            on_save: Callback(weights) to save weights to file
            on_new_target: Callback() to set new random target (play mode)
        """
        self.weights = copy.deepcopy(weights)
        self.on_controller_changed = on_controller_changed
        self.on_weights_changed = on_weights_changed
        self.on_reset_sim = on_reset_sim
        self.on_save = on_save
        self.on_new_target = on_new_target

        # Current state
        self.current_controller = 'LQR'
        self.mode = 'tune'
        self.tuner_panel = None
        self.qp_horizon = 10

        # Create tkinter window
        self.root = tk.Tk()
        self.root.title("Flying Car Controller")
        self.root.geometry("1200x950")

        # Create base UI (handles plots, common controls)
        self.base_ui = BaseControllerUI(
            self.root,
            self.weights,
            on_reset_sim,
            on_save,
            on_new_target
        )

        # Build the full UI
        self._create_ui()

    def _create_ui(self):
        """Create the complete UI layout."""
        main_frame = self.base_ui.main_frame

        # Top bar: Controller selection + Mode toggle
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill='x', padx=10, pady=5)

        # Controller selection
        ctrl_frame = ttk.Frame(top_frame)
        ctrl_frame.pack(side='left', padx=5)

        ttk.Label(ctrl_frame, text="Controller:").pack(side='left', padx=5)

        self.controller_var = tk.StringVar(value=self.current_controller)
        self.controller_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.controller_var,
            values=list(self.CONTROLLERS.keys()),
            state='readonly',
            width=10
        )
        self.controller_combo.pack(side='left', padx=5)
        self.controller_combo.bind('<<ComboboxSelected>>', self._on_controller_select)

        # Controller description label
        self.controller_desc = ttk.Label(ctrl_frame, text=self.CONTROLLERS['LQR']['description'],
                                          foreground='gray')
        self.controller_desc.pack(side='left', padx=10)

        # Mode toggle button
        self.mode_btn = ttk.Button(top_frame, text="Switch to Play Mode",
                                    command=self._on_mode_toggle)
        self.mode_btn.pack(side='right', padx=5)

        # Tuner panel container (will hold controller-specific controls)
        self.tuner_container = ttk.Frame(main_frame)
        self.tuner_container.pack(fill='x', padx=10, pady=5)

        # Create initial tuner panel
        self._create_tuner_panel()

        # Common controls row (scenario + actions)
        common_frame = ttk.Frame(main_frame)
        common_frame.pack(fill='x', padx=10, pady=5)
        self.tune_mode_widgets = [common_frame]

        # Scenario controls
        self.base_ui.create_scenario_controls(common_frame)

        # Actions
        self.base_ui.create_common_controls(common_frame)

        # History/Duration slider
        self.base_ui.create_mode_slider(main_frame)

        # Plots
        self.base_ui.create_plots(main_frame)

    def _create_tuner_panel(self):
        """Create the tuner panel for the current controller."""
        # Remove existing tuner panel
        if self.tuner_panel:
            self.tuner_panel.hide()
            self.tuner_panel = None

        # Clear the container
        for widget in self.tuner_container.winfo_children():
            widget.destroy()

        # Get tuner class for current controller
        controller_info = self.CONTROLLERS.get(self.current_controller)
        if not controller_info:
            return

        tuner_class = controller_info.get('tuner_class')
        if not tuner_class:
            # No tuner for this controller - show message
            ttk.Label(self.tuner_container,
                      text=f"No tuner available for {self.current_controller}",
                      foreground='gray').pack(pady=10)
            return

        # Create the tuner panel
        if self.current_controller == 'QP':
            self.tuner_panel = tuner_class(
                self.tuner_container,
                self.weights,
                self._on_tuner_weights_changed,
                on_horizon_changed=self._on_horizon_changed
            )
        else:
            self.tuner_panel = tuner_class(
                self.tuner_container,
                self.weights,
                self._on_tuner_weights_changed
            )

        # Show the panel (visibility depends on mode)
        self._update_tuner_visibility()

    def _on_controller_select(self, event=None):
        """Handle controller selection change."""
        new_controller = self.controller_var.get()
        if new_controller == self.current_controller:
            return

        self.current_controller = new_controller

        # Update description
        desc = self.CONTROLLERS[new_controller].get('description', '')
        self.controller_desc.config(text=desc)

        # Recreate tuner panel
        self._create_tuner_panel()

        # Get horizon for QP controller
        horizon = None
        if new_controller == 'QP' and self.tuner_panel:
            horizon = self.tuner_panel.get_horizon()

        # Notify callback
        self.on_controller_changed(new_controller, horizon)

        # Reset simulation with new controller
        self.base_ui.clear_plot()
        self.on_reset_sim()

        print(f"Switched to {new_controller} controller")

    def _on_mode_toggle(self):
        """Toggle between tune and play modes."""
        if self.mode == "tune":
            self.mode = "play"
            self.mode_btn.config(text='Switch to Tune Mode')
            self.base_ui.time_window = 5.0
            self.on_new_target()
        else:
            self.mode = "tune"
            self.mode_btn.config(text='Switch to Play Mode')
            self.base_ui.time_window = self.base_ui.pause_after_reset + 0.1
            self.on_reset_sim()

        self.base_ui.set_mode(self.mode)
        self._update_tuner_visibility()

        print(f"Switched to {self.mode} mode")

    def _update_tuner_visibility(self):
        """Update visibility of tuner panel based on mode."""
        if self.mode == "tune":
            # Show tuner panel in tune mode
            if self.tuner_panel:
                self.tuner_panel.show()
            for widget in self.tune_mode_widgets:
                widget.pack(fill='x', padx=10, pady=5)
        else:
            # Hide tuner panel in play mode
            if self.tuner_panel:
                self.tuner_panel.hide()
            for widget in self.tune_mode_widgets:
                widget.pack_forget()

    def _on_tuner_weights_changed(self, weights):
        """Handle weight changes from tuner panel."""
        self.weights = weights
        self.base_ui.weights = weights
        self.on_weights_changed(weights)

    def _on_horizon_changed(self, horizon):
        """Handle horizon change from QP tuner."""
        self.qp_horizon = horizon
        # Notify controller to update horizon
        if self.current_controller == 'QP':
            self.on_controller_changed('QP', horizon)

    def get_current_controller(self):
        """Get the currently selected controller type."""
        return self.current_controller

    def get_scenario(self):
        """Get current test scenario settings."""
        return self.base_ui.get_scenario()

    def clear_plot(self):
        """Clear all plot data."""
        self.base_ui.clear_plot()

    def update_plot(self, t, ctrl, gimbal_angles, gimbal_cmd,
                    qpos=None, qvel=None, target_pos=None, target_yaw=None):
        """Update plots with new data."""
        self.base_ui.update_plot(t, ctrl, gimbal_angles, gimbal_cmd,
                                  qpos, qvel, target_pos, target_yaw)

    def update(self):
        """Process tkinter events (call from main loop)."""
        self.root.update()

    def close(self):
        """Close the UI window."""
        self.base_ui.close()
        self.root.destroy()

    @property
    def mode(self):
        """Get current mode."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """Set current mode."""
        self._mode = value

    _mode = "tune"
