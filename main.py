import mujoco
import mujoco.viewer
import numpy as np
import time

from controllers import FlyingCarLQR, FlyingCarQP
from controllers.base import load_controller_weights, save_controller_weights
from ui import ControllerUI


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

    # Load controller weights from JSON
    weights = load_controller_weights("lqr_weights.json")

    # Controller registry
    controllers = {
        'LQR': lambda w, h: FlyingCarLQR(model, w),
        'QP': lambda w, h: FlyingCarQP(model, w, horizon=h or 10),
    }

    # Initialize with LQR controller
    current_controller_type = 'LQR'
    controller = controllers['LQR'](weights, None)

    # Get initial scenario from weights
    scenario = weights.get("test_scenario", {})
    start_pos = np.array(scenario.get("start_pos", [0, 0, 2]))
    start_yaw = np.radians(scenario.get("start_yaw", 0))
    target_pos = np.array(scenario.get("target_pos", [2, 1, 3]))
    target_yaw = np.radians(scenario.get("target_yaw", 0))

    # Flags to trigger actions from main thread
    reset_requested = False
    new_target_requested = False

    def reset_simulation():
        """Reset simulation to start state from current scenario."""
        nonlocal reset_requested
        reset_requested = True

    def on_controller_changed(controller_type, horizon=None):
        """Callback when controller type changes via UI."""
        nonlocal controller, current_controller_type
        current_controller_type = controller_type

        # Create new controller
        controller = controllers[controller_type](weights, horizon)
        print(f"Controller changed to {controller_type}" +
              (f" (horizon={horizon})" if horizon else ""))

    def on_weights_changed(new_weights):
        """Callback when weights are changed via UI."""
        nonlocal weights
        weights = new_weights
        controller.update_weights(new_weights)
        print(f"{current_controller_type} weights updated, controller recomputed")

    def on_save(weights_to_save):
        """Callback to save weights to JSON."""
        save_controller_weights(weights_to_save, "lqr_weights.json")

    def on_new_target():
        """Set a new random target position and yaw (play mode)."""
        nonlocal target_pos, target_yaw
        target_pos, target_yaw = random_target()
        print(f"New target: pos={target_pos}, yaw={np.degrees(target_yaw):.1f}°")

    # Setup controller UI
    ui = ControllerUI(
        weights=weights,
        on_controller_changed=on_controller_changed,
        on_weights_changed=on_weights_changed,
        on_reset_sim=reset_simulation,
        on_save=on_save,
        on_new_target=on_new_target
    )

    def apply_reset(viewer):
        """Apply simulation reset using current scenario from UI."""
        nonlocal target_pos, target_yaw, reset_requested

        scenario = ui.get_scenario()
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
        ui.clear_plot()

        reset_requested = False
        print(f"Simulation reset: start={start_pos}, target={target_pos}")

    # Keyboard callback for manual controls
    def key_callback(keycode):
        nonlocal reset_requested, new_target_requested
        if keycode == ord(' '):  # Spacebar
            if ui.mode == "tune":
                reset_requested = True
            else:
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

                # Update plots in UI
                gimbal_angles = data.qpos[7:15]
                gimbal_cmd = data.ctrl[4:12]
                ui.update_plot(data.time, data.ctrl, gimbal_angles, gimbal_cmd,
                               qpos=data.qpos, qvel=data.qvel,
                               target_pos=target_pos, target_yaw=target_yaw)

                # Process tkinter events
                ui.update()

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
            ui.close()


if __name__ == "__main__":
    main()
