import mujoco
import mujoco.viewer
import numpy as np
import time

from LQR import FlyingCarLQR


def add_thrust_arrow(viewer, site_id, thrust, max_thrust, data):
    """Add an arrow showing thrust magnitude at a thruster site."""
    if thrust < 0.1:
        return

    # Get site position and orientation
    site_pos = data.site_xpos[site_id].copy()
    site_mat = data.site_xmat[site_id].reshape(3, 3)

    # Arrow points in -z direction of site (opposite thrust direction), scaled by magnitude
    scale = thrust / max_thrust * 1.0  # Max arrow length 1.0m
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

    # Initialize LQR controller (this also computes hover state)
    controller = FlyingCarLQR(model)

    # Set initial state to hover equilibrium
    data.qpos[:] = controller.hover_qpos
    data.qvel[:] = controller.hover_qvel
    data.ctrl[:] = controller.hover_ctrl

    # Target position and yaw
    target_pos = np.array([2.0, 1.0, 3.0])
    target_yaw = 0.0

    # Keyboard callback for new random target
    def key_callback(keycode):
        nonlocal target_pos, target_yaw
        if keycode == ord(' '):  # Spacebar
            target_pos, target_yaw = random_target()
            print(f"New target: pos={target_pos}, yaw={np.degrees(target_yaw):.1f}°")

    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start_time = time.time()

        while viewer.is_running():
            # Compute control input
            target_quat = yaw_to_quat(target_yaw)
            data.ctrl[:] = controller.compute_control(data, target_pos, target_quat)

            # Step simulation
            mujoco.mj_step(model, data)

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


if __name__ == "__main__":
    main()
