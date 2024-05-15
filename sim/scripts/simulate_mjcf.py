# mypy: disable-error-code="valid-newtype"
"""Defines a simple demo script for simulating a MJCF to observe the physics.

Run with mjpython:
    mjpython mjpython sim/scripts/simulate_mjcf.py
"""

import time

import mujoco
import mujoco.viewer

from sim.env import stompy_mjcf_path

model = mujoco.MjModel.from_xml_path(stompy_mjcf_path())
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
