# mypy: disable-error-code="valid-newtype"
"""Defines a simple demo script for simulating a MJCF to observe the physics.

Run with mjpython:
    mjpython sim/scripts/simulate_mjcf.py --record
"""

import argparse
import logging
import time
from pathlib import Path

import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np

from sim.env import stompy_mjcf_path
from sim.logging import configure_logging

logger = logging.getLogger(__name__)


def simulate(model_path: str | Path, duration: float, framerate: float, record_video: bool) -> None:
    frames: list[np.ndarray] = []
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while data.time < duration:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if record_video and len(frames) < data.time * framerate:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

        if record_video:
            video_path = "mjcf_simulation.mp4"
            media.write_video(video_path, frames, fps=framerate)
            # print(f"Video saved to {video_path}")
            logger.info("Video saved to %s", video_path)


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="MuJoCo Simulation")
    parser.add_argument("--model_path", type=str, default=str(stompy_mjcf_path()), help="Path to the MuJoCo XML file")
    parser.add_argument("--duration", type=int, default=3, help="Duration of the simulation in seconds")
    parser.add_argument("--framerate", type=int, default=30, help="Frame rate for video recording")
    parser.add_argument("--record", action="store_true", help="Flag to record video")
    args = parser.parse_args()

    simulate(args.model_path, args.duration, args.framerate, args.record)
