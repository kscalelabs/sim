# mypy: disable-error-code="valid-newtype"
"""Defines a simple demo script for simulating a MJCF to observe the physics.

Run with mjpython:
    mjpython mjpython sim/scripts/simulate_mjcf.py
"""

import time
import mujoco
import mujoco.viewer
import mediapy as media
import argparse


def simulate(model_path, duration, framerate, record_video):
    frames = []
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    while data.time < duration:
        step_start = time.time()
        mujoco.mj_step(model, data)

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
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo Simulation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MuJoCo XML model file")
    parser.add_argument("--duration", type=int, default=3, help="Duration of the simulation in seconds")
    parser.add_argument("--framerate", type=int, default=60, help="Frame rate for video recording")
    parser.add_argument("--record", action="store_true", help="Flag to record video")

    args = parser.parse_args()

    simulate(args.model_path, args.duration, args.framerate, args.record)
