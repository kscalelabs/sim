from enum import Enum
from typing import List

import numpy as np

from sim.utils.helpers import draw_vector

import torch  # isort: skip


class CommandMode(Enum):
    FIXED = "fixed"
    OSCILLATING = "oscillating"
    KEYBOARD = "keyboard"
    RANDOM = "random"


class CommandManager:
    """Manages robot commands"""

    def __init__(
        self,
        num_envs: int = 1,
        mode: str = "fixed",
        default_cmd: List[float] = [0.3, 0.0, 0.0, 0.0],
        device="cpu",
        env_cfg=None,
    ):
        self.num_envs = num_envs
        self.mode = CommandMode(mode)
        self.device = device
        self.default_cmd = torch.tensor(default_cmd, device=self.device)
        self.commands = self.default_cmd.repeat(num_envs, 1)
        self.time = 0
        self.env_cfg = env_cfg

        # Mode-specific parameters
        if self.mode == CommandMode.OSCILLATING:
            self.osc_period = 5.0  # secs
            self.min_x_vel = env_cfg.commands.ranges.lin_vel_x[0] if env_cfg else 0.0
            self.max_x_vel = env_cfg.commands.ranges.lin_vel_x[1] if env_cfg else 0.3
            self.osc_amplitude = (self.max_x_vel - self.min_x_vel) / 2
            self.osc_offset = (self.max_x_vel + self.min_x_vel) / 2
        elif self.mode == CommandMode.RANDOM:
            self.cmd_ranges = {
                'lin_vel_x': env_cfg.commands.ranges.lin_vel_x,
                'lin_vel_y': env_cfg.commands.ranges.lin_vel_y,
                'ang_vel_yaw': env_cfg.commands.ranges.ang_vel_yaw,
                'heading': env_cfg.commands.ranges.heading
            } if env_cfg else {
                'lin_vel_x': [-0.05, 0.23],
                'lin_vel_y': [-0.05, 0.05],
                'ang_vel_yaw': [-0.5, 0.5],
                'heading': [-np.pi, np.pi]
            }
            self.resampling_time = env_cfg.commands.resampling_time if env_cfg else 8.0
            self.last_sample_time = 0.0
        elif self.mode == CommandMode.KEYBOARD:
            try:
                import pygame
                pygame.init()
                pygame.display.set_mode((100, 100))
                self.x_vel_cmd = 0.0
                self.y_vel_cmd = 0.0
                self.yaw_vel_cmd = 0.0
            except ImportError:
                print("WARNING: pygame not found, falling back to fixed commands")
                self.mode = CommandMode.FIXED

    def close(self):
        if self.mode == CommandMode.KEYBOARD:
            import pygame
            pygame.quit()

    def update(self, dt: float) -> torch.Tensor:
        """Updates and returns commands based on current mode."""
        self.time += dt

        if self.mode == CommandMode.FIXED:
            return self.commands
        elif self.mode == CommandMode.OSCILLATING:
            # Oscillate x velocity between min and max
            x_vel = self.osc_offset + self.osc_amplitude * torch.sin(
                torch.tensor(2 * np.pi * self.time / self.osc_period)
            )
            self.commands[:, 0] = x_vel.to(self.device)
        elif self.mode == CommandMode.RANDOM:
            if self.time - self.last_sample_time >= self.resampling_time:
                self.last_sample_time = self.time
                # Generate random commands within training ranges
                new_commands = torch.tensor([
                    np.random.uniform(*self.cmd_ranges['lin_vel_x']),
                    np.random.uniform(*self.cmd_ranges['lin_vel_y']),
                    0.0,
                    np.random.uniform(*self.cmd_ranges['heading'])
                ], device=self.device) if self.env_cfg and self.env_cfg.commands.heading_command else torch.tensor([
                    np.random.uniform(*self.cmd_ranges['lin_vel_x']),
                    np.random.uniform(*self.cmd_ranges['lin_vel_y']),
                    np.random.uniform(*self.cmd_ranges['ang_vel_yaw']),
                    0.0
                ], device=self.device)
                self.commands = new_commands.repeat(self.num_envs, 1)
        elif self.mode == CommandMode.KEYBOARD:
            self._handle_keyboard_input()
            self.commands[:, 0] = torch.tensor(self.x_vel_cmd, device=self.device)
            self.commands[:, 1] = torch.tensor(self.y_vel_cmd, device=self.device)
            self.commands[:, 2] = torch.tensor(self.yaw_vel_cmd, device=self.device)

        return self.commands

    def draw(self, gym, viewer, env_handles, robot_positions, actual_vels) -> None:
        """Draws command and actual velocity arrows for all robots."""
        if viewer is None:
            return

        gym.clear_lines(viewer)
        cmd_vels = self.commands[:, :2].cpu().numpy()
        for env_handle, robot_pos, cmd_vel, actual_vel in zip(env_handles, robot_positions, cmd_vels, actual_vels):
            draw_vector(gym, viewer, env_handle, robot_pos, cmd_vel, (0.0, 1.0, 0.0))  # cmd vector (green)
            draw_vector(gym, viewer, env_handle, robot_pos, actual_vel, (1.0, 0.0, 0.0))  # vel vector (red)

    def _handle_keyboard_input(self):
        """Handles keyboard input for command updates."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        keys = pygame.key.get_pressed()

        # Update movement commands based on arrow keys
        if keys[pygame.K_UP]:
            self.x_vel_cmd = min(self.x_vel_cmd + 0.0005, 0.5)
        if keys[pygame.K_DOWN]:
            self.x_vel_cmd = max(self.x_vel_cmd - 0.0005, -0.5)
        if keys[pygame.K_LEFT]:
            self.y_vel_cmd = min(self.y_vel_cmd + 0.0005, 0.5)
        if keys[pygame.K_RIGHT]:
            self.y_vel_cmd = max(self.y_vel_cmd - 0.0005, -0.5)

        # Yaw control
        if keys[pygame.K_a]:
            self.yaw_vel_cmd = min(self.yaw_vel_cmd + 0.001, 0.5)
        if keys[pygame.K_z]:
            self.yaw_vel_cmd = max(self.yaw_vel_cmd - 0.001, -0.5)

        # Reset commands
        if keys[pygame.K_SPACE]:
            self.x_vel_cmd = 0.0
            self.y_vel_cmd = 0.0
            self.yaw_vel_cmd = 0.0
