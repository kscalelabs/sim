import glob
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import norm
from scipy.spatial import distance
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

__REDUCE__ = lambda b: "mean" if b else "none"



def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def bce(pred, target, logits=True, reduce=False):
    """Computes the BCE loss between predictions and targets."""
    if logits:
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction=__REDUCE__(reduce)
        )
    return F.binary_cross_entropy(pred, target, reduction=__REDUCE__(reduce))


def l1_quantile(y_true, y_pred, quantile=0.3):
    """
    Compute the quantile loss.

    Args:
    y_true (torch.Tensor): Ground truth values
    y_pred (torch.Tensor): Predicted values
    quantile (float): Quantile to compute, must be between 0 and 1

    Returns:
    torch.Tensor: Quantile loss
    """
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.mean(loss)


def threshold_l2_expectile(diff, threshold=1e-2, expectile=0.99, reduce=False):
    weight = torch.where(torch.abs(diff) > threshold, expectile, (1 - expectile))
    loss = weight * (diff**2)
    reduction = __REDUCE__(reduce)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    return loss


def l2_expectile(diff, expectile=0.7, reduce=False):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    loss = weight * (diff**2)
    reduction = __REDUCE__(reduce)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    return loss


def mse_expectile(pred, target, expectile=0.7, reduce=False):
    diff = pred - target
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    loss = weight * (diff**2)
    reduction = __REDUCE__(reduce)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    return loss


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .squeeze(0)
        .shape
    )


def gaussian_logprob(eps, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * eps.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)

def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)

class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enc(cfg):
    """Returns a TOLD encoder."""
    pixels_enc_layers, state_enc_layers = None, None
    if cfg.modality in {"pixels", "all"}:
        C = int(3 * cfg.frame_stack)
        pixels_enc_layers = [
            NormalizeImg(),
            nn.Conv2d(C, cfg.num_channels, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
        ]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), pixels_enc_layers)
        pixels_enc_layers.extend(
            [
                Flatten(),
                nn.Linear(np.prod(out_shape), cfg.latent_dim),
                nn.LayerNorm(cfg.latent_dim),
                nn.Sigmoid(),
            ]
        )
        if cfg.modality == "pixels":
            return ConvExt(nn.Sequential(*pixels_enc_layers))
    if cfg.modality in {"state", "all"}:
        state_dim = (
            cfg.obs_shape[0] if cfg.modality == "state" else cfg.obs_shape["state"][0]
        )
        state_enc_layers = [
            nn.Linear(state_dim, cfg.enc_dim),
            nn.LayerNorm(cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
            nn.LayerNorm(cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.Tanh(),
        ]
        if cfg.modality == "state":
            return nn.Sequential(*state_enc_layers)
    else:
        raise NotImplementedError

    encoders = {}
    for k in cfg.obs_shape:
        if k == "state":
            encoders[k] = nn.Sequential(*state_enc_layers)
        elif k.endswith("rgb"):
            encoders[k] = ConvExt(nn.Sequential(*pixels_enc_layers))
        else:
            raise NotImplementedError
    return Multiplexer(nn.ModuleDict(encoders))


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ReLU(), layer_norm=True):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    layers = [nn.Linear(in_dim, mlp_dim[0]), nn.LayerNorm(mlp_dim[0]) if layer_norm else nn.Identity(), act_fn]
    for i in range(len(mlp_dim) - 1):
        layers += [nn.Linear(mlp_dim[i], mlp_dim[i + 1]), nn.LayerNorm(mlp_dim[i + 1]) if layer_norm else nn.Identity(), act_fn]
    layers += [nn.Linear(mlp_dim[-1], out_dim)]
    return nn.Sequential(*layers)


def dynamics(in_dim, mlp_dim, out_dim, act_fn=nn.Mish()):
    """Returns a dynamics network."""
    return nn.Sequential(
        mlp(in_dim, mlp_dim, out_dim, act_fn),
        nn.LayerNorm(out_dim),
        act_fn,
    )


def q(in_dim, mlp_dim, act_fn=nn.ReLU(), layer_norm=True):
    """Returns a Q-function that uses Layer Normalization."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    layers = [nn.Linear(in_dim, mlp_dim[0]), nn.LayerNorm(mlp_dim[0]) if layer_norm else nn.Identity(), act_fn]
    for i in range(len(mlp_dim) - 1):
        layers += [nn.Linear(mlp_dim[i], mlp_dim[i + 1]), nn.LayerNorm(mlp_dim[i + 1]) if layer_norm else nn.Identity(), act_fn]
    layers += [nn.Linear(mlp_dim[-1], 1)]
    return nn.Sequential(*layers)


def v(in_dim, mlp_dim, act_fn=nn.ReLU(), layer_norm=True):
    """Returns a Q-function that uses Layer Normalization."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    layers = [nn.Linear(in_dim, mlp_dim[0]), nn.LayerNorm(mlp_dim[0]) if layer_norm else nn.Identity(), act_fn]
    for i in range(len(mlp_dim) - 1):
        layers += [nn.Linear(mlp_dim[i], mlp_dim[i + 1]), nn.LayerNorm(mlp_dim[i + 1]) if layer_norm else nn.Identity(), act_fn]
    layers += [[nn.Linear(mlp_dim[-1], 1)]]
    return nn.Sequential(*layers)


def aug(cfg):
    if cfg.modality == "state":
        return nn.Identity()
    else:
        augs = {}
        for k in cfg.obs_shape:
            if k == "state":
                augs[k] = nn.Identity()
            else:
                raise NotImplementedError
        return Multiplexer(nn.ModuleDict(augs))


class ConvExt(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        if x.ndim > 4:
            batch_shape = x.shape[:-3]
            out = self.conv(x.view(-1, *x.shape[-3:]))
            out = out.view(*batch_shape, *out.shape[1:])
        else:
            out = self.conv(x)
        return out


class Multiplexer(nn.Module):

    def __init__(self, choices):
        super().__init__()
        self.choices = choices

    def forward(self, x, key=None):
        if isinstance(x, dict):
            if key is not None:
                return self.choices[key](x)
            return {k: self.choices[k](_x) for k, _x in x.items()}
        return self.choices(x)
        
class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device(cfg.buffer_device)
        self.capacity = int(cfg.max_episode_length // cfg.action_repeat)
        if cfg.modality in {"pixels", "state"}:
            dtype = torch.float32 if cfg.modality == "state" else torch.uint8
            self.next_obses = torch.empty(
                (cfg.num_envs, self.capacity, *init_obs.shape[1:]),
                dtype=dtype,
                device=self.device,
            )
            self.obses = torch.empty(
                (cfg.num_envs, self.capacity, *init_obs.shape[1:]),
                dtype=dtype,
                device=self.device,
            )
            self.obses[:, 0] = init_obs.clone().to(self.device, dtype=dtype)
        elif cfg.modality == "all":
            self.obses = {}
            for k, v in init_obs.items():
                assert k in {"rgb", "state"}
                dtype = torch.float32 if k == "state" else torch.uint8
                self.next_obses[k] = torch.empty(
                    (cfg.num_envs, self.capacity, *v.shape[1:]), dtype=dtype, device=self.device
                )
                self.obses[k] = torch.empty(
                    (cfg.num_envs, self.capacity, *v.shape[1:]), dtype=dtype, device=self.device
                )
                self.obses[k][:, 0] = v.clone().to(self.device, dtype=dtype)
        else:
            raise ValueError
        self.actions = torch.empty(
            (cfg.num_envs, self.capacity, cfg.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty(
            (cfg.num_envs, self.capacity,), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty(
            (cfg.num_envs, self.capacity,), dtype=torch.bool, device=self.device
        )
        self.successes = torch.empty(
            (cfg.num_envs, self.capacity,), dtype=torch.bool, device=self.device
        )
        self.masks = torch.zeros(
            (cfg.num_envs, self.capacity,), dtype=torch.float32, device=self.device
        )
        self.cumulative_reward = torch.tensor([0.] * cfg.num_envs)
        self.done = torch.tensor([False] * cfg.num_envs)
        self.success = torch.tensor([False] * cfg.num_envs)
        self._idx = 0

    def __len__(self):
        return self._idx

    @property
    def episode_length(self):
        num_dones = self.dones[:, :self._idx].sum().item()
        if num_dones > 0:
            return float(self._idx) * self.cfg.num_envs / num_dones
        return float(self._idx)

    @classmethod
    def from_trajectory(cls, cfg, obses, actions, rewards, dones=None, masks=None):
        """Constructs an episode from a trajectory."""

        if cfg.modality in {"pixels", "state"}:
            episode = cls(cfg, obses[0])
            episode.obses[1:] = torch.tensor(
                obses[1:], dtype=episode.obses.dtype, device=episode.device
            )
        elif cfg.modality == "all":
            episode = cls(cfg, {k: v[0] for k, v in obses.items()})
            for k, v in obses.items():
                episode.obses[k][1:] = torch.tensor(
                    obses[k][1:], dtype=episode.obses[k].dtype, device=episode.device
                )
        else:
            raise NotImplementedError
        episode.actions = torch.tensor(
            actions, dtype=episode.actions.dtype, device=episode.device
        )
        episode.rewards = torch.tensor(
            rewards, dtype=episode.rewards.dtype, device=episode.device
        )
        episode.dones = (
            torch.tensor(dones, dtype=episode.dones.dtype, device=episode.device)
            if dones is not None
            else torch.zeros_like(episode.dones)
        )
        episode.masks = (
            torch.tensor(masks, dtype=episode.masks.dtype, device=episode.device)
            if masks is not None
            else torch.ones_like(episode.masks)
        )
        episode.cumulative_reward = torch.sum(episode.rewards)
        episode.done = True
        episode._idx = cfg.episode_length
        return episode

    @property
    def first(self):
        return len(self) == 0

    @property
    def full(self):
        return len(self) == self.capacity

    @property
    def buffer_capacity(self):
        return self.capacity

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done, timeouts, success=False):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if self._idx == self.capacity - 1:
                    self.next_obses[k][:, self._idx] = v.clone().to(self.obses[k].device, dtype=self.obses[k].dtype)
                elif self._idx < self.capacity - 1:
                    self.obses[k][:, self._idx + 1] = v.clone().to(self.obses[k].device, dtype=self.obses[k].dtype)
                    self.next_obses[k][:, self._idx] = self.obses[k][:, self._idx + 1].clone()
        else:
            if self._idx == self.capacity - 1:
                self.next_obses[:, self._idx] = obs.clone().to(self.obses.device, dtype=self.obses.dtype)
            elif self._idx < self.capacity - 1:
                self.obses[:, self._idx + 1] = obs.clone().to(self.obses.device, dtype=self.obses.dtype)
                self.next_obses[:, self._idx] = self.obses[:, self._idx + 1].clone()
        self.actions[:, self._idx] = action.detach().cpu()
        self.rewards[:, self._idx] = reward.detach().cpu()
        self.dones[:, self._idx] = done.detach().cpu()
        self.masks[:, self._idx] = 1.0 - timeouts.detach().cpu().float()  # TODO
        self.cumulative_reward += reward.detach().cpu()
        self.done = done.detach().cpu()
        self.success = torch.logical_or(self.success, success.detach().cpu())
        self.successes[:, self._idx] = torch.tensor(self.success).to(self.device)
        self._idx += 1

class ReplayBuffer:
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default."""

    def __init__(self, cfg, dataset=None):
        self.cfg = cfg
        self.buffer_device = torch.device(cfg.buffer_device)
        self.device = torch.device(cfg.device)
        self.batch_size = self.cfg.batch_size

        print("Replay buffer device: ", self.buffer_device)
        print("Replay buffer sample device: ", self.device)

        if dataset is not None:
            self.capacity = max(
                dataset["rewards"].shape[0], cfg.max_offline_buffer_size
            )
            print("Offline dataset size: ", dataset["rewards"].shape[0])
        else:
            self.capacity = max(cfg.train_steps, cfg.max_buffer_size)

        print("Maximum capacity of the buffer is: ", self.capacity)

        if cfg.modality in {"pixels", "state"}:
            dtype = torch.float32 if cfg.modality == "state" else torch.uint8
            # Note self.obs_shape always has single frame, which is different from cfg.obs_shape
            self.obs_shape = (
                cfg.obs_shape if cfg.modality == "state" else (3, *cfg.obs_shape[-2:])
            )
            self._obs = torch.empty(
                (self.capacity, *self.obs_shape), dtype=dtype, device=self.buffer_device
            )
            self._next_obs = torch.empty(
                (self.capacity, *self.obs_shape), dtype=dtype, device=self.buffer_device
            )
        elif cfg.modality == "all":
            self.obs_shape = {}
            self._obs, self._next_obs = {}, {}
            for k, v in cfg.obs_shape.items():
                assert k in {"rgb", "state"}
                dtype = torch.float32 if k == "state" else torch.uint8
                self.obs_shape[k] = v if k == "state" else (3, *v[-2:])
                self._obs[k] = torch.empty(
                    (self.capacity, *self.obs_shape[k]),
                    dtype=dtype,
                    device=self.buffer_device,
                )
                self._next_obs[k] = self._obs[k].clone()
        else:
            raise ValueError

        self._action = torch.empty(
            (self.capacity, cfg.action_dim),
            dtype=torch.float32,
            device=self.buffer_device,
        )
        self._reward = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.buffer_device
        )
        self._mask = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.buffer_device
        )
        self._done = torch.empty(
            (self.capacity,), dtype=torch.bool, device=self.buffer_device
        )
        self._success = torch.empty(
            (self.capacity,), dtype=torch.bool, device=self.buffer_device
        )
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.buffer_device
        )
        self.ep_len = int(self.cfg.max_episode_length // self.cfg.action_repeat)
        self._eps = 1e-6
        self._full = False
        self.idx = 0
        self._sampling_idx = 0
        if dataset is not None:
            self.init_from_offline_dataset(dataset)

        self._aug = aug(cfg)

    def init_from_offline_dataset(self, dataset):
        assert self.idx == 0 and not self._full
        n_transitions = int(len(dataset["rewards"]) * self.cfg.data_first_percent)

        def copy_data(dst, src, n):
            assert isinstance(dst, dict) == isinstance(src, dict)
            if isinstance(dst, dict):
                for k in dst:
                    copy_data(dst[k], src[k], n)
            else:
                dst[:n] = torch.from_numpy(src[:n])

        copy_data(self._obs, dataset["observations"], n_transitions)
        copy_data(self._next_obs, dataset["next_observations"], n_transitions)
        copy_data(self._action, dataset["actions"], n_transitions)
        if self.cfg.task.startswith("xarm"):
            # success = self._calc_sparse_success(dataset['success'])
            # copy_data(self._reward, success.astype(np.float32), n_transitions)
            if self.cfg.sparse_reward:
                copy_data(
                    self._reward, dataset["success"].astype(np.float32), n_transitions
                )
            copy_data(self._success, dataset["success"], n_transitions)
        else:
            copy_data(self._reward, dataset["rewards"], n_transitions)
        copy_data(self._mask, dataset["masks"], n_transitions)
        copy_data(self._done, dataset["dones"], n_transitions)
        self.idx = (self.idx + n_transitions) % self.capacity
        self._full = n_transitions >= self.capacity
        mask_idxs = np.array([n_transitions - i for i in range(1, self.cfg.horizon)])
        # _, episode_ends, _ = get_trajectory_boundaries_and_returns(dataset)
        # mask_idxs = np.array([np.array(episode_ends) - i for i in range(1, self.cfg.horizon)]).T.flatten()
        mask_idxs = np.clip(mask_idxs, 0, n_transitions - 1)
        self._priorities[mask_idxs] = 0

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):  
        idxs = torch.arange(self.idx, self.idx + self.cfg.num_envs * self.ep_len) % self.capacity
        self._sampling_idx = (self.idx + self.cfg.num_envs * self.ep_len) % self.capacity
        mask_copy = episode.masks.clone()
        mask_copy[:, episode._idx - self.cfg.horizon:] = 0.
        if self.cfg.modality in {"pixels", "state"}:
            self._obs[idxs] = (
                episode.obses.flatten(0, 1)
                if self.cfg.modality == "state"
                else episode.obses[:, -3:].flatten(0, 1)
            )
            self._next_obs[idxs] = (
                episode.next_obses.flatten(0, 1)
                if self.cfg.modality == "state"
                else episode.next_obses[:, -3:].flatten(0, 1)
            )
        elif self.cfg.modality == "all":
            for k, v in episode.obses.items():
                assert k in {"rgb", "state"}
                assert k in self._obs
                assert k in self._next_obs
                if k == "rgb":
                    self._obs[k][idxs] = episode.obses[k][:-1, -3:].flatten(0, 1)
                    self._next_obs[k][idxs] = episode.obses[k][1:, -3:].flatten(0, 1)
                else:
                    self._obs[k][idxs] = episode.obses[k][:-1].flatten(0, 1)
                    self._next_obs[k][idxs] = episode.obses[k][1:].flatten(0, 1)
        self._action[idxs] = episode.actions.flatten(0, 1)
        self._reward[idxs] = episode.rewards.flatten(0, 1)
        self._mask[idxs] = mask_copy.flatten(0, 1)
        self._done[idxs] = episode.dones.flatten(0, 1)
        self._success[idxs] = episode.successes.flatten(0, 1)
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = (
                1.0
                if self.idx == 0
                else self._priorities[: self.idx].max().to(self.device).item()
            )
        mask = torch.arange(self.ep_len) > self.ep_len - self.cfg.horizon
        mask = torch.cat([mask] * self.cfg.num_envs)
        new_priorities = torch.full((self.ep_len  * self.cfg.num_envs,), max_priority, device=self.buffer_device)
        new_priorities[mask] = 0
        new_priorities = new_priorities * self._mask[idxs]
        self._priorities[idxs] = new_priorities
        self._full = self._full or (self.idx + self.ep_len * self.cfg.num_envs > self.capacity)
        if episode.full:
            self.idx = (self.idx + self.ep_len * self.cfg.num_envs) % self.capacity

    def _set_bs(self, bs):
        self.batch_size = bs

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs.to(self.buffer_device)] = (
            priorities.squeeze(1).to(self.buffer_device) + self._eps
        )

    def _get_obs(self, arr, idxs, bs=None, frame_stack=None):
        if isinstance(arr, dict):
            return {
                k: self._get_obs(v, idxs, bs=bs, frame_stack=frame_stack)
                for k, v in arr.items()
            }
        if arr.ndim <= 2:  # if self.cfg.modality == 'state':
            return arr[idxs].cuda(self.device)
        obs = torch.empty(
            (
                self.cfg.batch_size if bs is None else bs,
                3 * self.cfg.frame_stack if frame_stack is None else 3 * frame_stack,
                *arr.shape[-2:],
            ),
            dtype=arr.dtype,
            device=torch.device(self.device),
        )
        obs[:, -3:] = arr[idxs].cuda(self.device)
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack if frame_stack is None else frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i + 1) * 3 : -i * 3] = arr[_idxs].cuda(self.device)
        return obs.float()

    def sample(self, bs=None):
        probs = (
            self._priorities if self._full else self._priorities[:self._sampling_idx]
        ) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        if torch.isnan(self._priorities).any():
            print(torch.isnan(self._priorities).any())
            print(torch.where(torch.isnan(self._priorities)))
        idxs = torch.from_numpy(
            np.random.choice(
                total,
                self.cfg.batch_size if bs is None else bs,
                p=probs.cpu().numpy(),
                replace=((not self._full) or (self.cfg.batch_size > self.capacity)),
            )
        ).to(self.buffer_device) % self.capacity
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        idxs_in_horizon = torch.stack([idxs + t for t in range(self.cfg.horizon)]) % self.capacity

        obs = self._aug(self._get_obs(self._obs, idxs, bs=bs))
        next_obs = [
            self._aug(self._get_obs(self._next_obs, _idxs, bs=bs))
            for _idxs in idxs_in_horizon
        ]
        if isinstance(next_obs[0], dict):
            next_obs = {k: torch.stack([o[k] for o in next_obs]) for k in next_obs[0]}
        else:
            next_obs = torch.stack(next_obs)
        action = self._action[idxs_in_horizon]
        reward = self._reward[idxs_in_horizon]
        mask = self._mask[idxs_in_horizon]
        done = torch.logical_not(self._done[idxs_in_horizon]).float()

        if not action.is_cuda:
            action, reward, done, idxs, weights = (
                action.cuda(self.device),
                reward.cuda(self.device),
                done.cuda(self.device),
                idxs.cuda(self.device),
                weights.cuda(self.device),
            )
        return (
            obs,
            next_obs,
            action,
            reward.unsqueeze(2),
            done.unsqueeze(2),
            idxs,
            weights,
        )