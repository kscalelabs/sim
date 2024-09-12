import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import asdict
import sim.tdmpc.src.algorithm.helper as h


class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Qs = nn.ModuleList(
			[h.q(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim) for _ in range(cfg.num_q)])
		self.apply(h.orthogonal_init)
		for m in [self._reward, *self._Qs]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in self._Qs:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		Qs = torch.stack([self._Qs[i](x) for i in range(self.cfg.num_q)], dim=0)
		return Qs


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).to(self.device)
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=3 * self.cfg.lr)
		self.aug = nn.Identity()
		self.model.eval()
		self.model_target.eval()

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {
				'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict(),
				'config': asdict(self.cfg),
				}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.save_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(self.model.Q(z, self.model.pi(z, self.cfg.min_std)), dim=0)[0]
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		obs = obs.clone().to(self.device, dtype=torch.float32).unsqueeze(1)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, self.cfg.num_envs, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(1, num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(1, self.cfg.num_samples + num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.num_envs, self.cfg.action_dim, device=self.device)
		std = 2 * torch.ones(horizon, self.cfg.num_envs, self.cfg.action_dim, device=self.device)
		
		if isinstance(t0, bool) and t0 and hasattr(self, '_prev_mean') and self._prev_mean.shape[0] > 1:
			_prev_h = self._prev_mean.shape[0] - 1
			mean[:_prev_h] = self._prev_mean[1:]
		elif torch.is_tensor(t0) and t0.any() and hasattr(self, '_prev_mean') and self._prev_mean.shape[0] > 1:
			_prev_h = self._prev_mean.shape[0] - 1
			mean[:_prev_h] = self._prev_mean[1:]

		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(2) + std.unsqueeze(2) * \
				torch.randn(horizon, self.cfg.num_envs, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=-2)

			# Compute elite actions
			value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(-1), self.cfg.num_elites, dim=-1).indices
			elite_value, elite_actions = value.squeeze(-1).gather(-1, elite_idxs), actions.gather(-2, elite_idxs.unsqueeze(-1).repeat(horizon, 1, 1, self.cfg.action_dim))

			# Update parameters
			max_value = elite_value.max(1, keepdim=True)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(1, keepdim=True)
			_mean = torch.sum(score.unsqueeze(0).unsqueeze(-1) * elite_actions, dim=-2) / (score.sum(-1, keepdim=True).unsqueeze(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0).unsqueeze(-1) * (elite_actions - _mean.unsqueeze(2)) ** 2, dim=-2) / (score.sum(-1, keepdim=True).unsqueeze(0) + 1e-9))
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs
		select_indices = torch.multinomial(score, 1)
		actions = elite_actions.gather(-2, select_indices.unsqueeze(0).unsqueeze(-1).repeat(horizon, 1, 1, self.cfg.action_dim)).squeeze(-2)
		self._prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(self.model.Q(z, a), dim=0)[0]
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward, mask=1.0):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * mask * \
			torch.min(self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)), dim=0)[0]
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, mask, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		loss_mask = torch.ones_like(mask[0], device=self.device)
            

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg.horizon):
			if t > 0:
				loss_mask = loss_mask * mask[t - 1]
			# Predictions
			Qs = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, mask[t], reward[t])
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += loss_mask[t] * rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += loss_mask[t] * rho * h.mse(reward_pred, reward[t])
			for i in range(self.cfg.num_q):
				value_loss += loss_mask[t] * rho * h.mse(Qs[i], td_target)
				priority_loss += loss_mask[t] * rho * h.l1(Qs[i], td_target)

		# Optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		if step % self.cfg.update_freq == 0:
			# Update policy + target network
			pi_loss = self.update_pi(zs)
			h.ema(self.model, self.model_target, self.cfg.tau)

		self.model.eval()
		return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss if step % self.cfg.update_freq == 0 else 0.,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
