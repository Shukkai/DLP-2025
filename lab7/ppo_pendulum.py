#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh
import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import wandb
from torch.distributions import Normal
from tqdm import tqdm
import os

def orthogonal_init(layer: nn.Linear, gain: float = math.sqrt(2), bias_const: float = 0.0):
    """Orthogonal weight init with configurable gain (PPO detail #2)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            orthogonal_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        # Mean head (very small init, centred around 0)
        self.mu_head = orthogonal_init(nn.Linear(64, out_dim), gain=0.01)
        # State-independent log-std (initially 0)
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().clamp(1e-3, 2.0)
        dist = Normal(mu, std)
        raw_action = dist.rsample()             # before tanh
        tanh_action = torch.tanh(raw_action) * 2.0  # env range [−2,2]
        return tanh_action, raw_action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            orthogonal_init(nn.Linear(64, 64)), nn.Tanh(),
            orthogonal_init(nn.Linear(64, 1), gain=1.0),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)


def compute_gae(next_value: torch.Tensor, rewards: List[torch.Tensor], masks: List[torch.Tensor],
                values: List[torch.Tensor], gamma: float, lam: float):
    """Generalised Advantage Estimation (PPO/A2C detail)."""
    values = values + [next_value]
    gae, returns = 0.0, []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + values[i])
    return returns


def minibatch_generator(epochs: int, mb_size: int, *tensors):
    batch_size = tensors[0].size(0)
    for _ in range(epochs):
        idxs = torch.randperm(batch_size)
        for start in range(0, batch_size, mb_size):
            end = start + mb_size
            batch_idx = idxs[start:end]
            yield (t[batch_idx] for t in tensors)

class PPOAgent:
    def __init__(self, env: gym.Env, args):
        self.env = env
        self.gamma, self.lam = args.discount_factor, args.tau
        self.mb_size, self.eps_clip = args.batch_size, args.epsilon
        self.rollout_len, self.upd_epochs = args.rollout_len, args.update_epoch
        self.entropy_coef = args.entropy_weight
        self.device = torch.device(args.device)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        self.actor_lr0, self.critic_lr0 = args.actor_lr, args.critic_lr

        self.max_steps = args.max_steps
        self.total_steps = 0
        self.seed = args.seed

        # rollout buffers
        self.reset_buffers()
        self.video_folder = args.video_folder
        self.save_model_path = args.save_model_path
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.save_model_path, exist_ok=True)

        self.inference = args.inference
        self.load_model_path = args.load_model_path

    def reset_buffers(self):
        self.states, self.raw_actions, self.actions = [], [], []
        self.rewards, self.dones, self.values, self.log_probs = [], [], [], []

    def select_action(self, obs: np.ndarray, evaluate: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, raw_action, dist = self.actor(obs_t)
            value = self.critic(obs_t)
        if evaluate:
            return action.cpu().numpy()

        log_prob = dist.log_prob(raw_action).sum(-1)

        # store rollout data (detach to cut graph)
        self.states.append(obs_t.detach())
        self.raw_actions.append(raw_action.detach())
        self.actions.append(action.detach())
        self.values.append(value.detach())
        self.log_probs.append(log_prob.detach())
        return action.cpu().numpy()
    
    def update(self, next_obs: np.ndarray):
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_value = self.critic(next_obs_t)
        returns = compute_gae(next_value, self.rewards, self.dones, self.values,
                              self.gamma, self.lam)

        # stack tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.raw_actions)      # note: raw actions for log-prob maths
        returns = torch.stack(returns).detach()
        values = torch.stack(self.values).detach()
        log_probs_old = torch.stack(self.log_probs).detach()
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # advantage normalisation

        # linear LR decay
        frac = 1.0 - (self.total_steps / self.max_steps)
        for pg in self.actor_opt.param_groups:
            pg['lr'] = self.actor_lr0 * frac
        for pg in self.critic_opt.param_groups:
            pg['lr'] = self.critic_lr0 * frac

        actor_loss_hist, critic_loss_hist, kl_hist, clipfrac_hist = [], [], [], []

        for (mb_states, mb_actions, mb_values, mb_logp_old, mb_returns, mb_adv) in minibatch_generator(
                self.upd_epochs, self.mb_size,
                states, actions, values, log_probs_old, returns, adv):

            # new distributions
            _, _, dist = self.actor(mb_states)
            logp = dist.log_prob(mb_actions).sum(-1)
            ratio = (logp - mb_logp_old).exp()

            # surrogate objective
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            # actor update
            self.actor_opt.zero_grad()
            (policy_loss - self.entropy_coef * entropy).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            # critic update (Huber loss)
            value_pred = self.critic(mb_states)
            value_loss = F.smooth_l1_loss(value_pred, mb_returns)
            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()

            # stats
            approx_kl = (mb_logp_old - logp).mean().item()
            clipfrac = (torch.abs(ratio - 1.0) > self.eps_clip).float().mean().item()
            actor_loss_hist.append(policy_loss.item())
            critic_loss_hist.append(value_loss.item())
            kl_hist.append(approx_kl)
            clipfrac_hist.append(clipfrac)

        # clear buffers
        self.reset_buffers()
        return (np.mean(actor_loss_hist), np.mean(critic_loss_hist),
                np.mean(kl_hist), np.mean(clipfrac_hist))

    def train(self, args):
        obs, _ = self.env.reset(seed=self.seed)
        episode_return, ep_len = 0.0, 0

        for ep in tqdm(range(1, args.num_episodes + 1)):
            for _ in range(self.rollout_len):
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # store reward / mask
                self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
                self.dones.append(torch.tensor(0.0 if terminated else 1.0, device=self.device))

                obs = next_obs
                self.total_steps += 1
                episode_return += reward
                ep_len += 1

                if done:
                    obs, _ = self.env.reset()
                    wandb.log({"episode_return": episode_return, "episode_len": ep_len})
                    episode_return, ep_len = 0.0, 0

            # update after rollout segment
            a_loss, c_loss, kl, clipf = self.update(obs)
            wandb.log({"actor_loss": a_loss, "critic_loss": c_loss,
                       "approx_kl": kl, "clip_frac": clipf, "steps": self.total_steps})

            if ep % 10 == 0:
                avg_ret= self.evaluate(episodes=20)
                wandb.log({"eval_avg_return": avg_ret})
                if avg_ret > -150:
                    self.env.close()
                    break
        self.env.close()

    def evaluate(self, episodes: int = 20):
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        if self.inference:
            env = RecordVideo(env, video_folder=f"{self.video_folder}/inference")
        else:
            env = RecordVideo(env, video_folder=f"{self.video_folder}/step_{self.total_steps}")
        total = 0.0
        if self.inference:
            model = torch.load(self.load_model_path)
            self.actor.load_state_dict(model["actor_state_dict"])
            self.critic.load_state_dict(model["critic_state_dict"])
        for i in tqdm(range(episodes)):
            rng = np.random.randint(0, 1000000)
            obs, _ = env.reset(seed=rng)
            done = False
            while not done:
                act = self.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, _ = env.step(act)
                done = terminated or truncated
                total += reward
            
        avg_ret = total / episodes
        if avg_ret > -150 and not self.inference:
            print(f"🎉 Reached > -150 at step {self.total_steps}!")
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'env_name': "Pendulum-v1",
                'env_config': {"render_mode": "rgb_array"},
                'seeds': rng,
            }, f"{self.save_model_path}/ppo_snapshot_{self.total_steps}.pth")
            env.close()
        return avg_ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run_name", type=str, default="pendulum-ppo-enhanced")
    parser.add_argument("--actor_lr", type=float, default=4e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-3)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--num_episodes", type=int, default=199)
    parser.add_argument("--rollout_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_epoch", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--video_folder", type=str, default="gae_videos")
    parser.add_argument("--save_model_path", type=str, default="gae_models/")
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--load_model_path", type=str, default="")
    args = parser.parse_args()

    # env
    env = gym.make("Pendulum-v1")

    
    agent = PPOAgent(env, args)
    if args.inference:
        avg_ret = agent.evaluate(episodes=1)
        print(f"Inference: Avg Return = {avg_ret:.2f}")
    else:
        wandb.init(project="PPO-Pendulum-Enhanced", name=args.wandb_run_name, save_code=True)
        agent.train(args)
