#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh
import math, random, argparse, os
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

# Initialization helper
def ortho(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.)
    return layer

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            ortho(nn.Linear(in_dim, 128), math.sqrt(2)), nn.Tanh(),
            ortho(nn.Linear(128, 128), math.sqrt(2)), nn.Tanh(),
        )
        self.mu_head = ortho(nn.Linear(128, out_dim), 0.01)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
        self.act_limit = 1.0

    def forward(self, x):
        h = self.body(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().clamp(1e-3, 2.0)
        dist = Normal(mu, std)
        raw = dist.rsample()
        act = torch.tanh(raw) * self.act_limit
        return act, raw, dist

class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            ortho(nn.Linear(in_dim, 128), math.sqrt(2)), nn.Tanh(),
            ortho(nn.Linear(128, 128), math.sqrt(2)), nn.Tanh(),
            ortho(nn.Linear(128, 1), 1.0),
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

def compute_gae(next_v, rews, masks, vals, gamma, lam):
    vals = vals + [next_v]
    gae, returns = 0.0, []
    for i in reversed(range(len(rews))):
        delta = rews[i] + gamma * vals[i+1] * masks[i] - vals[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns.insert(0, gae + vals[i])
    return returns

def ppo_iter(epochs, mb, *tensors):
    B = tensors[0].shape[0]
    ids = np.arange(B)
    for _ in range(epochs):
        np.random.shuffle(ids)
        for start in range(0, B, mb):
            j = ids[start:start+mb]
            yield (t[j] for t in tensors)

class PPOAgent:
    def __init__(self, env, args):
        self.env = env
        self.gamma, self.lam = args.discount_factor, args.tau
        self.clip_eps = args.epsilon
        self.ent_start, self.ent_end = args.entropy_weight, 0.0
        self.kl_target = 0.01
        self.kl_coef = 0.5
        self.mb_size = args.batch_size
        self.upd_epochs = args.update_epoch
        self.rollout_len = args.rollout_len
        self.max_steps = args.max_steps
        self.device = torch.device(args.device)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.act_opt = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.crit_opt = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        self.actor_sched = CosineAnnealingLR(self.act_opt, args.max_steps)
        self.critic_sched = CosineAnnealingLR(self.crit_opt, args.max_steps)

        self.reset_buf()
        self.steps = 0
        self.video_folder = args.video_folder
        self.save_model_path = args.save_model_path
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.save_model_path, exist_ok=True)

    def reset_buf(self):
        self.S, self.A_raw, self.V, self.LP, self.R, self.M = [], [], [], [], [], []

    @torch.no_grad()
    def select(self, obs, eval=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        a, raw, dist = self.actor(obs_t)
        v = self.critic(obs_t)
        if eval:
            return a.cpu().numpy(), v
        self.S.append(obs_t)
        self.A_raw.append(raw)
        self.V.append(v)
        self.LP.append(dist.log_prob(raw).sum(-1))
        return a.cpu().numpy(), v

    def rollout(self):
        obs, _ = self.env.reset()
        for _ in range(self.rollout_len):
            act, _ = self.select(obs)
            next_obs, r, term, trunc, _ = self.env.step(act)
            done = term or trunc
            self.R.append(torch.tensor(r, device=self.device, dtype=torch.float32))
            self.M.append(torch.tensor(0.0 if done else 1.0, device=self.device, dtype=torch.float32))
            obs = next_obs
            self.steps += 1
            if done:
                obs, _ = self.env.reset()
        return obs

    def update(self, next_obs):
        next_v = self.critic(torch.tensor(next_obs, dtype=torch.float32, device=self.device))
        rets = compute_gae(next_v, self.R, self.M, self.V, self.gamma, self.lam)

        S = torch.stack(self.S)
        A = torch.stack(self.A_raw)
        V = torch.stack(self.V).detach()
        LP = torch.stack(self.LP).detach()
        R = torch.stack(rets).detach()
        ADV = (R - V - (R - V).mean()) / (R - V).std().clamp(min=1e-8)

        self.actor_sched.step()
        self.critic_sched.step()

        v_losses, p_losses, kl_list = [], [], []

        for s_mb, a_mb, v_mb, lp_mb, r_mb, adv_mb in ppo_iter(self.upd_epochs, self.mb_size, S, A, V, LP, R, ADV):
            _, _, dist = self.actor(s_mb)
            logp = dist.log_prob(a_mb).sum(-1)
            ratio = (logp - lp_mb).exp()
            kl = (lp_mb - logp).mean()

            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_mb
            actor_loss = -torch.min(surr1, surr2).mean() + self.kl_coef * kl
            entropy = dist.entropy().mean()

            self.act_opt.zero_grad()
            (actor_loss - self.ent_start * entropy).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.act_opt.step()

            value_loss = 0.5 * F.mse_loss(self.critic(s_mb), r_mb)
            self.crit_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.crit_opt.step()
            self.actor_sched.step()
            self.critic_sched.step()

            v_losses.append(value_loss.item())
            p_losses.append(actor_loss.item())
            kl_list.append(kl.item())

        self.reset_buf()
        return np.mean(p_losses), np.mean(v_losses), np.mean(kl_list)

    def train(self, total_iters=1500):
        for it in tqdm(range(1, total_iters+1)):
            nxt = self.rollout()
            p_loss, v_loss, kl = self.update(nxt)
            wandb.log({"actor_loss": p_loss, "critic_loss": v_loss, "kl": kl, "steps": self.steps})
            if it % 10 == 0:
                avg = self.evaluate()
                wandb.log({"eval_avg_return": avg, "steps": self.steps})
                print(f"Iter {it} | Steps {self.steps} | Eval {avg:.1f}")
                if avg > 2500:
                    print(f"\n🎉 Reached > 2500 reward at step {self.steps}")
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'steps': self.steps
                    }, f"{self.save_model_path}/ppo_snapshot_{self.steps}.pth")
                    break

    def evaluate(self, episodes=10):
        total = 0.0
        for _ in range(episodes):
            obs, _ = self.env.reset(seed=random.randint(0,1e6))
            done = False
            while not done:
                act, _ = self.select(obs, eval=True)
                obs, r, term, trunc, _ = self.env.step(act)
                done = term or trunc
                total += r
        return total / episodes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor_lr", type=float, default=3e-4)
    ap.add_argument("--critic_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--rollout_len", type=int, default=2048)
    ap.add_argument("--update_epoch", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--entropy_weight", type=float, default=0.01)
    ap.add_argument("--discount_factor", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--wandb_run", type=str, default="walker2d-ppo-enhanced")
    ap.add_argument("--max_steps", type=int, default=3000000)
    ap.add_argument("--video_folder", type=str, default="ppo_videos")
    ap.add_argument("--save_model_path", type=str, default="ppo_models")
    ap.add_argument("--device", type=str, default="cuda:1")
    args = ap.parse_args()

    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    wandb.init(project="Lab7-PPO-Walker", name=args.wandb_run, save_code=True)
    agent = PPOAgent(env, args)
    agent.train()
    
