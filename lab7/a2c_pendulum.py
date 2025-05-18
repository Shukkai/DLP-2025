#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import wandb
import math
import argparse
import os

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

def t(x):
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        x = np.array([x])
    return torch.from_numpy(x).float()
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
    
    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        
        return torch.distributions.Normal(means, stds)

class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        return self.model(X)
def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done)
        discounted.append(ret)
    
    return discounted[::-1]
def process_memory(memory, gamma=0.99, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []

    for action, reward, state, next_state, done in memory:
        actions.append(np.array(action))
        rewards.append(np.array(reward))
        states.append(np.array(state))
        next_states.append(np.array(next_state))
        dones.append(np.array(done))
    
    if discount_rewards:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(np.stack(actions)).view(-1, 1)
    states = t(np.stack(states))
    next_states = t(np.stack(next_states))
    rewards = t(np.stack(rewards)).view(-1, 1)
    dones = t(np.stack(dones)).view(-1, 1)
    return actions, rewards, states, next_states, dones

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

class A2CLearner():
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(memory, self.gamma, discount_rewards)

        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma*critic(next_states)*(1-dones)
        value = critic(states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()
        
        actor_loss = (-logs_probs*advantage.detach()).mean() - entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()
        
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        self.actor_optim.step()

        # critic
        critic_loss = F.smooth_l1_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        self.critic_optim.step()
        wandb.log({
            "Train/step": steps,
            "Train/actor_loss": actor_loss,
            "Train/critic_loss": critic_loss,
        })
class Runner():
    def __init__(self, env, max_steps):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.max_steps = max_steps
    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state, _ = self.env.reset()
    
    def run(self, max_steps, memory=None):
        if not memory: memory = []
        
        for i in range(max_steps):
            if self.done: self.reset()
            if self.steps >= self.max_steps:
                break
            dists = actor(t(self.state))
            actions = dists.sample().detach().data.numpy()
            actions_clipped = np.clip(actions, self.env.action_space.low.min(), env.action_space.high.max())

            next_state, reward, terminated, truncated, _ = self.env.step(actions_clipped)
            self.done = terminated or truncated
            memory.append((actions, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    print("episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)                    
        
        return memory
def test(env, actor, total_steps, video_folder, save_model_path, eval_episodes=20 ):
    total_reward = 0
    wrapped_env = gym.wrappers.RecordVideo(env, video_folder=f"{video_folder}/step_{total_steps}")
    for ep in range(eval_episodes):
        seed = np.random.randint(0, 1000000)
        state, _ = wrapped_env.reset(seed=seed)
        # wrapped_env.render()
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                dists = actor(t(state))
                actions = dists.mean.detach().cpu().numpy()  # Use mean for deterministic policy
            actions_clipped = np.clip(actions, env.action_space.low.min(), env.action_space.high.max())
            next_state, reward, terminated, truncated, _ = wrapped_env.step(actions_clipped)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        total_reward += episode_reward

    average_reward = total_reward / eval_episodes
    print(f"Eval: Avg Score = {average_reward:.2f} over {eval_episodes} episodes at step {total_steps}")
    if average_reward > -150:
        print(f" Reached > -150 at step {total_steps}!")
        torch.save({
            'actor_state_dict': learner.actor.state_dict(),
            'critic_state_dict': learner.critic.state_dict(),
            'env_name': "Pendulum-v1",
            'env_config': {"render_mode": "rgb_array"},
            'seeds': seed,
        }, f"{save_model_path}/a2c_snapshot_{total_steps}.pth")
    wrapped_env.close()
    return average_reward


def inference(env, actor, video_folder):
    total_reward = 0
    wrapped_env = gym.wrappers.RecordVideo(env, video_folder=f"{video_folder}/inference")
    done = False
    state, _ = wrapped_env.reset()
    while not done:
        with torch.no_grad():
            dists = actor(t(state))
            actions = dists.mean.detach().cpu().numpy()
        actions_clipped = np.clip(actions, env.action_space.low.min(), env.action_space.high.max())
        next_state, reward, terminated, truncated, _ = wrapped_env.step(actions_clipped)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    wrapped_env.close()
    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run_name", type=str, default="Pendulum-v1")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_episodes", type=int, default=5000)
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--steps_on_memory", type=int, default=32)
    parser.add_argument("--entropy_beta", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--critic_lr", type=float, default=4e-3)
    parser.add_argument("--actor_lr", type=float, default=4e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--video_folder", type=str, default="a2c_videos")
    parser.add_argument("--save_model_path", type=str, default="a2c_models/")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--load_model_path", type=str, default="")
    args = parser.parse_args()
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    os.makedirs(args.video_folder, exist_ok=True)
    os.makedirs(args.save_model_path, exist_ok=True)
    # config
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions, activation=Mish)
    critic = Critic(state_dim, activation=Mish)

    learner = A2CLearner(actor, critic, args.gamma, args.entropy_beta, args.actor_lr, args.critic_lr, args.max_grad_norm)
    runner = Runner(env, args.max_steps)
    total_steps = (args.episode_length*args.max_episodes)//args.steps_on_memory
    if args.inference:
        model = torch.load(args.load_model_path)
        learner.actor.load_state_dict(model["actor_state_dict"])
        learner.critic.load_state_dict(model["critic_state_dict"])
        total_reward = inference(env, learner.actor, args.video_folder)
        print(f"Inference: Total Reward = {total_reward:.2f}")
        env.close()
        exit()
    else:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
        for i in range(total_steps):
            memory = runner.run(args.steps_on_memory)
            wandb.log({
                "Train/episode": len(runner.episode_rewards),
                "Train/episode_reward": runner.episode_reward,
            })
            if len(runner.episode_rewards) % 10 == 0:
                average_reward= test(runner.env, learner.actor,runner.steps,args.video_folder,args.save_model_path)
                wandb.log({
                    "Eval/episode": len(runner.episode_rewards),
                    "Eval/episode_reward": average_reward,
                })
                if average_reward > -150:
                    env.close()
                    break
            learner.learn(memory, runner.steps, discount_rewards=False)