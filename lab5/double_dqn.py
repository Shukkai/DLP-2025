# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL - Double DQN Implementation
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import torch.nn.functional as F
import math
# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Register ALE environments
gym.register_envs(ale_py)
wandb.login(key="dd34bf1500f37d65014102de33b37a9c94671f07")

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.5 * mu_range)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions, atom_size=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer('support', torch.linspace(v_min, v_max, atom_size))

        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.feature.apply(init_weights)
        # Calculate feature size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            feature_size = self.feature(sample).view(1, -1).size(1)

        self.fc_h_v = NoisyLinear(feature_size, 512)
        self.fc_h_a = NoisyLinear(feature_size, 512)
        self.fc_z_v = NoisyLinear(512, atom_size)
        self.fc_z_a = NoisyLinear(512, n_actions * atom_size)

    def dist(self, x):
        """Get distribution for each action-value pair."""
        feature = self.feature(x)
        v_h = F.relu(self.fc_h_v(feature))
        a_h = F.relu(self.fc_h_a(feature))
        
        v = self.fc_z_v(v_h)
        a = self.fc_z_a(a_h)
        
        v = v.view(-1, 1, self.atom_size)
        a = a.view(-1, self.n_actions, self.atom_size)
        
        q = v + a - a.mean(1, keepdim=True)
        dist = F.softmax(q, dim=2)
        
        return dist

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        self.fc_h_v.reset_noise()
        self.fc_h_a.reset_noise()
        self.fc_z_v.reset_noise()
        self.fc_z_a.reset_noise()


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0  # Initialize max priority to 1.0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)  # Update max priority

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None
        probs = self.priorities[:len(self.buffer)] / np.sum(self.priorities[:len(self.buffer)])
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        transitions = [self.buffer[i] for i in indices]
        return transitions, indices, weights

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)  # Update max priority

class DoubleDQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        set_seed(42)
        

        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")

        
        self.env.reset(seed=42)
        self.test_env.reset(seed=42)

        self.num_actions = self.env.action_space.n

        self.preprocessor = AtariPreprocessor()
        input_shape = (4, 84, 84)  # 4 stacked frames

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Initialize networks based on environment type and Rainbow flag
        self.q_net = RainbowDQN(
            input_shape=input_shape,
            n_actions=self.num_actions,
            atom_size=args.atom_size,
            v_min=args.v_min,
            v_max=args.v_max
        ).to(self.device)
        self.target_net = RainbowDQN(
            input_shape=input_shape,
            n_actions=self.num_actions,
            atom_size=args.atom_size,
            v_min=args.v_min,
            v_max=args.v_max
        ).to(self.device)
        # Ensure support tensors are on the correct device
        self.q_net.support = self.q_net.support.to(self.device)
        self.target_net.support = self.target_net.support.to(self.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=args.lr, weight_decay=1e-5)
        
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.replay_start_size
        )
        
        self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_episode_steps * 1000,
            eta_min=args.lr * 0.01
        )

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.n_steps = args.n_steps  # List of n-step values
        self.n_step_buffers = {n: [] for n in self.n_steps}  # Separate buffer for each n-step

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)

        # Rainbow DQN specific parameters
        self.use_rainbow = args.use_rainbow
        self.atom_size = args.atom_size
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.beta_anneal_steps = args.beta_anneal_steps
        self.beta = self.beta_start

        self.train_freq = args.train_frequency
        self.best_avg_eval_reward = -float('inf')  # Track the best running average reward
        self.eval_rewards_window = deque(maxlen=20)  # Keep last 20 evaluation rewards



    def _get_n_step_info(self, n_step_buffer, n):
        """Calculate n-step return and next state."""
        reward = 0
        for i in range(len(n_step_buffer)):
            reward += (self.gamma ** i) * n_step_buffer[i][2]
        next_state = n_step_buffer[-1][3]
        done = n_step_buffer[-1][4]
        return reward, next_state, done

    def _store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer with multiple n-step returns."""
        # Store for each n-step value
        for n in self.n_steps:
            self.n_step_buffers[n].append((state, action, reward, next_state, done))
            
            if len(self.n_step_buffers[n]) < n and not done:
                continue
                
            # Calculate n-step return
            reward_n, next_state_n, done_n = self._get_n_step_info(self.n_step_buffers[n], n)
            state_n, action_n = self.n_step_buffers[n][0][0], self.n_step_buffers[n][0][1]
            
            # Store with max priority
            self.memory.add((state_n, action_n, reward_n, next_state_n, done_n), self.memory.max_priority)
                
            if done:
                self.n_step_buffers[n] = []
            else:
                self.n_step_buffers[n].pop(0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        state_tensor = state_tensor / 255.0  # normalize
        if self.use_rainbow:
            self.q_net.reset_noise()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        # Decay function for epsilon-greedy exploration
        self.epsilon = max(self.epsilon_min, self.epsilon_start - (self.epsilon_start - self.epsilon_min) * self.env_count / self.epsilon_decay_steps)
        self.train_count += 1
       
        transitions, indices, is_weights = self.memory.sample(self.batch_size)
        if transitions is None:
            return
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
            
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        states = states / 255.0  # normalize
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        next_states = next_states / 255.0  # normalize
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Rainbow DQN: Distributional RL
        with torch.no_grad():
            # Get next state distribution
            next_dist = self.target_net.dist(next_states)
            next_actions = self.q_net(next_states).argmax(1)
            next_dist = next_dist[range(self.batch_size), next_actions]
            
            # Project distribution
            t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * self.target_net.support
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / ((self.v_max - self.v_min) / (self.atom_size - 1))
            l = b.floor().long()
            u = b.ceil().long()
            
            # Create offset tensor on the same device
            offset = torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size, device=self.device).long().unsqueeze(1).expand(self.batch_size, self.atom_size)
            
            # Initialize projection distribution on the same device
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            
            # Project onto the support
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
        # Get current distribution
        dist = self.q_net.dist(states)
        log_p = torch.log(dist[range(self.batch_size), actions])
        
        # Compute loss
        loss = -(proj_dist * log_p).sum(1).mean()
        
        # Anneal beta
        self.beta = min(1.0, self.beta_start + (self.beta_end - self.beta_start) * (self.train_count / self.beta_anneal_steps))
        
        # Update priorities
        td_errors = torch.abs(proj_dist - dist[range(self.batch_size), actions]).sum(1).detach().cpu().numpy()
        
        self.memory.update_priorities(indices, td_errors)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        if self.train_count < self.replay_start_size:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            if self.use_rainbow:
                self.target_net.reset_noise()
                self.q_net.reset_noise()

        if self.train_count % 1000 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"[Train #{self.train_count}] Total Env Steps: {self.env_count} Loss: {loss.item():.4f}")
            wandb.log({
                "Train Loss": loss.item(),
                "Learning Rate": current_lr,
                "Train Count": self.train_count,
                "Total Env Steps": self.env_count,
                "Epsilon": self.epsilon,
                "Beta": self.beta,
                "Train Frequency": self.train_freq
            })

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)

            # Random no-ops at start (important for Atari)
            no_op_steps = np.random.randint(0, 31)  # random [0, 30]
            for _ in range(no_op_steps):
                next_obs, _, terminated, truncated, _ = self.env.step(0)  # "NO-OP" usually action 0
                done = terminated or truncated
                if done:
                    next_obs, _ = self.env.reset()
                state = self.preprocessor.reset(next_obs)

                
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                reward = np.clip(reward, -1.0, 1.0)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                    
                self._store_transition(state, action, reward, next_state, done)

                for _ in range(self.train_per_step):
                    self.train()

                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Train Frequency": self.train_freq
                    })
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Train Frequency": self.train_freq
            })
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 10 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                    "Best Reward": self.best_reward,
                    "Train Frequency": self.train_freq
                })

    def evaluate(self):
        self.q_net.eval()
        
        obs, _ = self.test_env.reset()
        
        state = self.preprocessor.reset(obs)

        no_op_steps = np.random.randint(0, 31)
        for _ in range(no_op_steps):
            next_obs, _, terminated, truncated, _ = self.test_env.step(0)
            done = terminated or truncated
            if done:
                next_obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(next_obs)


            
        done = False
        total_reward = 0
        eval_losses = []
        eval_q_values = []

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = q_values.argmax().item()
                eval_q_values.append(q_values.max().item())
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            
            state = self.preprocessor.step(next_obs)
            
                
            # Calculate evaluation loss
            with torch.no_grad():
                next_q_values = self.target_net(torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device))
                target_q_value = reward + (1 - float(done)) * self.gamma * next_q_values.max().item()
                current_q_value = q_values[0, action].item()
                eval_loss = F.smooth_l1_loss(torch.tensor([current_q_value]), torch.tensor([target_q_value]))
                eval_losses.append(eval_loss.item())
            
            total_reward += reward
        
        self.q_net.train()
        
        # Log evaluation metrics
        avg_eval_loss = np.mean(eval_losses) if eval_losses else 0.0
        avg_eval_q = np.mean(eval_q_values) if eval_q_values else 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[Eval] Total Env Steps: {self.env_count} Eval Reward: {total_reward:.2f} Loss: {avg_eval_loss:.4f} Q mean: {avg_eval_q:.3f}")
        wandb.log({
            "Eval Loss": avg_eval_loss,
            "Eval Q Mean": avg_eval_q,
            "Eval Reward": total_reward,
            "Eval Learning Rate": current_lr,
            "Eval Count": self.train_count,
            "Total Env Steps": self.env_count,
            "Train Frequency": self.train_freq
        })
        ### Save model if better
        if total_reward > self.best_avg_eval_reward:
            print(f"🏆 New best running average: {total_reward:.2f}, saving model...")
            self.best_avg_eval_reward = total_reward
            model_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(self.q_net.state_dict(), model_path)
            wandb.log({
                "Best Eval Running Average": total_reward,
                "Saved Best Model at Steps": self.env_count
            })
        
        return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results_double_dqn")
    parser.add_argument("--wandb-run-name", type=str, default="double-dqn-run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000000)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--use-prioritized-replay", type=bool, default=True, help="Use prioritized replay buffer instead of uniform")
    parser.add_argument("--env-name", type=str, default="ALE/Breakout-v5", help="Name of the environment to use")
    parser.add_argument("--n-steps", type=str, default="1,3,5", help="Comma-separated list of n-step values")
    
    # Rainbow DQN specific parameters
    parser.add_argument("--use-rainbow", type=bool, default=True, help="Use Rainbow DQN instead of standard DQN")
    parser.add_argument("--atom-size", type=int, default=51, help="Number of atoms for distributional RL")
    parser.add_argument("--v-min", type=float, default=-10, help="Minimum value of support")
    parser.add_argument("--v-max", type=float, default=10, help="Maximum value of support")
    parser.add_argument("--beta-start", type=float, default=0.4, help="Initial value of beta for prioritized replay")
    parser.add_argument("--beta-end", type=float, default=1.0, help="Final value of beta for prioritized replay")
    parser.add_argument("--beta-anneal-steps", type=int, default=100000, help="Number of steps to anneal beta")
    parser.add_argument("--episodes", type=int, default=200000, help="Number of episodes to train")
    parser.add_argument("--train-frequency", type=int, default=4, help="Number of steps between training")
    parser.add_argument("--eval-frequency", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of episodes to evaluate")
    parser.add_argument("--eval-interval", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to evaluate")
    args = parser.parse_args()
    
    # Convert n-steps string to list of integers
    args.n_steps = [int(n) for n in args.n_steps.split(',')]
    
    if args.env_name == "CartPole-v1":
        wandb.init(project="DLP-Lab5-DoubleDQN-CartPole", name=args.wandb_run_name, save_code=True)
    elif args.env_name == "ALE/Breakout-v5":
        wandb.init(project="DLP-Lab5-DoubleDQN-Atari", name=args.wandb_run_name, save_code=True)
    elif args.env_name == "ALE/Pong-v5":
        wandb.init(project="DLP-Lab5-DoubleDQN-Pong", name=args.wandb_run_name, save_code=True)
    
    agent = DoubleDQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.num_episodes) 