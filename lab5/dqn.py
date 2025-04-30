# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
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
def set_seed(seed=1234):  # Changed to 1234, a commonly used seed in RL research
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set environment seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set cv2 seed
    cv2.setRNGSeed(seed)

# Register ALE environments
gym.register_envs(ale_py)
wandb.login(key="dd34bf1500f37d65014102de33b37a9c94671f07")

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()

        ########## YOUR CODE HERE (5~10 lines) ##########

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        # Initialize weights using Kaiming initialization
        self.apply(init_weights)

    def forward(self, x):
        return self.network(x)

class DQN_atari(nn.Module):
    """
        Design the architecture of your deep Q network for Atari games
        - Input: 4 stacked 84x84 grayscale frames
        - Output: Q-values for each action
    """
    def __init__(self, input_channels, num_actions):
        super(DQN_atari, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        # Initialize weights using Kaiming initialization
        self.apply(init_weights)

    def forward(self, x):
        # Reshape input to (batch_size, channels, height, width)
        if len(x.shape) == 3:  # If input is (channels, height, width)
            x = x.unsqueeze(0)  # Add batch dimension
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
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

        

class UniformReplayBuffer:
    """
        A simple replay buffer with uniform sampling
    """ 
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        # Uniform sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        return transitions


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # Set random seed
        set_seed(args.seed if args is not None else 1234)
        
        # Create environments first
        self.is_atari = "ALE" in env_name or "Atari" in env_name
        if self.is_atari:
            self.env = gym.make(env_name, render_mode="rgb_array")
            self.test_env = gym.make(env_name, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name, render_mode="rgb_array")
            self.test_env = gym.make(env_name, render_mode="rgb_array")
        
        # Set environment seeds
        self.env.reset(seed=args.seed if args is not None else 1234)
        self.test_env.reset(seed=args.seed if args is not None else 1234)

        self.num_actions = self.env.action_space.n
        
        # Calculate input dimension based on environment type
        if self.is_atari:
            self.preprocessor = AtariPreprocessor()
            input_shape = (4, 84, 84)
        else:
            self.preprocessor = None
            input_dim = self.env.observation_space.shape[0]  # For CartPole: 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Initialize networks based on environment type and Rainbow flag
        if self.is_atari:
            self.q_net = DQN_atari(input_shape[0], self.num_actions).to(self.device)
            self.target_net = DQN_atari(input_shape[0], self.num_actions).to(self.device)
        else:
            self.q_net = DQN(input_dim, self.num_actions).to(self.device)
            self.target_net = DQN(input_dim, self.num_actions).to(self.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Initialize optimizer and scheduler
        # self.optimizer = optim.AdamW(self.q_net.parameters(), lr=args.lr, weight_decay=1e-5)
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=args.lr, alpha=0.99, eps=1e-3)
        
        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.replay_start_size
        )
        
        # Main cosine annealing scheduler
        self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_episode_steps * args.episodes,  # Total training steps
            eta_min=args.lr * 0.01  # Minimum learning rate is 1% of initial
        )

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_start = args.epsilon_start
        self.eps_decay_steps = args.eps_decay_steps
        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = UniformReplayBuffer(capacity=args.memory_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        state_tensor = state_tensor / 255.0  # normalize
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            if self.is_atari:
                # Add random no-op actions at the start
                no_op_steps = np.random.randint(0, 31)
                for _ in range(no_op_steps):
                    action = 0  # usually "NO-OP" is action 0 in Atari
                    next_obs, _, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    if done:
                        next_obs, _ = self.env.reset()
                    if self.is_atari:
                        state = self.preprocessor.reset(next_obs)
                    else:
                        state = next_obs


            
            done = False
            total_reward = 0
            step_count = 0
            episode_q_values = []
            episode_losses = []

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                reward = np.clip(reward, -1.0, 1.0)
                done = terminated or truncated
                
                if self.is_atari:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs
                    
                # Add transition to memory
                self.memory.add((state, action, reward, next_state, done))

                if self.env_count % 4 == 0:  # Train every 4 steps
                    for _ in range(self.train_per_step):
                        self.train()
                        # Get the latest loss and Q values
                        if hasattr(self, 'last_loss'):
                            episode_losses.append(self.last_loss)
                        if hasattr(self, 'last_q_values'):
                            episode_q_values.extend(self.last_q_values.tolist())

                state = next_state
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
                        "Current Reward": reward,
                        "Running Average Reward": total_reward / (step_count + 1),
                        "Memory Size": len(self.memory),
                        "Episode Progress": step_count / self.max_episode_steps
                    })
            
            # Log episode summary
            avg_episode_q = np.mean(episode_q_values) if episode_q_values else 0.0
            avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Average Episode Q": avg_episode_q,
                "Average Episode Loss": avg_episode_loss,
                "Episode Length": step_count,
                "Memory Size": len(self.memory)
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
                    "Evaluation Episode": ep
                })

    def evaluate(self):
        # Set network to evaluation mode
        self.q_net.eval()
        
        obs, _ = self.test_env.reset()
        
        if self.is_atari:
            state = self.preprocessor.reset(obs)
        else:
            state = obs
            
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
            
            if self.is_atari:
                next_state = self.preprocessor.step(next_obs)
            else:
                next_state = next_obs
                
            # Calculate evaluation loss
            with torch.no_grad():
                next_q_values = self.target_net(torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(self.device))
                target_q_value = reward + (1 - float(done)) * self.gamma * next_q_values.max().item()
                current_q_value = q_values[0, action].item()
                eval_loss = F.smooth_l1_loss(torch.tensor([current_q_value]), torch.tensor([target_q_value]))
                eval_losses.append(eval_loss.item())
            
            state = next_state
            total_reward += reward
        
        # Set network back to training mode
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
            "Total Env Steps": self.env_count
        })
        
        return total_reward


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilon-greedy exploration
        self.epsilon = max(self.epsilon_min, self.epsilon_start - (self.epsilon_start - self.epsilon_min) * self.env_count / self.eps_decay_steps)
            
        self.train_count += 1
        
        transitions = self.memory.sample(self.batch_size)
        if transitions is None:
            return
        is_weights = None
            
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)/255.0
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)/255.0
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Standard DQN
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        td_errors = target_q_values - q_values
        mse_loss = F.mse_loss(q_values, target_q_values, reduction='mean')
        rms_loss = torch.sqrt(mse_loss + 1e-8)  # add small epsilon for numerical stability
        
        # Store for episode summary
        self.last_loss = rms_loss.item()
        self.last_q_values = q_values.detach().cpu()
        
        # Optimize the model with gradient clipping
        self.optimizer.zero_grad()
        # loss.backward()
        rms_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update learning rate
        if self.train_count < self.replay_start_size:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"[Train #{self.train_count}] Total Env Steps: {self.env_count} Loss: {rms_loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f} LR: {current_lr:.6f}")
            wandb.log({
                "Train Loss": rms_loss.item(),
                "Train Q Mean": q_values.mean().item(),
                "Train Q Std": q_values.std().item(),
                "Learning Rate": current_lr,
                "Train Count": self.train_count,
                "Total Env Steps": self.env_count,
                "TD Error Mean": td_errors.mean().item(),
                "TD Error Std": td_errors.std().item(),
                "Target Q Mean": target_q_values.mean().item(),
                "Target Q Std": target_q_values.std().item()
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--epsilon-min", type=float, default=0.01)  
    parser.add_argument("--eps-decay-steps", type=int, default=1000000)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--env-name", type=str, default="ALE/Breakout-v5", help="Name of the environment to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility (default: 1234)")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    # Set global seed
    set_seed(args.seed)
    
    if args.env_name == "CartPole-v1":
        wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    elif args.env_name == "ALE/Pong-v5":
        wandb.init(project="DLP-Lab5-DQN-Pong-vanilla", name=args.wandb_run_name, save_code=True)
    
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.episodes)
