import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
import math
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.apply(init_weights)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.network(x)

class DQN_atari(nn.Module):
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
            
            # nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(128 * 7 * 7, 512),
            # nn.ReLU(),
            # nn.Linear(512, num_actions)
        )
        self.apply(init_weights)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.network(x)
    
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    is_atari = "ALE" in args.env_name or "Atari" in args.env_name or "Pong" in args.env_name
    if is_atari:
        env = gym.make(args.env_name, render_mode="rgb_array")
        preprocessor = AtariPreprocessor()
    else:
        env = gym.make(args.env_name, render_mode="rgb_array")
        preprocessor = None
    
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    num_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]  # For CartPole: 4
    input_channels = 4  # For Atari: 4 stacked frames

    if is_atari:
        model = DQN_atari(input_channels, num_actions).to(device)
    else:
        model = DQN(input_dim, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    total_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if is_atari:
            state = preprocessor.reset(obs)
        else:
            state = obs
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)
            frame_idx += 1

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if is_atari:
                state = preprocessor.step(next_obs)
            else:
                state = next_obs
            total_reward += reward

        total_rewards.append(total_reward)

        # Adjust frame size to be divisible by 16
        adjusted_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            if (h, w) != (new_h, new_w):
                frame = cv2.resize(frame, (new_w, new_h))
            adjusted_frames.append(frame)
            
        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
            for f in adjusted_frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")

    # Calculate and display statistics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    max_reward = np.max(total_rewards)
    min_reward = np.min(total_rewards)
    
    print("\nEvaluation Statistics:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Number of Episodes: {len(total_rewards)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--env-name", type=str, default="CartPole-v1", help="Environment name")
    args = parser.parse_args()
    evaluate(args)
