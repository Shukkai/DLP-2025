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

        # Calculate feature size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            feature_size = self.feature(sample).view(1, -1).size(1)

        self.fc_h_v = NoisyLinear(feature_size, 512)
        self.fc_h_a = NoisyLinear(feature_size, 512)
        self.fc_z_v = NoisyLinear(512, atom_size)
        self.fc_z_a = NoisyLinear(512, n_actions * atom_size)

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support.to(x.device), dim=2)
        return q

    def dist(self, x):
        x = x / 255.0  # Normalize pixel values
        x = self.feature(x)

        v = F.relu(self.fc_h_v(x))
        v = self.fc_z_v(v).view(-1, 1, self.atom_size)
        a = F.relu(self.fc_h_a(x))
        a = self.fc_z_a(a).view(-1, self.n_actions, self.atom_size)

        q_atoms = v + a - a.mean(1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # Prevent 0
        return dist

    def reset_noise(self):
        self.fc_h_v.reset_noise()
        self.fc_h_a.reset_noise()
        self.fc_z_v.reset_noise()
        self.fc_z_a.reset_noise()
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
        # return resized
        normalized = resized / 255.0    
        return normalized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
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

    if args.use_rainbow:    
        model = RainbowDQN(
            input_shape=(4, 84, 84),  # 4 stacked frames
            n_actions=num_actions,
            atom_size=51,
            v_min=-10,
            v_max=10
        ).to(device)
    else:
        if is_atari:    
            model = DQN_atari(
                input_channels=input_channels,
                num_actions=num_actions
            ).to(device)
        else:
            model = DQN(
                input_dim=input_dim,
                num_actions=num_actions
            ).to(device)
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
        if is_atari:
            # Random No-op steps
            no_op_steps = np.random.randint(0, 10)
            for _ in range(no_op_steps):
                obs, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    obs, _ = env.reset()
                state = preprocessor.reset(obs)
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
    parser.add_argument("--use-rainbow", type=bool, default=False, help="Use Rainbow model instead of DQN")
    args = parser.parse_args()
    evaluate(args)
