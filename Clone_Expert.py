import numpy as np
import torch
import torch.nn as nn
import gymnasium_robotics
import gymnasium as gym
from gymnasium import spaces
from torch.utils.data import DataLoader

import minari

from Transition_Dataset import TransitionDataset

# Train a policy network from the expert dataset using behavior cloning

# BC policy version
VERSION = 2

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load expert dataset
minari_dataset = minari.load_dataset("pickandplace/expert-v1")

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space
assert isinstance(observation_space, spaces.Dict)
assert isinstance(action_space, spaces.Box)

# Input will be 25 (observation) + 3 (goal delta) + 3 (object delta) = 31
input_dim = 31
# Output will be 4 (x, y, z, gripper)
output_dim = action_space.shape[0]

policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
loss_fn = nn.SmoothL1Loss(beta=0.1) # Start with Huber loss

transition_dataset = TransitionDataset(minari_dataset)
dataloader = DataLoader(transition_dataset, batch_size=1024, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net.to(device)
policy_net.train()

for epoch in range(10):
    losses = []
    for x, a in dataloader:
        x = x.to(device)
        a = a.to(device)

        pred = policy_net(x)
        loss = loss_fn(pred, a)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) # Add gradient clipping for stability
        optimizer.step()

        losses.append(loss.item())

    print(f"epoch {epoch}: loss={np.mean(losses):.6f}")

# Save the policy network
torch.save(policy_net.state_dict(), f"bc_policy_v{VERSION}.pt")

# Test the policy network

# Use goal and object delta vectors, rather than pure coordinates, to improve learning
def make_features(obs):
    obs_vec = obs["observation"] # (25,)
    ag = obs["achieved_goal"] # (3,)
    dg = obs["desired_goal"] # (3,)
    grip = obs_vec[0:3] # (3,)

    goal_delta = dg - ag # (3,)
    obj_delta  = ag - grip # (3,)

    x = np.concatenate([obs_vec, goal_delta, obj_delta], axis=0) # (31,)
    return x

def eval_policy(policy_net, n_episodes=50, render=False):
    print("Begin policy evaluation...")
    gym.register_envs(gymnasium_robotics)

    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=300,
                   render_mode="human" if render else None)

    policy_net.eval()
    successes = 0
    lengths = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            steps = 0

            while not done:
                x = make_features(obs)
                x_t = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0) # (1, 31)

                action = policy_net(x_t).squeeze(0).cpu().numpy().astype(np.float32) # (4,)
                # Clip action to [-1, 1]
                action = np.clip(action, -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(action)

                steps += 1
                done = terminated or truncated or (info.get("is_success", 0.0) > 0)

            successes += int(info.get("is_success", 0.0) > 0)
            lengths.append(steps)

    env.close()
    policy_net.train()

    success_rate = successes / n_episodes
    print(f"BC eval: success_rate={success_rate:.3f}, avg_steps={np.mean(lengths):.1f}")
    return success_rate

# eval_policy(policy_net, n_episodes=500, render=False)
# eval_policy(policy_net, n_episodes=5, render=True)