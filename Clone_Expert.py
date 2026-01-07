import numpy as np
import torch
import torch.nn as nn
import gymnasium_robotics
import gymnasium as gym
from gymnasium import spaces
from torch.utils.data import DataLoader

import minari

from Transition_Dataset import TransitionDataset
from Policy_Network import PolicyNetwork

# Train a policy network from the expert dataset using behavior cloning

# BC policy version
VERSION = 2

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