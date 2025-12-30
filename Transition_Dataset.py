from torch.utils.data import Dataset
import numpy as np
import torch

# Credit: Chat

class TransitionDataset(Dataset):
    """
    Flattens a Minari dataset of episodes into per-timestep transitions for BC.

    Each item is (x_t, a_t) where:
      x_t = concat(observation_t, desired_goal_t)  -> shape (28,)
      a_t = action_t                               -> shape (4,)
    """
    def __init__(self, minari_dataset):
        self.X = []
        self.A = []

        for ep in minari_dataset:
            obs = ep.observations     # dict of arrays over time
            acts = ep.actions         # array over time

            # Pull arrays
            obs_vec = np.asarray(obs["observation"])     # (T_obs, 25)
            goal_vec = np.asarray(obs["desired_goal"])   # (T_obs, 3)
            acts = np.asarray(acts)                      # (T_act, 4)

            # Align lengths: actions are usually length T, observations sometimes T+1
            T = min(len(acts), len(obs_vec), len(goal_vec))

            x = np.concatenate([obs_vec[:T], goal_vec[:T]], axis=1)  # (T, 28)
            a = acts[:T]                                             # (T, 4)

            self.X.append(torch.as_tensor(x, dtype=torch.float32))
            self.A.append(torch.as_tensor(a, dtype=torch.float32))

        self.X = torch.cat(self.X, dim=0)  # (N, 28)
        self.A = torch.cat(self.A, dim=0)  # (N, 4)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx]
