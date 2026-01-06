from torch.utils.data import Dataset
import numpy as np
import torch

# Convert minari episodes to dataset of transitions (state and corresponding action at each timestep)

class TransitionDataset(Dataset):

    def __init__(self, minari_dataset):
        # X (state) is (T, 31) and A (action) is (T, 4)
        X_list, A_list = [], []

        for ep in minari_dataset:
            obs = ep.observations
            acts = np.asarray(ep.actions) # (dx, dy, dz, gripper)

            obs_vec = np.asarray(obs["observation"]) # Entire observation vector
            ag = np.asarray(obs["achieved_goal"]) # Block position
            dg = np.asarray(obs["desired_goal"]) # Goal position

            # Ensure all arrays have same length
            T = min(len(acts), len(obs_vec), len(ag), len(dg))

            # Extract the vectors (for direction and magnitude)
            grip = obs_vec[:T, 0:3]
            goal_delta = dg[:T] - ag[:T]
            obj_delta  = ag[:T] - grip

            x = np.concatenate([obs_vec[:T], goal_delta, obj_delta], axis=1)  # (T, 31)
            a = acts[:T]                                                      # (T, 4)

            # Convert to torch tensors
            X_list.append(torch.as_tensor(x, dtype=torch.float32))
            A_list.append(torch.as_tensor(a, dtype=torch.float32))

        # Flatten into transition dataset
        self.X = torch.cat(X_list, dim=0)  # (N, 31)
        self.A = torch.cat(A_list, dim=0)  # (N, 4)

    # Number of episodes
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx]