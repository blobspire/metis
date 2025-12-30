from torch.utils.data import Dataset
import numpy as np
import torch

# Credit: Chat

class TransitionDataset(Dataset):
    """
    Each item is (x_t, a_t) where:
      x_t = concat(
            observation_t (25),
            goal_delta_t  = desired_goal - achieved_goal (3),
            obj_delta_t   = achieved_goal - gripper_pos  (3)
          ) -> shape (31,)
      a_t = action_t -> shape (4,)
    """
    def __init__(self, minari_dataset):
        X_list, A_list = [], []

        for ep in minari_dataset:
            obs = ep.observations
            acts = np.asarray(ep.actions)  # (T_act, 4)

            obs_vec = np.asarray(obs["observation"])      # (T_obs, 25)
            ag = np.asarray(obs["achieved_goal"])         # (T_obs, 3)
            dg = np.asarray(obs["desired_goal"])          # (T_obs, 3)

            T = min(len(acts), len(obs_vec), len(ag), len(dg))

            grip = obs_vec[:T, 0:3]          # (T, 3)
            goal_delta = dg[:T] - ag[:T]     # (T, 3)
            obj_delta  = ag[:T] - grip       # (T, 3)

            x = np.concatenate([obs_vec[:T], goal_delta, obj_delta], axis=1)  # (T, 31)
            a = acts[:T]                                                      # (T, 4)

            X_list.append(torch.as_tensor(x, dtype=torch.float32))
            A_list.append(torch.as_tensor(a, dtype=torch.float32))

        self.X = torch.cat(X_list, dim=0)  # (N, 31)
        self.A = torch.cat(A_list, dim=0)  # (N, 4)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx]