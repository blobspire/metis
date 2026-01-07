import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

from Policy_Network import PolicyNetwork

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

# Calculate the success rate of the policy
def eval_policy(policy_path, n_episodes=50, render=False, device="cpu"):
    print("Begin policy evaluation...")
    gym.register_envs(gymnasium_robotics)

    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=300,
                   render_mode="human" if render else None)
    
    # Input will be 25 (observation) + 3 (goal delta) + 3 (object delta) = 31
    input_dim = 31
    # Output will be 4 (x, y, z, gripper)
    output_dim = env.action_space.shape[0]

    policy = PolicyNetwork(input_dim, output_dim).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

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

                action = policy(x_t).squeeze(0).cpu().numpy().astype(np.float32) # (4,)
                # Clip action to [-1, 1]
                action = np.clip(action, -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(action)

                steps += 1
                done = terminated or truncated or (info.get("is_success", 0.0) > 0)

            successes += int(info.get("is_success", 0.0) > 0)
            lengths.append(steps)

    env.close()
    policy.train()

    success_rate = successes / n_episodes
    print(f"BC eval: success_rate={success_rate:.3f}, avg_steps={np.mean(lengths):.1f}")
    return success_rate

POLICY_PATH = "bc_policy_v2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test the policy network
eval_policy(policy_path=POLICY_PATH, n_episodes=100, render=False, device=device)
eval_policy(policy_path=POLICY_PATH, n_episodes=5, render=True, device=device)