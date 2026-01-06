import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

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
def eval_policy(policy_net, n_episodes=50, render=False, device="cpu"):
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