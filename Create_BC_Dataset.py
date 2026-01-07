import numpy as np
import gymnasium as gym
import gymnasium_robotics
import torch
from minari import DataCollector
from tqdm.auto import tqdm
from pathlib import Path
from Policy_Network import PolicyNetwork
import json

# Create minari dataset from BC rollouts

# Clip and cast action to float32
def a(action):
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    return action

# Use goal and object delta vectors, rather than pure coordinates, to improve learning
def make_features(obs):
    obs_vec = obs["observation"] # (25,)
    ag = obs["achieved_goal"] # (3,)
    dg = obs["desired_goal"] # (3,)s
    grip = obs_vec[0:3] # (3,)

    goal_delta = dg - ag # (3,)
    obj_delta  = ag - grip # (3,)

    x = np.concatenate([obs_vec, goal_delta, obj_delta], axis=0) # (31,)
    return x

# User level config
RUN_VERSION = 1
POLICY_PATH = "bc_policy_v2.pt"
DATASET_ID = f"pickandplace/bc-rollouts-v{RUN_VERSION}"
N_EPISODES = 3000
MAX_STEPS = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dir to store dataset
Path.home().joinpath(".minari", "datasets").mkdir(parents=True, exist_ok=True)

# Set up Fetch environment
gym.register_envs(gymnasium_robotics)
env = DataCollector(gym.make('FetchPickAndPlace-v4', max_episode_steps=MAX_STEPS)) # Uses 'sparse' rewards by default

# Input will be 25 (observation) + 3 (goal delta) + 3 (object delta) = 31
input_dim = 31
# Output will be 4 (x, y, z, gripper)
output_dim = env.action_space.shape[0]

# Load policy
policy = PolicyNetwork(input_dim, output_dim).to(device)
policy.load_state_dict(torch.load(POLICY_PATH, map_location=device))
policy.eval()

successes = 0 # Track the number of successes across all episodes
failure_count = 0

# Collect rollouts
with torch.no_grad():
    for i in tqdm(range(N_EPISODES)):
        obs, info = env.reset()
        done = False

        truncated = False
        terminated = False
        stage_name = "" # We're using policy now, rather than expert, so this is unknown / fluid
        step_count = 0

        while not done:
            x = make_features(obs)
            x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0) # (1, 31)

            action = policy(x_t).squeeze(0).cpu().numpy().astype(np.float32) # (4,)
            # Clip action to [-1, 1]
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(a(action))

            step_count += 1
            done = terminated or truncated or (info.get("is_success", 0.0) > 0)

        # Update failure count if failure occured
        if truncated:
            failure_count += 1

        successes += int(info.get("is_success", 0.0) > 0)

        # Log episode stats to json
        episode_log = {
        "episode": i,
        "success": float(info.get("is_success")),
        "terminated": terminated,
        "truncated": truncated,
        "steps": step_count,
        "end_stage": stage_name,
        "total_failures": failure_count
        }

        log_path = Path(f"bc_collection_log_{RUN_VERSION}.jsonl")
        with log_path.open("a") as f:
            f.write(json.dumps(episode_log) + "\n")

print(f"Rollouts: success rate = {successes / N_EPISODES:.3f}")

# Create dataset from rollouts
dataset = env.create_dataset(
    dataset_id=DATASET_ID,
    algorithm_name="BCPolicyRollout",
)

env.close()
print("Dataset saved:", DATASET_ID)