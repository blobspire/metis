import gymnasium as gym
import gymnasium_robotics
import numpy as np

from tqdm.auto import tqdm
import minari
from minari import DataCollector

from pathlib import Path
import json

# Create Expert model and then generate expert demonstrations. This will consist of hardcoded conditionals to guide arm movement.

# User level config
RUN_VERSION = 1 # Used to create unique dataset and log names

# Clip and cast action to float32
def a(action):
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    return action

# Create dir to store dataset
Path.home().joinpath(".minari", "datasets").mkdir(parents=True, exist_ok=True)

# Set up Fetch environment
gym.register_envs(gymnasium_robotics)
env = DataCollector(gym.make('FetchPickAndPlace-v4', max_episode_steps=300)) # Uses 'sparse' rewards by default

tolerance = 0.01 # Acceptable discrepancy between desired and actual location (to prevent constant motor activations)

total_episodes = 1_000
failure_count = 0 # Track the number of failures across all training
stage_name = "(0) Configuration" # Used for debugging / logging truncations
step_count = 0 # Track the step count within each episode. Declare outside for global logging

for i in tqdm(range(total_episodes)):
    obs, info = env.reset()
    step_count = 0 # Reset step count
    terminated = False
    truncated = False

    # Begin robot action stages 

    # (1) Open gripper
    stage_name = "(1) Open gripper"
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(a(action))
    step_count += 1

    # (2) Move arm above block
    if not terminated and not truncated:
        stage_name = "(2) Move arm above block"
        # size: [0.025 0.025 0.025]
        # Block is 0.05m x 0.05m x 0.05m
        ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
        block_pos = obs["achieved_goal"]
        done = False
        while not done:
            k = 1.5 # Scaling factor
            dx = block_pos[0] - ee_pos[0]
            dy = block_pos[1] - ee_pos[1]
            dz = (block_pos[2] + 0.05) - ee_pos[2]
            action = np.array([k*dx, k*dy, k*dz, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(a(action))
            step_count += 1
            # Update positions
            ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
            block_pos = obs["achieved_goal"]
            done = (abs(ee_pos[0] - block_pos[0]) < tolerance and abs(ee_pos[1] - block_pos[1]) < tolerance and abs(ee_pos[2] - (block_pos[2] + 0.05)) < tolerance) or terminated or truncated

    # (3) Move ee to block level
    if not terminated and not truncated:
        stage_name = "(3) Move ee to block level"
        ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
        block_pos = obs["achieved_goal"]
        done = False
        while not done:
            dz = block_pos[2] - ee_pos[2]
            k = 1.5 # Scaling factor
            action = np.array([0.0, 0.0, k*dz, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(a(action))
            step_count += 1
            # Update ee position (and block, though this is redundant in this env)
            ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
            block_pos = obs["achieved_goal"]
            done = (abs(ee_pos[2] - block_pos[2]) < tolerance) or terminated or truncated

    # (4) Close gripper to grab block
    if not terminated and not truncated:
        stage_name = "(4) Close gripper"
        action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(a(action))
        step_count += 1

    # (5) Move block to goal
    if not terminated and not truncated:
        stage_name = "(5) Move block to goal"
        done = False
        ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
        block_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]
        while not done:
            k = 1.5 # Scaling factor
            dx = goal_pos[0] - block_pos[0]
            dy = goal_pos[1] - block_pos[1]
            dz = goal_pos[2] - block_pos[2]
            action = np.array([k*dx, k*dy, k*dz, -0.1], dtype=np.float32) # -0.1 gripper force maintains grip on block

            obs, reward, terminated, truncated, info = env.step(a(action))
            # Update positions
            ee_pos = obs["observation"][0:3]
            block_pos = obs["achieved_goal"]
            step_count += 1
            # done = (abs(block_pos[0] - goal_pos[0]) < tolerance and abs(block_pos[1] - goal_pos[1]) < tolerance and abs(block_pos[2] - goal_pos[2]) < tolerance) or terminated or truncated
            done = info.get("is_success") > 0 or terminated or truncated
            # Can also use ee_pos, instead of block_pos for more realistic

    # Update failure count if failure occured
    if truncated:
        failure_count += 1

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

    log_path = Path(f"expert_collection_log_{RUN_VERSION}.jsonl")
    with log_path.open("a") as f:
        f.write(json.dumps(episode_log) + "\n")

print(f"total failure count: {failure_count}")

dataset = env.create_dataset(
    dataset_id = f"pickandplace/expert-v{RUN_VERSION}",
    algorithm_name="ExpertPolicy",
)

env.close()

# Display the locations of the ee, block, and goal
def DisplayPositions():
    block = obs["achieved_goal"]
    goal  = obs["desired_goal"]
    ee    = obs["observation"][0:3]

    print("EE:", ee)
    print("Block (achieved_goal):", block)
    print("Goal  (desired_goal) :", goal)