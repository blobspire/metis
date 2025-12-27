import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Create Expert model that will be used to generate expert demonstrations. This will consist of hardcoded conditionals to guide arm movement.

# Set up Fetch environment
gym.register_envs(gymnasium_robotics)
env = gym.make('FetchPickAndPlace-v4', render_mode='human') # Uses 'sparse' rewards by default

obs, info = env.reset()

# First, move arm above block
# size: [0.025 0.025 0.025]
# Block is 0.05m x 0.05m x 0.05m
done = False
while not done:
    ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
    block_pos = obs["achieved_goal"]

    action = np.array([0.0, 0.0, 0.0, 0.0])
    tolerance = 0.01 # Acceptable discrepancy between desired and actual location (to prevent constant motor activations)

    # Match x
    if ee_pos[0] < block_pos[0]: # Could add tolerance here
        # ee x is left of block x, move right
        action += np.array([0.1, 0, 0, 0])
    elif ee_pos[0] > block_pos[0]:
        # ee x is right of block x, move left
        action += np.array([-0.1, 0, 0, 0])
    # Match y
    if ee_pos[1] < block_pos[1]:
        # ee y is left of block y, move right
        action += np.array([0.0, 0.1, 0, 0])
    elif ee_pos[1] > block_pos[1]:
        # ee y is right of block y, move left
        action += np.array([0.0, -0.1, 0, 0])

    obs, reward, terminated, truncated, info = env.step(action)
    done = abs(ee_pos[0] - block_pos[0]) < tolerance and abs(ee_pos[1] - block_pos[1]) < tolerance

# Open gripper
action = np.array([0.0, 0.0, 0.0, 1.0])
obs, reward, terminated, truncated, info = env.step(action)

# Move ee to block level
ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
block_pos = obs["achieved_goal"]
while abs(ee_pos[2] - block_pos[2]) > 0.01:
    action = np.array([0.0, 0.0, -0.1, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    # Update ee position (and block, though this is redundant in this env)
    ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
    block_pos = obs["achieved_goal"]

# Close gripper to grab block
action = np.array([0.0, 0.0, 0.0, -1.0])
obs, reward, terminated, truncated, info = env.step(action)

# Move block to goal
done = False
while not done:
    ee_pos = obs["observation"][0:3] # [x, y, z]' in global coords
    block_pos = obs["achieved_goal"]
    goal_pos = obs["desired_goal"]

    action = np.array([0.0, 0.0, 0.0, -0.1])
    tolerance = 0.001 # Acceptable discrepancy between desired and actual location (to prevent constant motor activations)

    # Match x
    if block_pos[0] < goal_pos[0] - tolerance:
        # ee x is left of goal x, move right
        action += np.array([0.1, 0, 0, 0])
    elif block_pos[0] > goal_pos[0] + tolerance:
        # ee x is right of goal x, move left
        action += np.array([-0.1, 0, 0, 0])
    # Match y
    if block_pos[1] < goal_pos[1] - tolerance:
        # ee y is left of goal y, move right
        action += np.array([0.0, 0.1, 0, 0])
    elif block_pos[1] > goal_pos[1] + tolerance:
        # ee y is right of goal y, move left
        action += np.array([0.0, -0.1, 0, 0])
    # Move right above goal in z
    if block_pos[2] < goal_pos[2] - tolerance:
        # ee z is below goal z, move up
        action += np.array([0, 0, 0.1, 0])
    elif block_pos[2] > goal_pos[2] + tolerance:
        # ee z is above goal z, move down
        action += np.array([0, 0, -0.1, 0])

    obs, reward, terminated, truncated, info = env.step(action)
    done = abs(block_pos[0] - goal_pos[0]) < tolerance and abs(block_pos[1] - goal_pos[1]) < tolerance and abs(block_pos[2] - goal_pos[2]) < tolerance
    # Can also use ee_pos, instead of block_pos for more realistic

# Confirm success, distance, and reward
d = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
print("is_success:", info.get("is_success"), "dist:", d, "reward:", reward)

env.close()

# Display the locations of the ee, block, and goal
def DisplayPositions():
    block = obs["achieved_goal"]
    goal  = obs["desired_goal"]
    ee    = obs["observation"][0:3]

    print("EE:", ee)
    print("Block (achieved_goal):", block)
    print("Goal  (desired_goal) :", goal)