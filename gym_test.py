import gymnasium as gym
import time
import torch

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
done = False
total_reward = 0

max_angle = 0.2095    # ~12 degrees in radians
max_position = 2.4    # CartPole track limit
i = 0

while i < 1000:
    # Heuristic policy: account for both pole angle and cart position
    pole_angle = obs[2]
    cart_position = obs[0]

    # Adjust action: push in direction that reduces angle and keeps cart near center
    if pole_angle + 0.5 * cart_position < 0:
        action = 0  # push left
    else:
        action = 1  # push right

    obs, _, terminated, truncated, info = env.step(action)
    
    # Custom reward emphasizing pole angle and center position
    angle_reward = max(0, 1 - abs(obs[2]) / max_angle)
    position_reward = max(0, 1 - abs(obs[0]) / max_position)
    
    # Combine rewards, weight center position more strongly
    reward = angle_reward + 2 * position_reward  # position weighted more
    total_reward += reward
    
    done = terminated or truncated
    i += 1
    time.sleep(0.02)

print("Total custom reward:", total_reward)
env.close()
