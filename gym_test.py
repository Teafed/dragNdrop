import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
print("Total reward:", total_reward)
env.close()
