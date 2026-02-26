"""
debug.py

Diagnostic script — run this before worrying about training or demos.
Tests things in order so you can identify exactly where a problem is.

Tests:
    1. Env steps move shapes (basic sanity check)
    2. Rewards vary across episodes (reward function health)
    3. Group task works correctly (new task sanity check)
    4. Trained model outperforms random (if --model provided)
    5. Pygame renders without crashing (skip with --skip-render)

Usage:
    python debug.py                                       # tests 1–3 + render
    python debug.py --model models/shape_agent/best_model # also runs test 4
    python debug.py --skip-render                         # headless / CI use

Changes from original:
    - test_env_steps uses n_shapes=2 (matches training default, not 4)
    - Added test_group_task to verify group_by_color task is wired up
    - Render test checks for QUIT events so the window can actually be closed
    - Fixed render test to call pygame.display.flip() so the window is visible
"""

import argparse
import numpy as np
from shape_env import ShapeEnv


# ---------------------------------------------------------------------------
# Test 1: Basic step mechanics
# ---------------------------------------------------------------------------

def test_env_steps() -> bool:
    print("=== test 1: do env steps actually move shapes? ===")
    env    = ShapeEnv(n_shapes=2, render_mode=None)   # 2 shapes matches training
    obs, _ = env.reset()

    before = [(round(s.x, 1), round(s.y, 1)) for s in env.shapes]
    print(f"  shapes before: {before}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1}: action={np.round(action, 3)}  reward={reward:.4f}")

    after = [(round(s.x, 1), round(s.y, 1)) for s in env.shapes]
    print(f"  shapes after:  {after}")

    moved = before != after
    print(f"  shapes moved: {moved}")
    if not moved:
        print("  !! WARNING: shapes did not move — check step() clamp logic")
    print()
    return moved


# ---------------------------------------------------------------------------
# Test 2: Reward variance
# ---------------------------------------------------------------------------

def test_random_rewards() -> bool:
    print("=== test 2: do rewards vary across episodes? ===")
    env     = ShapeEnv(n_shapes=2, render_mode=None)
    rewards = []

    for episode in range(5):
        obs, _    = env.reset()
        ep_reward = 0.0
        for _ in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"  episode {episode+1}: total reward = {ep_reward:.4f}")

    reward_range = max(rewards) - min(rewards)
    print(f"  reward range across episodes: {reward_range:.4f}")
    if reward_range < 0.01:
        print("  !! WARNING: rewards are not varying — reward function may be broken")
    else:
        print("  reward function looks healthy")
    print()
    return reward_range > 0.01


# ---------------------------------------------------------------------------
# Test 3: Group task
# ---------------------------------------------------------------------------

def test_group_task() -> bool:
    print("=== test 3: does group_by_color task work? ===")
    goal = {
        "task":      "group_by_color",
        "axis":      "none",
        "direction": "none",
        "attribute": "color",
    }

    try:
        env    = ShapeEnv(n_shapes=2, goal=goal, render_mode=None)
        obs, _ = env.reset()

        print(f"  obs shape: {obs.shape} (expected: {env.observation_space.shape})")
        print(f"  shapes: {[(s.color_name, round(s.x), round(s.y)) for s in env.shapes]}")
        print(f"  targets: {[(round(t[0]), round(t[1])) for t in env.target_pos]}")

        # Take a few steps and check rewards
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

        print(f"  reward range over 10 steps: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print(f"  rank_corr (cohesion): {info['rank_corr']:.3f}")
        print(f"  task in info: {info.get('task', 'MISSING')}")

        ok = (obs.shape == env.observation_space.shape
              and info.get("task") == "group_by_color")
        print(f"  group task OK: {ok}")
        env.close()
    except Exception as e:
        print(f"  !! group task FAILED: {e}")
        return False

    print()
    return ok


# ---------------------------------------------------------------------------
# Test 4: Trained model vs random
# ---------------------------------------------------------------------------

def test_trained_model(model_path: str):
    print(f"=== test 4: does trained model outperform random? ===")
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("  stable-baselines3 not installed, skipping")
        return

    try:
        model = PPO.load(model_path)
        print(f"  loaded model from {model_path}")
    except Exception as e:
        print(f"  could not load model: {e}")
        return

    env = ShapeEnv(n_shapes=2, render_mode=None)

    # Run trained model
    obs, _          = env.reset()
    trained_rewards = []
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        trained_rewards.append(reward)
        if terminated or truncated:
            break

    # Run random agent for same number of steps
    obs, _         = env.reset()
    random_rewards = []
    for _ in range(len(trained_rewards)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        random_rewards.append(reward)
        if terminated or truncated:
            break

    trained_mean = float(np.mean(trained_rewards))
    random_mean  = float(np.mean(random_rewards))
    print(f"  trained agent mean reward: {trained_mean:.4f}")
    print(f"  random  agent mean reward: {random_mean:.4f}")

    if trained_mean > random_mean:
        print("  trained model outperforms random — learning is working")
    else:
        print("  !! trained model not beating random — may need more training")
    print()


# ---------------------------------------------------------------------------
# Test 5: Pygame render
# ---------------------------------------------------------------------------

def test_render():
    print("=== test 5: does pygame render without crashing? ===")
    try:
        import pygame
        env    = ShapeEnv(n_shapes=2, render_mode="human")
        obs, _ = env.reset()

        print("  pygame window should appear — running 60 frames then closing")
        print("  close the window manually or wait for auto-close")

        clock   = pygame.time.Clock()
        running = True

        for i in range(120):   # 2 seconds at 60 fps
            # Handle QUIT so the window is actually closeable
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()

            if env.window is not None:
                pygame.display.flip()
            clock.tick(60)

            if i % 40 == 0:
                positions = [(round(s.x), round(s.y)) for s in env.shapes]
                print(f"  frame {i:3d}: positions={positions}  reward={reward:.4f}")

            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
        print("  render test passed")
    except Exception as e:
        print(f"  !! render error: {e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diagnose the shape manipulation env")
    parser.add_argument(
        "--model", type=str, default=None,
        help="path to a trained model .zip to test (enables test 4)"
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="skip the pygame render test (useful on headless machines / CI)"
    )
    args = parser.parse_args()

    ok1 = test_env_steps()
    ok2 = test_random_rewards()
    ok3 = test_group_task()

    if args.model:
        test_trained_model(args.model)

    if not args.skip_render:
        test_render()
    else:
        print("=== test 5: skipped (--skip-render) ===\n")

    print("=== summary ===")
    print(f"  env steps move shapes : {ok1}")
    print(f"  rewards vary          : {ok2}")
    print(f"  group task works      : {ok3}")
    print()
    if all([ok1, ok2, ok3]):
        print("all core tests passed — environment is healthy.")
    else:
        print("some tests failed — check warnings above.")