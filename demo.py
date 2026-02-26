"""
demo.py

Standalone demo — loads a trained model and runs it live in a pygame window.
Owns all pygame initialization itself rather than relying on the env's lazy
init, which avoids the "video system not initialized" error.

Compatible with all supported tasks: sort_by_size, group_by_color, cluster.

Usage:
    python demo.py --model models/shape_agent/best_model
    python demo.py --model models/shape_agent/best_model --prompt "sort right to left"
    python demo.py --model models/bc_agent/bc_model --prompt "group shapes by color"
    python demo.py --random     # watch a random agent (no model needed)
"""

import argparse
import sys
import numpy as np
import pygame

from shape_env import ShapeEnv, WINDOW_W, WINDOW_H, FPS, BG_COLOR
from llm_goal_parser import parse_goal


def draw_env(surface, env, font):
    """Draw current env state onto an existing pygame surface."""
    surface.fill(BG_COLOR)

    # Ghost circles at target positions
    for i, (tx, ty) in enumerate(env.target_pos):
        pygame.draw.circle(surface, (80, 80, 80),
                           (int(tx), int(ty)),
                           env.shapes[i].radius, 2)

    for shape in env.shapes:
        shape.draw(surface, font)

    # HUD — build goal string that works for all task types
    goal     = env.goal
    task     = goal["task"]
    score    = env._compute_score()
    rank     = env._compute_rank_corr()

    if task == "sort_by_size":
        goal_str = (f"task: {task} | axis: {goal['axis']} | "
                    f"dir: {goal['direction']}   "
                    f"progress: {score:.2%}   sort: {rank:+.2f}")
    else:
        goal_str = (f"task: {task} | attr: {goal.get('attribute','?')}   "
                    f"progress: {score:.2%}   cohesion: {rank:.2f}")

    surface.blit(font.render(goal_str, True, (200, 200, 200)), (10, 10))


def run_demo(model_path: str, prompt: str, use_random: bool):
    # Pygame must be initialised here, before ShapeEnv, to own the display
    pygame.init()
    window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("shape manipulation — demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 12)

    # Parse goal
    goal = parse_goal(prompt)
    print(f"goal: {goal}")

    # Env (render_mode=None — we draw manually so we control the loop)
    env    = ShapeEnv(n_shapes=2, goal=goal, render_mode=None)
    obs, _ = env.reset()

    # Load model
    if use_random:
        print("running random agent (no model loaded)")
        model = None
    else:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"loaded model from {model_path}")
        except Exception as e:
            print(f"could not load model: {e}")
            pygame.quit()
            sys.exit(1)

    total_reward = 0.0
    steps        = 0
    episode      = 1
    print(f"\nepisode {episode} — close window or press Q to quit\n")

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if not running:
            break

        # Agent picks action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps        += 1

        # Draw
        draw_env(window, env, font)

        # Highlight the shape that was just moved
        shape_idx = int(np.clip(
            round((action[0] + 1.0) / 2.0 * (env.n_shapes - 1)),
            0, env.n_shapes - 1
        ))
        s = env.shapes[shape_idx]
        pygame.draw.circle(window, (255, 255, 255),
                           (int(s.x), int(s.y)), s.radius + 4, 2)

        # Step / episode overlay
        status_str = "SOLVED!" if terminated else ""
        step_label = font.render(
            f"episode {episode}   step {steps}   {status_str}",
            True, (180, 180, 100)
        )
        window.blit(step_label, (10, WINDOW_H - 24))

        pygame.display.flip()
        clock.tick(FPS)

        # Episode reset
        if terminated or truncated:
            status = "SOLVED" if terminated else "timed out"
            print(f"episode {episode} {status} — "
                  f"steps: {steps}  total reward: {total_reward:.3f}")
            obs, _       = env.reset()
            total_reward = 0.0
            steps        = 0
            episode     += 1
            print(f"episode {episode}")

    pygame.quit()
    print("demo closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run a trained agent demo")
    parser.add_argument(
        "--model", type=str, default="models/shape_agent/best_model",
        help="path to saved SB3 model (without .zip extension)",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="sort the shapes from smallest to largest left to right",
        help="natural language goal prompt",
    )
    parser.add_argument(
        "--random", action="store_true",
        help="use a random agent instead of a trained model",
    )
    args = parser.parse_args()
    run_demo(args.model, args.prompt, args.random)