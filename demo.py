"""
demo.py

standalone demo — loads a trained model and runs it live in a pygame window.
owns all pygame initialisation itself rather than relying on the env's lazy
init, which avoids the "video system not initialized" error.

the goal encoder is loaded from the same directory as the model. if it's not
found, the encoding defaults to zeros (goal-blind policy — useful for sanity
checking that the model moves shapes at all).

usage:
    python demo.py --model models/shape_agent/best_model
    python demo.py --model models/shape_agent/best_model --prompt "sort right to left"
    python demo.py --model models/shape_agent/best_model --prompt "group shapes by color"
    python demo.py --model models/shape_agent/best_model --prompt "arrange shapes in a grid"
    python demo.py --random     # watch a random agent (no model needed)
"""

import argparse
import os
import random
import sys
import numpy as np
import torch
import pygame

from shape_env import ShapeEnv, WINDOW_W, WINDOW_H, FPS, BG_COLOR
from llm_goal_parser import parse_goal, get_embedding
from config import GOAL_ENCODING_DIM, TASK_POOL


def load_goal_encoder(model_path: str):
    """
    try to load the goal encoder saved alongside the model.
    returns a GoalEncoder instance (random init if file not found).
    """
    from bc_train import GoalEncoder

    encoder_path = os.path.join(os.path.dirname(model_path), "goal_encoder.pt")
    encoder      = GoalEncoder()

    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        print(f"goal encoder loaded from {encoder_path}")
    else:
        print(f"no goal encoder found at {encoder_path} — using zero encoding")

    encoder.eval()
    return encoder


def encode_goal(goal_encoder, prompt: str) -> np.ndarray:
    """project a prompt string to a GOAL_ENCODING_DIM vector."""
    raw_emb = get_embedding(prompt)
    with torch.no_grad():
        emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
        encoding = goal_encoder(emb_t).squeeze(0).numpy()
    return encoding


def draw_env(surface, env, font):
    """draw current env state onto an existing pygame surface."""
    surface.fill(BG_COLOR)

    # ghost circles at target positions
    for i, (tx, ty) in enumerate(env.target_pos):
        pygame.draw.circle(surface, (80, 80, 80),
                           (int(tx), int(ty)),
                           env.shapes[i].radius, 2)

    for shape in env.shapes:
        shape.draw(surface, font)

    goal      = env.goal
    task      = goal["task"]
    score     = env._compute_score()
    rank      = env._compute_rank_corr()
    goal_str  = (f"task: {task}   progress: {score:.2%}   "
                 f"rank/cohesion: {rank:+.2f}")
    surface.blit(font.render(goal_str, True, (200, 200, 200)), (10, 10))


def run_demo(model_path: str, prompt: str, use_random: bool, multi_task: bool):
    pygame.init()
    window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("shape manipulation — demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 12)

    if use_random:
        goal_encoder = None
        model        = None
        print("running random agent (no model loaded)")
    else:
        goal_encoder = load_goal_encoder(model_path)
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"model loaded from {model_path}")
        except Exception as e:
            print(f"could not load model: {e}")
            pygame.quit()
            sys.exit(1)

    def sample_episode_goal():
        """sample a prompt, parse it, and compute its encoding."""
        p        = random.choice(TASK_POOL) if multi_task else prompt
        g        = parse_goal(p)
        if goal_encoder is not None:
            enc = encode_goal(goal_encoder, p)
        else:
            enc = np.zeros(GOAL_ENCODING_DIM, dtype=np.float32)
        return p, g, enc

    cur_prompt, goal, encoding = sample_episode_goal()
    print(f"goal: {goal}")

    env = ShapeEnv(goal=goal, render_mode=None)
    env.set_goal_encoding(encoding)
    obs, _ = env.reset()

    total_reward = 0.0
    steps        = 0
    episode      = 1
    print(f"\nepisode {episode} — {cur_prompt}")
    print("close window or press Q to quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if not running:
            break

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps        += 1

        draw_env(window, env, font)

        shape_idx = int(np.clip(
            round((action[0] + 1.0) / 2.0 * (env.n_shapes - 1)),
            0, env.n_shapes - 1
        ))
        s = env.shapes[shape_idx]
        pygame.draw.circle(window, (255, 255, 255),
                           (int(s.x), int(s.y)), s.radius + 4, 2)

        status_str = "SOLVED!" if terminated else ""
        step_label = font.render(
            f"ep {episode}  step {steps}  n_shapes {env.n_shapes}  "
            f"{cur_prompt}  {status_str}",
            True, (180, 180, 100)
        )
        window.blit(step_label, (10, WINDOW_H - 24))

        pygame.display.flip()
        clock.tick(FPS)

        if terminated or truncated:
            status = "SOLVED" if terminated else "timed out"
            print(f"episode {episode} {status} — "
                  f"steps: {steps}  reward: {total_reward:.3f}  "
                  f"n_shapes: {env.n_shapes}  task: {cur_prompt}")

            # sample new task and reset env for next episode
            cur_prompt, goal, encoding = sample_episode_goal()
            env.goal = goal
            env.set_goal_encoding(encoding)
            obs, _       = env.reset()
            total_reward = 0.0
            steps        = 0
            episode     += 1
            print(f"episode {episode} — {cur_prompt}")

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
    parser.add_argument(
        "--multi-task", action="store_true",
        help="sample a random task from TASK_POOL each episode instead of using --prompt",
    )
    args = parser.parse_args()
    run_demo(args.model, args.prompt, args.random, args.multi_task)
