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
    python demo.py --model models/shape_agent/best_model --prompt "sort shapes smallest to largest"
    python demo.py --model models/shape_agent/best_model --prompt "group shapes by color"
    python demo.py --model models/shape_agent/best_model --prompt "move all shapes to the left side"
    python demo.py --oracle --prompt "arrange shapes in a horizontal line evenly spaced"
    python demo.py --oracle --sequential
    python demo.py --random
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

   for shape in env.shapes:
      shape.draw(surface, font)

   # draw cursor — borrow env's own draw method by temporarily
   # pointing its window at our surface
   _prev_window   = env.window
   env.window     = surface
   env._draw_cursor()
   env.window     = _prev_window

   task     = env.goal.get("task", "arrange_in_sequence")
   score    = env._compute_score()
   rank     = env._compute_rank_corr()
   goal_str = (f"task: {task}   progress: {score:.2%}   "
               f"rank/cohesion: {rank:+.2f}")
   surface.blit(font.render(goal_str, True, (200, 200, 200)), (10, 10))


def run_oracle_demo(sequential: bool, prompt: str = None):
    """
    watch the oracle solve episodes.

    --prompt:     use this prompt for every episode (oracle stays on one task).
    --sequential: cycle through TASK_POOL in order, ignoring --prompt.
    default:      random task each episode.
    """
    from oracle import OraclePolicy, IDLE_THRESHOLD
    import torch
    from bc_train import GoalEncoder

    pygame.init()
    window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("shape manipulation — oracle demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 12)

    goal_encoder = GoalEncoder()
    goal_encoder.eval()

    task_pool   = list(TASK_POOL)
    task_cursor = 0

    def next_prompt():
        nonlocal task_cursor
        if prompt is not None and not sequential:
            return prompt
        if sequential:
            p           = task_pool[task_cursor % len(task_pool)]
            task_cursor += 1
            return p
        return random.choice(task_pool)

    cur_prompt = next_prompt()
    goal       = parse_goal(cur_prompt)
    raw_emb    = get_embedding(cur_prompt)
    with torch.no_grad():
        emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
        encoding = goal_encoder(emb_t).squeeze(0).numpy()

    env    = ShapeEnv(goal=goal, render_mode=None)
    env.set_goal_encoding(encoding)
    oracle = OraclePolicy(env, noise_std=0.0)
    obs, _ = env.reset()

    episode      = 1
    steps        = 0
    total_reward = 0.0
    paused       = False
    print(f"episode {episode} — {cur_prompt}  (Q quit  N next  SPACE pause  D dump state)")

    # keybind legend — rendered once per frame in top-right corner
    legend_lines = [
        "Q  quit",
        "N  next episode",
        "SPC  pause / unpause",
        "D  dump state to console",
    ]

    def draw_legend(surface, font):
        x = WINDOW_W - 160
        for i, line in enumerate(legend_lines):
            surface.blit(font.render(line, True, (140, 140, 140)), (x, 28 + i * 14))

    def dump_state(env, oracle, step, score):
        print(f"\n--- state dump  ep {episode}  step {step} ---")
        print(f"  task        : {env.goal.get('task')}  score={score:.4f}  "
              f"idle_threshold={IDLE_THRESHOLD:.2f}")
        print(f"  oracle      : phase={oracle.phase}  "
              f"committed_shape={oracle.committed_shape}  "
              f"committed_target={oracle.committed_target}")
        if hasattr(oracle, '_group_zones') and oracle._group_zones:
            print(f"  group zones : {oracle._group_zones}")
        print(f"  cursor      : ({env.cx:.1f}, {env.cy:.1f})  "
              f"holding={env.holding}  grabbed_idx={env.grabbed_idx}")
        for i, s in enumerate(env.shapes):
            attr = getattr(s, 'color_name', '?')
            print(f"  shape {i}     : ({s.x:.1f}, {s.y:.1f})  "
                  f"color={attr}  size={s.size:.1f}  type={s.shape_type}")
        print()

    running = True
    while running:
        skip = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_n:
                    skip = True
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"[demo] {'paused' if paused else 'resumed'}  "
                          f"ep {episode}  step {steps}")
                if event.key == pygame.K_d:
                    dump_state(env, oracle, steps, env._compute_score())

        if not running:
            break

        if not paused and not skip:
            action                                   = oracle.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1
        else:
            # still need info for the HUD when paused
            terminated = truncated = False
            info       = {"score": env._compute_score(), "task": env.goal.get("task")}

        draw_env(window, env, font)
        draw_legend(window, font)

        # bottom bar: oracle phase + idle indicator
        score       = env._compute_score()
        idle_now    = score >= IDLE_THRESHOLD
        phase_str   = oracle.phase or "none"
        status_str  = "SOLVED!" if (not paused and terminated) else ""
        pause_str   = "  [PAUSED]" if paused else ""
        bottom_line = (
            f"ORACLE  ep {episode}  step {steps}  "
            f"score {score:.3f}  phase={phase_str}  "
            f"{'IDLE ' if idle_now else ''}"
            f"{status_str}{pause_str}"
        )
        window.blit(
            font.render(bottom_line, True,
                        (255, 220, 80) if paused else (100, 220, 180)),
            (10, WINDOW_H - 24)
        )
        pygame.display.flip()
        clock.tick(FPS)

        if not paused and (terminated or truncated or skip):
            status = "SOLVED" if terminated else ("skipped" if skip else "timed out")
            print(f"episode {episode} {status} — "
                  f"steps: {steps}  reward: {total_reward:.2f}  "
                  f"score: {info['score']:.3f}  task: {cur_prompt}")

            cur_prompt = next_prompt()
            goal       = parse_goal(cur_prompt)
            raw_emb    = get_embedding(cur_prompt)
            with torch.no_grad():
                emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
                encoding = goal_encoder(emb_t).squeeze(0).numpy()

            env.goal = goal
            env.set_goal_encoding(encoding)
            oracle   = OraclePolicy(env, noise_std=0.0)
            obs, _   = env.reset()

            episode     += 1
            steps        = 0
            total_reward = 0.0
            paused       = False
            print(f"episode {episode} — {cur_prompt}")

    pygame.quit()
    print("oracle demo closed")


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
    parser = argparse.ArgumentParser(description="run a trained agent or oracle demo")
    parser.add_argument(
        "--model", type=str, default="models/shape_agent/best_model",
        help="path to saved SB3 model (without .zip extension)",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="sort shapes from smallest to largest left to right",
        help="natural language goal prompt",
    )
    parser.add_argument(
        "--random", action="store_true",
        help="use a random agent instead of a trained model",
    )
    parser.add_argument(
        "--multi-task", action="store_true",
        help="sample a random task from TASK_POOL each episode",
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="watch the oracle instead of a trained model",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="cycle through TASK_POOL in order (use with --oracle or --multi-task)",
    )
    args = parser.parse_args()

    if args.oracle:
        run_oracle_demo(sequential=args.sequential, prompt=args.prompt)
    else:
        run_demo(args.model, args.prompt, args.random, args.multi_task)
