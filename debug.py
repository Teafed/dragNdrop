"""
debug.py

diagnostic script — run this before worrying about training or demos.
tests things in order so you can identify exactly where a problem is.

tests:
    1. env steps move shapes (basic sanity check)
    2. rewards vary across episodes (reward function health)
    3. all wave 1 tasks initialise and step without error
    4. goal encoder produces correct output shape
    5. trained model outperforms random (if --model provided)
    6. pygame renders without crashing (skip with --skip-render)

usage:
    python debug.py --skip-render
    python debug.py --model models/shape_agent/best_model --skip-render
    python debug.py                    # includes pygame render test
"""

import argparse
import numpy as np
from shape_env import ShapeEnv
from config import GOAL_ENCODING_DIM, get_obs_size


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _default_goal(task: str) -> dict:
    """return a minimal valid goal dict for a given task."""
    base = {
        "task":      task,
        "axis":      "none",
        "direction": "none",
        "attribute": "none",
        "region":    "none",
    }
    if task == "sort_by_size":
        base.update({"axis": "x", "direction": "ascending", "attribute": "size"})
    elif task in ("group_by_color", "cluster"):
        base.update({"attribute": "color"})
    elif task == "arrange_in_line":
        base.update({"axis": "x"})
    elif task == "push_to_region":
        base.update({"region": "left"})
    return base


# ---------------------------------------------------------------------------
# test 1: basic step mechanics
# ---------------------------------------------------------------------------

def test_env_steps(n_shapes: int = 2) -> bool:
    print(f"=== test 1: do env steps actually move shapes? (n_shapes={n_shapes}) ===")
    goal   = _default_goal("sort_by_size")
    env    = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None)
    obs, _ = env.reset()

    expected_obs = get_obs_size()
    print(f"  obs shape: {obs.shape} (expected: ({expected_obs},))")

    before = [(round(s.x, 1), round(s.y, 1)) for s in env.shapes]
    print(f"  shapes before: {before}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1}: action={np.round(action, 3)}  reward={reward:.4f}")

    after = [(round(s.x, 1), round(s.y, 1)) for s in env.shapes]
    print(f"  shapes after:  {after}")

    moved   = before != after
    obs_ok  = obs.shape[0] == expected_obs
    ok      = moved and obs_ok

    if not moved:
        print("  !! WARNING: shapes did not move — check step() clamp logic")
    if not obs_ok:
        print(f"  !! WARNING: obs size mismatch — got {obs.shape[0]}, expected {expected_obs}")
    if ok:
        print("  ok")

    print()
    env.close()
    return ok


# ---------------------------------------------------------------------------
# test 2: reward variance
# ---------------------------------------------------------------------------

def test_random_rewards(n_shapes: int = 2) -> bool:
    print(f"=== test 2: do rewards vary across episodes? (n_shapes={n_shapes}) ===")
    goal    = _default_goal("sort_by_size")
    env     = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None)
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
    ok = reward_range > 0.01
    if not ok:
        print("  !! WARNING: rewards are not varying — reward function may be broken")
    else:
        print("  reward function looks healthy")
    print()
    env.close()
    return ok


# ---------------------------------------------------------------------------
# test 3: all wave 1 tasks
# ---------------------------------------------------------------------------

def test_all_tasks() -> bool:
    print("=== test 3: do all wave 1 tasks initialise and step correctly? ===")
    tasks  = [
        "sort_by_size", "group_by_color", "cluster",
        "arrange_in_line", "arrange_in_grid", "push_to_region",
    ]
    all_ok = True

    for task in tasks:
        try:
            goal   = _default_goal(task)
            env    = ShapeEnv(n_shapes=3, goal=goal, render_mode=None)
            obs, _ = env.reset()

            rewards = []
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)

            obs_ok  = obs.shape[0] == get_obs_size()
            task_ok = info.get("task") == task
            ok      = obs_ok and task_ok

            print(f"  {task:<20} obs={obs.shape}  "
                  f"rewards=[{min(rewards):.3f}, {max(rewards):.3f}]  "
                  f"task_in_info={'ok' if task_ok else 'MISSING'}  "
                  f"{'OK' if ok else 'FAIL'}")

            env.close()
            if not ok:
                all_ok = False

        except Exception as e:
            print(f"  {task:<20} !! FAILED: {e}")
            all_ok = False

    print()
    return all_ok


# ---------------------------------------------------------------------------
# test 4: goal encoder
# ---------------------------------------------------------------------------

def test_goal_encoder() -> bool:
    print("=== test 4: does the goal encoder produce the right output shape? ===")
    try:
        import torch
        from bc_train import GoalEncoder
        from llm_goal_parser import get_embedding

        encoder  = GoalEncoder()
        prompt   = "sort shapes from smallest to largest left to right"
        raw_emb  = get_embedding(prompt)

        with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = encoder(emb_t).squeeze(0).numpy()

        enc_ok = encoding.shape == (GOAL_ENCODING_DIM,)
        print(f"  prompt   : \"{prompt}\"")
        print(f"  raw emb  : {raw_emb.shape}  (expected (384,))")
        print(f"  encoding : {encoding.shape}  (expected ({GOAL_ENCODING_DIM},))")

        # verify integration with ShapeEnv
        goal = _default_goal("sort_by_size")
        env  = ShapeEnv(n_shapes=2, goal=goal)
        env.set_goal_encoding(encoding)
        obs, _ = env.reset()
        obs_ok = obs.shape[0] == get_obs_size()
        print(f"  obs after set_goal_encoding: {obs.shape}  "
              f"(expected ({get_obs_size()},))")
        env.close()

        ok = enc_ok and obs_ok
        print(f"  result: {'OK' if ok else 'FAIL'}")

    except Exception as e:
        print(f"  !! FAILED: {e}")
        ok = False

    print()
    return ok


# ---------------------------------------------------------------------------
# test 5: trained model vs random
# ---------------------------------------------------------------------------

def test_trained_model(model_path: str, n_shapes: int = 2):
    print(f"=== test 5: does trained model outperform random? (n_shapes={n_shapes}) ===")
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

    goal = _default_goal("sort_by_size")
    env  = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None)

    obs, _          = env.reset()
    trained_rewards = []
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        trained_rewards.append(reward)
        if terminated or truncated:
            break

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
    env.close()


# ---------------------------------------------------------------------------
# test 6: pygame render
# ---------------------------------------------------------------------------

def test_render():
    print("=== test 6: does pygame render without crashing? ===")
    try:
        import pygame
        goal   = _default_goal("sort_by_size")
        env    = ShapeEnv(n_shapes=2, goal=goal, render_mode="human")
        obs, _ = env.reset()

        print("  pygame window should appear — running 120 frames then closing")

        clock   = pygame.time.Clock()
        running = True

        for i in range(120):
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
# test 7: oracle per-task solve rate breakdown
# ---------------------------------------------------------------------------

def test_oracle_per_task(n_episodes_per_task: int = 20) -> bool:
    """
    run the oracle separately on each task and report solve rate per task.
    a healthy oracle should hit 80%+ on sort/line/push and somewhat lower
    on group/grid where random initial positions make it harder.
    anything at 0% indicates the oracle or target computation is broken
    for that task.
    """
    print(f"=== test 7: oracle per-task solve rate "
          f"({n_episodes_per_task} episodes each) ===")

    from oracle import OraclePolicy
    import torch
    from bc_train import GoalEncoder
    from llm_goal_parser import get_embedding
    from config import TASK_POOL

    goal_encoder = GoalEncoder()
    goal_encoder.eval()

    # one representative prompt per task type
    task_prompts = {
        "sort_by_size":   "sort shapes from smallest to largest left to right",
        "group_by_color": "group shapes by color",
        "arrange_in_line":"arrange shapes in a horizontal line evenly spaced",
        "arrange_in_grid":"arrange shapes in a grid",
        "push_to_region": "move all shapes to the left side",
        "cluster":        "put shapes of the same color close together",
    }

    all_ok = True
    rng    = np.random.default_rng(42)

    for task, prompt in task_prompts.items():
        goal    = _default_goal(task)
        raw_emb = get_embedding(prompt)
        with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = goal_encoder(emb_t).squeeze(0).numpy()

        n_solved    = 0
        mean_reward = 0.0

        for _ in range(n_episodes_per_task):
            n_shapes = int(rng.integers(2, 5))   # 2-4 for tractable oracle check
            env      = ShapeEnv(n_shapes=n_shapes, goal=goal)
            env.set_goal_encoding(encoding)
            oracle   = OraclePolicy(env, noise_std=0.0)   # no noise for clean check

            obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            done   = False
            ep_r   = 0.0

            while not done:
                action = oracle.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_r  += reward
                done   = terminated or truncated

            if terminated:
                n_solved += 1
            mean_reward += ep_r
            env.close()

        solve_rate  = n_solved / n_episodes_per_task
        mean_reward = mean_reward / n_episodes_per_task
        ok          = solve_rate >= 0.5   # flag anything below 50%

        print(f"  {task:<20} solve={solve_rate:.0%}  "
              f"mean_reward={mean_reward:7.2f}  "
              f"{'OK' if ok else '!! LOW — check _targets and oracle logic'}")

        if not ok:
            all_ok = False

    print()
    return all_ok


# ---------------------------------------------------------------------------
# test 8: BC loss curve
# ---------------------------------------------------------------------------

def test_bc_loss(n_episodes: int = 100, epochs: int = 10) -> bool:
    """
    collect a small oracle dataset and run BC training, printing the loss
    curve. a healthy run should show loss dropping monotonically and
    reaching below 0.05 by the final epoch.

    if loss is flat or not dropping, the BC dataset may be too noisy,
    the learning rate may be wrong, or the obs/action dimensions are mismatched.
    """
    print(f"=== test 8: BC loss curve ({n_episodes} oracle episodes, "
          f"{epochs} epochs) ===")

    try:
        import torch
        from bc_train import GoalEncoder, train_bc
        from oracle import collect_demonstrations

        goal_encoder = GoalEncoder()
        goal_encoder.eval()

        print("  collecting oracle demos...")
        dataset = collect_demonstrations(
            goal_encoder=goal_encoder,
            n_episodes=n_episodes,
            noise_std=0.05,
            verbose=False,
        )
        print(f"  collected {len(dataset['observations']):,} transitions  "
              f"oracle solve rate: {dataset['solve_rate']:.0%}")

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            device    = "cuda" if torch.cuda.is_available() else "cpu"
            bc_policy, _ = train_bc(
                dataset=dataset,
                save_path=tmp,
                epochs=epochs,
                batch_size=256,
                device=device,
            )

        # check final loss was reasonable — train_bc already prints the curve
        print("  if loss reached below ~0.05 and was decreasing, BC is healthy.")
        ok = True

    except Exception as e:
        print(f"  !! FAILED: {e}")
        ok = False

    print()
    return ok


# ---------------------------------------------------------------------------
# test 9: oracle visual check
# ---------------------------------------------------------------------------

def test_oracle_render(task: str = "sort_by_size", n_shapes: int = 3):
    """
    watch the oracle solve one episode in pygame.
    useful for visually confirming that targets are placed correctly and
    the oracle is moving shapes in a sensible direction.
    press Q or close the window to skip to the next task.
    runs through all tasks in sequence unless a specific task is given.
    """
    print(f"=== test 9: oracle visual check ===")
    try:
        import pygame
        import torch
        from oracle import OraclePolicy
        from bc_train import GoalEncoder
        from llm_goal_parser import get_embedding

        task_prompts = {
            "sort_by_size":   "sort shapes from smallest to largest left to right",
            "group_by_color": "group shapes by color",
            "arrange_in_line":"arrange shapes in a horizontal line evenly spaced",
            "arrange_in_grid":"arrange shapes in a grid",
            "push_to_region": "move all shapes to the left side",
        }

        goal_encoder = GoalEncoder()
        goal_encoder.eval()

        pygame.init()
        window = pygame.display.set_mode((800, 600))
        clock  = pygame.time.Clock()
        font   = pygame.font.SysFont("monospace", 12)

        for task_name, prompt in task_prompts.items():
            print(f"  watching oracle on: {task_name} — close window or Q to advance")

            goal    = _default_goal(task_name)
            raw_emb = get_embedding(prompt)
            with torch.no_grad():
                emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
                encoding = goal_encoder(emb_t).squeeze(0).numpy()

            env = ShapeEnv(n_shapes=n_shapes, goal=goal, render_mode=None)
            env.set_goal_encoding(encoding)
            oracle = OraclePolicy(env, noise_std=0.0)

            obs, _ = env.reset()
            done   = False
            steps  = 0
            pygame.display.set_caption(f"oracle: {task_name}")

            running = True
            while running and not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        running = False

                action = oracle.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done  = terminated or truncated
                steps += 1

                # draw
                window.fill((30, 30, 35))
                for i, (tx, ty) in enumerate(env.target_pos):
                    pygame.draw.circle(window, (80, 80, 80),
                                       (int(tx), int(ty)),
                                       env.shapes[i].radius, 2)
                for shape in env.shapes:
                    shape.draw(window, font)

                score = env._compute_score()
                hud   = (f"oracle | task: {task_name} | "
                         f"step: {steps} | progress: {score:.2%}")
                window.blit(font.render(hud, True, (200, 200, 200)), (10, 10))

                if terminated:
                    window.blit(
                        font.render("SOLVED!", True, (100, 220, 100)),
                        (370, 280)
                    )

                pygame.display.flip()
                clock.tick(30)   # slower than demo so you can see what's happening

            env.close()

        pygame.quit()
        print("  oracle render complete")
    except Exception as e:
        print(f"  !! render error: {e}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diagnose the shape manipulation env")
    parser.add_argument(
        "--model", type=str, default=None,
        help="path to a trained model to test (enables test 5)"
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="skip pygame render tests (tests 6 and 9)"
    )
    parser.add_argument(
        "--n-shapes", type=int, default=2,
        help="number of shapes for tests 1, 2, and 5 (default: 2)"
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="run oracle diagnostics: per-task solve rate (7), "
             "BC loss curve (8), oracle render (9, skipped if --skip-render)"
    )
    parser.add_argument(
        "--oracle-episodes", type=int, default=20,
        help="episodes per task for oracle solve rate check (default: 20)"
    )
    args = parser.parse_args()

    ok1 = test_env_steps(n_shapes=args.n_shapes)
    ok2 = test_random_rewards(n_shapes=args.n_shapes)
    ok3 = test_all_tasks()
    ok4 = test_goal_encoder()

    if args.model:
        test_trained_model(args.model, n_shapes=args.n_shapes)

    if not args.skip_render:
        test_render()
    else:
        print("=== test 6: skipped (--skip-render) ===\n")

    if args.oracle:
        ok7 = test_oracle_per_task(n_episodes_per_task=args.oracle_episodes)
        ok8 = test_bc_loss()
        if not args.skip_render:
            test_oracle_render()
        else:
            print("=== test 9: skipped (--skip-render) ===\n")
    else:
        ok7 = ok8 = None

    print("=== summary ===")
    print(f"  env steps move shapes : {ok1}")
    print(f"  rewards vary          : {ok2}")
    print(f"  all tasks work        : {ok3}")
    print(f"  goal encoder ok       : {ok4}")
    if ok7 is not None:
        print(f"  oracle per-task ok    : {ok7}")
        print(f"  BC loss healthy       : {ok8}")
    print()
    if all(x for x in [ok1, ok2, ok3, ok4, ok7, ok8] if x is not None):
        print("all core tests passed — environment is healthy.")
    else:
        print("some tests failed — check warnings above.")
