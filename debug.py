"""
debug.py

diagnostic script — run this before training or when something breaks.

tests:
   1. env steps move cursor and shapes respond to grip (basic sanity)
   2. rewards vary across episodes (reward function health)
   3. all tasks initialise and step without error (starter + wave 3)
   4. goal encoder produces correct output shape
   5. trained model outperforms random (if --model provided)
   6. pygame renders cursor correctly (skip with --skip-render)
   7. oracle per-task solve rate (--oracle flag)
   8. BC loss curve (--oracle flag)
   9. oracle visual check (--oracle flag, skip with --skip-render)

usage:
   python debug.py --skip-render
   python debug.py --oracle --skip-render
   python debug.py --model models/shape_agent/best_model --skip-render
   python debug.py                     # includes all render tests
"""

import argparse
import numpy as np
from shape_env import ShapeEnv, GRIP_RADIUS
from config import GOAL_ENCODING_DIM, get_obs_size


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _default_goal(task: str) -> dict:
   """return a minimal valid goal dict for any supported task."""
   base = {
      "task":      task,
      "axis":      "none",
      "direction": "none",
      "attribute": "none",
      "region":    "none",
      "bounded":   False,
   }
   if task == "reach":
      pass   # no extra fields needed
   elif task == "touch":
      pass
   elif task == "drag":
      base.update({"region": "left", "bounded": True})
   elif task == "arrange_in_sequence":
      base.update({"axis": "x", "direction": "ascending",
                   "attribute": "size", "bounded": False})
   elif task == "arrange_in_line":
      base.update({"axis": "x", "bounded": True})
   elif task == "arrange_in_region":
      base.update({"region": "left", "bounded": True})
   elif task == "arrange_in_groups":
      base.update({"attribute": "color", "bounded": True})
   return base


# ---------------------------------------------------------------------------
# test 1: basic step mechanics
# ---------------------------------------------------------------------------

def test_env_steps(n_shapes: int = 2) -> bool:
   """
   check that:
   - obs shape is 108
   - cursor moves when dx/dy are non-zero
   - grip attaches to a shape when cursor is close enough
   """
   print(f"=== test 1: cursor mechanics and obs shape (n_shapes={n_shapes}) ===")
   goal   = _default_goal("arrange_in_sequence")
   env    = ShapeEnv(n_shapes=n_shapes, goal=goal)
   obs, _ = env.reset()

   expected_obs = get_obs_size()
   obs_ok = obs.shape[0] == expected_obs
   print(f"  obs shape  : {obs.shape}  (expected ({expected_obs},))  "
         f"{'ok' if obs_ok else '!! FAIL'}")

   # move cursor right
   cx_before = env.cx
   env.step(np.array([1.0, 0.0, -1.0], dtype=np.float32))
   cursor_moved = env.cx > cx_before
   print(f"  cursor moved right : {cursor_moved}  "
         f"({cx_before:.1f} -> {env.cx:.1f})")

   # move cursor to first shape and grip
   s = env.shapes[0]
   env.cx = s.x
   env.cy = s.y
   _, _, _, _, _ = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
   grip_ok = env.holding and env.grabbed_idx == 0
   print(f"  grip when overlapping : {grip_ok}  "
         f"(holding={env.holding}  grabbed_idx={env.grabbed_idx})")

   # release grip
   env.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))
   release_ok = not env.holding and env.grabbed_idx == -1
   print(f"  release grip          : {release_ok}  "
         f"(holding={env.holding}  grabbed_idx={env.grabbed_idx})")

   # no grab when cursor is far from shapes
   env.cx = 0.0
   env.cy = 0.0
   env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
   no_grab_ok = env.grabbed_idx == -1
   print(f"  no grab when far      : {no_grab_ok}  "
         f"(grabbed_idx={env.grabbed_idx})")

   ok = obs_ok and cursor_moved and grip_ok and release_ok and no_grab_ok
   print(f"  result: {'ok' if ok else '!! FAIL'}")
   print()
   env.close()
   return ok


# ---------------------------------------------------------------------------
# test 2: reward variance
# ---------------------------------------------------------------------------

def test_random_rewards(n_shapes: int = 2) -> bool:
   """
   check that rewards vary meaningfully when shapes actually move.
   wave 3 tasks require grip-and-drag to produce score changes — pure
   random cursor movement never grips so reward stays flat at STEP_PENALTY.
   this test moves cursor to a shape, grips, then drags randomly.
   """
   print(f"=== test 2: reward variance (grip-and-drag actions) ===")
   all_ok = True

   test_cases = [
      ("reach",               1),
      ("drag",                1),
      ("arrange_in_region",   n_shapes),
      ("arrange_in_groups",   n_shapes),
   ]
   rng = np.random.default_rng(0)

   for task, n_shp in test_cases:
      goal    = _default_goal(task)
      env     = ShapeEnv(n_shapes=n_shp, goal=goal)
      rewards = []

      for _ in range(5):
         obs, _    = env.reset()
         ep_reward = 0.0

         # move cursor onto first shape then drag randomly
         s = env.shapes[0]
         env.cx = s.x
         env.cy = s.y

         for step in range(40):
            # grip on for first 20 steps, random dx/dy; release after
            grip  = 1.0 if step < 20 else -1.0
            dx    = float(rng.uniform(-1.0, 1.0))
            dy    = float(rng.uniform(-1.0, 1.0))
            action = np.array([dx, dy, grip], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
               break

         rewards.append(ep_reward)
      env.close()

      reward_range = max(rewards) - min(rewards)
      ok           = reward_range > 0.01
      status       = "ok" if ok else "!! FLAT — reward may be broken"
      print(f"  {task:<22} reward range: {reward_range:.4f}  {status}")
      if not ok:
         all_ok = False

   print()
   return all_ok


# ---------------------------------------------------------------------------
# test 3: all tasks initialise and step
# ---------------------------------------------------------------------------

def test_all_tasks() -> bool:
   print("=== test 3: all tasks initialise and step without error ===")
   tasks = [
      ("reach",               1),
      ("touch",               1),
      ("drag",                1),
      ("arrange_in_sequence", 3),
      ("arrange_in_line",     3),
      ("arrange_in_region",   3),
      ("arrange_in_groups",   3),
   ]
   all_ok = True

   for task, n_shp in tasks:
      try:
         goal   = _default_goal(task)
         env    = ShapeEnv(n_shapes=n_shp, goal=goal)
         obs, _ = env.reset()

         rewards = []
         for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

         obs_ok  = obs.shape[0] == get_obs_size()
         task_ok = info.get("task") == task
         ok      = obs_ok and task_ok

         print(f"  {task:<22} n={n_shp}  obs={obs.shape}  "
               f"rewards=[{min(rewards):.3f}, {max(rewards):.3f}]  "
               f"task_in_info={'ok' if task_ok else 'MISSING'}  "
               f"{'OK' if ok else 'FAIL'}")

         env.close()
         if not ok:
            all_ok = False

      except Exception as e:
         print(f"  {task:<22} !! FAILED: {e}")
         all_ok = False

   print()
   return all_ok


# ---------------------------------------------------------------------------
# test 4: goal encoder + bicameral network
# ---------------------------------------------------------------------------

def test_goal_encoder() -> bool:
   print("=== test 4: goal encoder and bicameral network shapes ===")
   try:
      import torch
      from bc_train import GoalEncoder, BicameralNetwork
      from llm_goal_parser import get_embedding

      encoder = GoalEncoder()
      network = BicameralNetwork()
      prompt  = "sort shapes from smallest to largest left to right"
      raw_emb = get_embedding(prompt)

      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = encoder(emb_t).squeeze(0).numpy()

         # forward pass through bicameral network
         fake_obs = torch.zeros(1, get_obs_size())
         action   = network(fake_obs)

      enc_ok    = encoding.shape == (GOAL_ENCODING_DIM,)
      action_ok = action.shape == (1, 3)

      print(f"  raw embedding  : {raw_emb.shape}  (expected (384,))")
      print(f"  goal encoding  : {encoding.shape}  "
            f"(expected ({GOAL_ENCODING_DIM},))  "
            f"{'ok' if enc_ok else '!! FAIL'}")
      print(f"  bicameral fwd  : {tuple(action.shape)}  (expected (1, 3))  "
            f"{'ok' if action_ok else '!! FAIL'}")

      # verify integration with ShapeEnv
      goal = _default_goal("arrange_in_sequence")
      env  = ShapeEnv(n_shapes=2, goal=goal)
      env.set_goal_encoding(encoding)
      obs, _ = env.reset()
      obs_ok = obs.shape[0] == get_obs_size()
      print(f"  obs after set_goal_encoding: {obs.shape}  "
            f"(expected ({get_obs_size()},))  "
            f"{'ok' if obs_ok else '!! FAIL'}")
      env.close()

      ok = enc_ok and action_ok and obs_ok
      print(f"  result: {'ok' if ok else 'FAIL'}")

   except Exception as e:
      print(f"  !! FAILED: {e}")
      ok = False

   print()
   return ok


# ---------------------------------------------------------------------------
# test 5: trained model vs random
# ---------------------------------------------------------------------------

def test_trained_model(model_path: str, n_shapes: int = 2):
   print(f"=== test 5: trained model vs random (n_shapes={n_shapes}) ===")
   try:
      from stable_baselines3 import PPO
      from bc_train import BicameralPolicy
   except ImportError:
      print("  stable-baselines3 not installed, skipping")
      return

   try:
      model = PPO.load(model_path, custom_objects={
         "policy_class": BicameralPolicy})
      print(f"  loaded model from {model_path}")
   except Exception as e:
      print(f"  could not load model: {e}")
      return

   goal = _default_goal("arrange_in_sequence")
   env  = ShapeEnv(n_shapes=n_shapes, goal=goal)

   obs, _ = env.reset()
   trained_rewards = []
   for _ in range(50):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, _ = env.step(action)
      trained_rewards.append(reward)
      if terminated or truncated:
         break

   obs, _ = env.reset()
   random_rewards = []
   for _ in range(len(trained_rewards)):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, _ = env.step(action)
      random_rewards.append(reward)
      if terminated or truncated:
         break

   trained_mean = float(np.mean(trained_rewards))
   random_mean  = float(np.mean(random_rewards))
   print(f"  trained mean reward : {trained_mean:.4f}")
   print(f"  random  mean reward : {random_mean:.4f}")
   if trained_mean > random_mean:
      print("  trained beats random — learning is working")
   else:
      print("  !! trained not beating random — may need more training")
   print()
   env.close()


# ---------------------------------------------------------------------------
# test 6: pygame render with cursor
# ---------------------------------------------------------------------------

def test_render():
   print("=== test 6: pygame render with cursor ===")
   try:
      import pygame
      goal   = _default_goal("arrange_in_region")
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
            print(f"  frame {i:3d}: cx={env.cx:.0f} cy={env.cy:.0f}  "
                  f"holding={env.holding}  grabbed={env.grabbed_idx}  "
                  f"reward={reward:.4f}")

         if terminated or truncated:
            obs, _ = env.reset()

      env.close()
      print("  render test passed")
   except Exception as e:
      print(f"  !! render error: {e}")
   print()


# ---------------------------------------------------------------------------
# test 7: oracle per-task solve rate
# ---------------------------------------------------------------------------

def test_oracle_per_task(n_episodes_per_task: int = 20) -> bool:
   print(f"=== test 7: oracle per-task solve rate "
         f"({n_episodes_per_task} episodes each) ===")

   from oracle import OraclePolicy
   import torch
   from bc_train import GoalEncoder
   from llm_goal_parser import get_embedding

   goal_encoder = GoalEncoder()
   goal_encoder.eval()

   task_prompts = {
      "reach":               "move the cursor to the shape",
      "touch":               "click on the shape",
      "drag":                "drag the shape to the left side",
      "arrange_in_sequence": "sort shapes from smallest to largest left to right",
      "arrange_in_line":     "arrange shapes in a horizontal line evenly spaced",
      "arrange_in_region":   "move all shapes to the left side",
      "arrange_in_groups":   "group shapes by color",
   }
   # n_shapes per task — starter tasks use 1
   task_n_shapes = {
      "reach": 1, "touch": 1, "drag": 1,
      "arrange_in_sequence": 3, "arrange_in_line": 3,
      "arrange_in_region": 3, "arrange_in_groups": 3,
   }

   all_ok = True
   rng    = np.random.default_rng(42)

   for task, prompt in task_prompts.items():
      goal    = _default_goal(task)
      raw_emb = get_embedding(prompt)
      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = goal_encoder(emb_t).squeeze(0).numpy()

      n_shp    = task_n_shapes[task]
      n_solved = 0
      mean_r   = 0.0

      for _ in range(n_episodes_per_task):
         env = ShapeEnv(n_shapes=n_shp, goal=goal)
         env.set_goal_encoding(encoding)
         oracle = OraclePolicy(env, noise_std=0.0)
         obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
         oracle.reset()
         done = False
         ep_r = 0.0

         while not done:
            action = oracle.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_r += reward
            done  = terminated or truncated

         if terminated:
            n_solved += 1
         mean_r += ep_r
         env.close()

      solve_rate = n_solved / n_episodes_per_task
      mean_r     = mean_r / n_episodes_per_task
      # starter tasks expect high solve rates; wave 3 expect 50%+
      threshold  = 0.70 if task in ("reach", "touch", "drag") else 0.50
      ok         = solve_rate >= threshold

      print(f"  {task:<22} solve={solve_rate:.0%}  "
            f"mean_reward={mean_r:7.2f}  "
            f"{'OK' if ok else '!! LOW — check oracle / score function'}")

      if not ok:
         all_ok = False

   print()
   return all_ok


# ---------------------------------------------------------------------------
# test 8: BC loss curve
# ---------------------------------------------------------------------------

def test_bc_loss(n_episodes: int = 80, epochs: int = 10) -> bool:
   print(f"=== test 8: BC loss curve ({n_episodes} episodes, {epochs} epochs) ===")
   try:
      import torch, tempfile
      from bc_train import train_bc
      from oracle import collect_demonstrations

      print("  collecting oracle demos...")
      dataset = collect_demonstrations(
         n_episodes=n_episodes,
         noise_std=0.06,
         verbose=False,
         force=True,   # always fresh for the diagnostic
      )
      n_trans = len(dataset["observations"])
      print(f"  collected {n_trans:,} transitions")
      if n_trans == 0:
         print("  !! no transitions collected — oracle solve rate may be 0%")
         return False

      with tempfile.TemporaryDirectory() as tmp:
         device = "cuda" if torch.cuda.is_available() else "cpu"
         train_bc(
            dataset=dataset,
            save_path=tmp,
            epochs=epochs,
            batch_size=256,
            device=device,
         )

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

def test_oracle_render():
   print("=== test 9: oracle visual check ===")
   try:
      import pygame, torch
      from oracle import OraclePolicy
      from bc_train import GoalEncoder
      from llm_goal_parser import get_embedding

      task_prompts = {
         "reach":               "move the cursor to the shape",
         "touch":               "click on the shape",
         "drag":                "drag the shape to the left side",
         "arrange_in_sequence": "sort shapes from smallest to largest left to right",
         "arrange_in_region":   "move all shapes to the left side",
         "arrange_in_groups":   "group shapes by color",
      }
      task_n_shapes = {
         "reach": 1, "touch": 1, "drag": 1,
         "arrange_in_sequence": 3,
         "arrange_in_region": 3,
         "arrange_in_groups": 3,
      }

      goal_encoder = GoalEncoder()
      goal_encoder.eval()

      pygame.init()
      window = pygame.display.set_mode((800, 600))
      clock  = pygame.time.Clock()
      font   = pygame.font.SysFont("monospace", 12)

      for task_name, prompt in task_prompts.items():
         print(f"  watching oracle: {task_name}  (Q or close window to advance)")
         goal    = _default_goal(task_name)
         n_shp   = task_n_shapes[task_name]
         raw_emb = get_embedding(prompt)
         with torch.no_grad():
            emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
            encoding = goal_encoder(emb_t).squeeze(0).numpy()

         env = ShapeEnv(n_shapes=n_shp, goal=goal)
         env.set_goal_encoding(encoding)
         oracle = OraclePolicy(env, noise_std=0.0)
         obs, _ = env.reset()
         oracle.reset()
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
            obs, _, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            steps += 1

            # draw env directly (render_mode is None — we own the window)
            from shape_env import BG_COLOR
            window.fill(BG_COLOR)
            for shape in env.shapes:
               shape.draw(window, font)
            env.window = window
            env._draw_cursor()
            env.window = None

            score = env._compute_score()
            hud   = (f"oracle | task: {task_name} | "
                     f"step: {steps} | progress: {score:.2%} | "
                     f"phase: {oracle.phase}")
            window.blit(font.render(hud, True, (200, 200, 200)), (10, 10))

            if terminated:
               window.blit(
                  font.render("SOLVED!", True, (100, 220, 100)), (370, 280))

            pygame.display.flip()
            clock.tick(30)

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
   parser = argparse.ArgumentParser(
      description="diagnose the shape manipulation environment")
   parser.add_argument("--model",          type=str, default=None)
   parser.add_argument("--skip-render",    action="store_true")
   parser.add_argument("--n-shapes",       type=int, default=3)
   parser.add_argument("--oracle",         action="store_true",
                       help="run oracle diagnostics (tests 7-9)")
   parser.add_argument("--oracle-episodes",type=int, default=20)
   args = parser.parse_args()

   ok1 = test_env_steps(n_shapes=min(args.n_shapes, 2))
   ok2 = test_random_rewards(n_shapes=args.n_shapes)
   ok3 = test_all_tasks()
   ok4 = test_goal_encoder()

   if args.model:
      test_trained_model(args.model, n_shapes=args.n_shapes)

   if not args.skip_render:
      test_render()
   else:
      print("=== test 6: skipped (--skip-render) ===\n")

   ok7 = ok8 = None
   if args.oracle:
      ok7 = test_oracle_per_task(n_episodes_per_task=args.oracle_episodes)
      ok8 = test_bc_loss()
      if not args.skip_render:
         test_oracle_render()
      else:
         print("=== test 9: skipped (--skip-render) ===\n")

   print("=== summary ===")
   for label, result in [
      ("env cursor mechanics", ok1),
      ("rewards vary",         ok2),
      ("all tasks work",       ok3),
      ("goal encoder ok",      ok4),
      ("oracle solve rates",   ok7),
      ("BC loss healthy",      ok8),
   ]:
      if result is not None:
         print(f"  {label:<22} : {'ok' if result else 'FAIL'}")

   print()
   core = [ok1, ok2, ok3, ok4]
   if all(core):
      print("all core tests passed — environment is healthy.")
   else:
      print("some tests failed — check warnings above.")
