"""
debug.py

diagnostic script. run this before training or when something breaks.

tests:
   1. env steps move cursor and shapes respond to grip
   2. rewards vary across episodes; score functions respond to shape movement
   3. all tasks initialise and step without error (flat rewards expected)
   4. env loads bicameral network and obs vector correctly
   5. trained model outperforms random (if --model provided)
   6. oracle visual check; watch oracle solve each task (--oracle --render)
   7. oracle per-task solve rate (--oracle flag)
   8. BC loss curve (--oracle flag)
   9. per-task BC loss breakdown (--oracle flag)

usage:
   python debug.py                         # core tests only, no render
   python debug.py --oracle                # + oracle solve rates + BC loss
   python debug.py --oracle --render       # + oracle visual check
   python debug.py --model models/shape_agent/best_model
"""

import argparse
import numpy as np
from shape_env import ShapeEnv
from config import EMBEDDING_DIM, get_obs_size, get_action_size


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

def print_obs_vector(obs: np.ndarray) -> None:
   """print a human-readable breakdown of the obs vector by region."""
   from config import OBS_REGIONS
   # show up to 18 values, truncate the rest (goal embedding is huge)
   limit = 18
   print("(shape features: [x_norm, y_norm, size_norm, color_norm, shape_type_norm])")
   print(f"  {'region':<16}  {'size':>5}  values")
   print(f"  {'-'*16}  {'-'*5}  ------")
   for name, slc, desc in OBS_REGIONS:
      values = obs[slc]
      preview = np.array2string(
         values[:limit], precision=3, suppress_small=True, separator=", "
      )
      if len(values) > limit:
         preview += f" ... ({len(values)} total)"
      print(f"  {name:<16} {len(values):>5}  {desc}")
      print(f"  {preview}")
      print()

# ---------------------------------------------------------------------------
# test 1: basic step mechanics
# ---------------------------------------------------------------------------

def test_env_steps(n_shapes: int = 2) -> bool:
   """
   check that:
   - obs shape is 428
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

   reward ranges will differ across tasks — that's expected. what matters
   is that each range is clearly above zero, confirming the score function
   responds to shape movement. a flat range (~0.0) means the score function
   for that task is broken or unresponsive.
   """
   print(f"=== test 2: reward variance (grip-and-drag actions) ===")
   all_ok = True

   test_cases = [
      ("reach",               1),
      ("drag",                1),
      ("arrange_in_region",   n_shapes),
      ("arrange_in_groups",   n_shapes),
   ]
   for task, n_shp in test_cases:
      goal    = _default_goal(task)
      env     = ShapeEnv(n_shapes=n_shp, goal=goal)
      rewards = []
      # use a fresh unseeded rng per task — fixed seed 0 can produce unlucky
      # spawns where all shapes start inside the target region, giving flat
      # reward regardless of dragging. 10 episodes averages this out.
      rng = np.random.default_rng()

      for _ in range(10):
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
   """
   check that all tasks initialise and step without throwing an error.
   uses random actions — flat rewards of -0.020 (just the step penalty)
   are expected and correct for most tasks, since random actions rarely
   grip and drag shapes. this test is only checking for crashes, not
   that rewards are meaningful. "none" task excluded.
   """
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
            obs, reward, _, _, info = env.step(action)
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
# test 4: bicameral network and obs shapes
# ---------------------------------------------------------------------------

def test_network_shapes() -> bool:
   print("=== test 4: bicameral network and obs shapes ===")
   try:
      import torch
      from config import OBS_REGIONS
      from bc_train import BicameralNetwork
      from llm_goal_parser import get_embedding, parse_goal

      network = BicameralNetwork()

      # forward pass through bicameral network
      fake_obs = torch.zeros(1, get_obs_size())
      action   = network(fake_obs)

      prompt = "sort shapes from smallest to largest left to right"
      goal   = parse_goal(prompt)
      emb    = get_embedding(prompt)

      env      = ShapeEnv(n_shapes=3, goal=goal, goal_embedding=emb)
      obs, _   = env.reset()

      print(f"\n  --- obs regions ---")
      print(f"  task={goal.get('task', 'none')}; n_shapes=3")
      print()
      print_obs_vector(obs)
      print()

      # verify each region slice size matches config
      obs_ok = True
      for name, slc, _ in OBS_REGIONS:
         expected = slc.stop - slc.start
         actual   = obs[slc].size
         if actual != expected:
            print(f"  !! region '{name}' size mismatch: "
                  f"got {actual}, expected {expected}")
            obs_ok = False

      action_ok = action.shape == (1, get_action_size())
      emb_ok    = emb.shape    == (EMBEDDING_DIM,)

      print(f"  obs regions    : {'ok' if obs_ok else '!! FAIL'}")
      print(f"  bicameral fwd  : {tuple(action.shape)}  "
            f"(expected (1, {get_action_size()}))  "
            f"{'ok' if action_ok else '!! FAIL'}")
      print(f"  raw embedding  : {emb.shape}  "
            f"(expected ({EMBEDDING_DIM},))  "
            f"{'ok' if emb_ok else '!! FAIL'}")

      ok = obs_ok and action_ok and emb_ok
      print(f"  result: {'ok' if ok else '!! FAIL'}")

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
# test 7: oracle solve rates
# ---------------------------------------------------------------------------

def test_oracle_per_task(n_episodes_per_task: int = 20) -> bool:
   print(f"=== test 7: oracle per-task solve rate "
         f"({n_episodes_per_task} episodes each) ===")

   from oracle import OraclePolicy
   from llm_goal_parser import get_embedding

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

      n_shp    = task_n_shapes[task]
      n_solved = 0
      mean_r   = 0.0

      for _ in range(n_episodes_per_task):
         env = ShapeEnv(n_shapes=n_shp, goal=goal, goal_embedding=raw_emb)
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

      print("  note: this is a quick diagnostic run (80 episodes, 10 epochs).")
      print("  loss will be higher than a full training run — that's expected.")
      print("  look for: loss decreasing steadily across epochs (not plateauing")
      print("  after epoch 2), grip loss dropping below ~0.25, dxy below ~0.30.")
      print("  a plateau at high loss usually means too little data or lr too high.")
      ok = True

   except Exception as e:
      print(f"  !! FAILED: {e}")
      ok = False

   print()
   return ok



# ---------------------------------------------------------------------------
# test 9: per-task BC loss breakdown
# ---------------------------------------------------------------------------

def test_bc_loss_per_task(n_episodes: int = 80) -> bool:
   """
   collect oracle demos, train a quick BC pass, then evaluate MSE loss
   broken down by task. helps identify whether BC failure is uniform or
   concentrated on specific tasks (usually a data starvation issue).

   a healthy per-task loss should be below ~0.05. anything above 0.15
   on a specific task means the network hasn't learned that task at all.
   """
   print(f"=== test 9: per-task BC loss breakdown ({n_episodes} episodes) ===")
   try:
      import torch
      import torch.nn.functional as F
      import tempfile
      from bc_train import train_bc
      from oracle import collect_demonstrations
      from config import SUPPORTED_TASKS

      print("  collecting oracle demos...")
      dataset = collect_demonstrations(
         n_episodes=n_episodes,
         noise_std=0.06,
         verbose=False,
      )

      n_trans = len(dataset["observations"])
      print(f"  collected {n_trans:,} transitions")
      if n_trans == 0:
         print("  !! no transitions — oracle may be broken")
         return False

      # check that the dataset has a 'tasks' field (one entry per transition)
      if "tasks" not in dataset:
         print("  !! dataset has no 'tasks' key — cannot break down by task.")
         print("     add task labels to collect_demonstrations() in oracle.py")
         print("     to enable this test. skipping.")
         return False

      device = "cuda" if torch.cuda.is_available() else "cpu"

      with tempfile.TemporaryDirectory() as tmp:
         network = train_bc(
            dataset=dataset,
            save_path=tmp,
            epochs=10,
            batch_size=256,
            device=device,
         )

      network = network.to(device)
      network.eval()

      obs_t  = torch.tensor(dataset["observations"], dtype=torch.float32).to(device)
      act_t  = torch.tensor(dataset["actions"],      dtype=torch.float32).to(device)
      tasks  = dataset["tasks"]   # list of task name strings, one per transition

      print()
      print(f"  {'task':<25}  {'n':>6}  {'loss':>8}  status")
      print(f"  {'-'*25}  {'-'*6}  {'-'*8}  ------")

      all_ok = True

      with torch.no_grad():
         for task in SUPPORTED_TASKS:
            # mask for this task's transitions
            mask = [i for i, t in enumerate(tasks) if t == task]
            if not mask:
               print(f"  {task:<25}  {'0':>6}  {'n/a':>8}  !! no samples")
               all_ok = False
               continue

            idx       = torch.tensor(mask, dtype=torch.long)
            task_obs  = obs_t[idx]
            task_act  = act_t[idx]
            pred      = network(task_obs)

            # match train_bc loss: MSE for dx/dy, BCE for grip.
            # grip labels are ±1.0 from oracle; convert to 0/1 for BCE.
            # using MSE on raw logits vs ±1.0 produces nonsense values here.
            loss_dxy  = F.mse_loss(pred[:, 0:2], task_act[:, 0:2]).item()
            grip_tgt  = (task_act[:, 2] > 0.0).float()
            loss_grip = F.binary_cross_entropy_with_logits(
               pred[:, 2], grip_tgt).item()
            loss      = loss_dxy + 0.5 * loss_grip

            # flag tasks with suspiciously high loss.
            # these are in-sample losses after 10 diagnostic epochs, so they
            # will naturally be higher than a full training run. thresholds:
            #   dxy > 0.50 — network hasn't learned movement for this task
            #   grip > 0.50 — network hasn't learned grip for this task
            #   combined > 0.70 — overall learning has stalled on this task
            # grip BCE near ln(2)≈0.693 means predicting ~0.5 for all steps.
            # dxy MSE near 0.5 means similar (no learning).
            warn_dxy  = loss_dxy  > 0.50
            warn_grip = loss_grip > 0.50
            warn_all  = loss      > 0.70
            ok        = not (warn_dxy or warn_grip or warn_all)
            status    = "ok" if ok else "!! HIGH"
            if not ok:
               all_ok = False

            print(f"  {task:<25}  {len(mask):>6}  {loss:>8.4f}  {status}"
                  f"  (dxy={loss_dxy:.4f}  grip={loss_grip:.4f})")

      print()
      if not all_ok:
         print("  tasks marked HIGH have dxy>0.50, grip>0.50, or combined>0.70.")
         print("  these are in-sample diagnostic losses — a full training run")
         print("  (more episodes, more epochs) should bring them down further.")
         print("  if a task is HIGH after a full run, check: data starvation")
         print("  (too few samples), oracle solve rate (test 7), or whether the")
         print("  task requires longer action sequences than others.")
      else:
         print("  all per-task losses look healthy.")

   except Exception as e:
      print(f"  !! FAILED: {e}")
      all_ok = False

   print()
   return all_ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description="diagnose the shape manipulation environment")
   parser.add_argument("--model",           type=str, default=None)
   parser.add_argument("--render",          action="store_true",
                       help="enable pygame render tests (tests 6, and 6 in --oracle mode)")
   parser.add_argument("--n-shapes",        type=int, default=3)
   parser.add_argument("--oracle",          action="store_true",
                       help="run oracle diagnostics (tests 7-9)")
   parser.add_argument("--oracle-episodes", type=int, default=40)
   args = parser.parse_args()

   ok1 = test_env_steps(n_shapes=min(args.n_shapes, 2))
   ok2 = test_random_rewards(n_shapes=args.n_shapes)
   ok3 = test_all_tasks()
   ok4 = test_network_shapes()

   ok5 = None
   if args.model:
      ok5 = True
      test_trained_model(args.model, n_shapes=args.n_shapes)
   else:
      print("=== test 5: skipped (no --model provided) ===\n")

   ok7 = ok8 = ok6 = ok9 = None
   if args.oracle:
      if args.render:
         print("=== test 6: skipped (function removed lol) ===\n")
         ok6 = True
      else:
         print("=== test 6: skipped (no --render flag) ===\n")
      ok7 = test_oracle_per_task(n_episodes_per_task=args.oracle_episodes)
      ok8 = test_bc_loss()
      ok9 = test_bc_loss_per_task(n_episodes=args.oracle_episodes)
   else:
      print("=== tests 6-9: skipped (no --oracle flag) ===\n")

   print("=== summary ===")
   for label, result in [
      ("test 1  env mechanics  ", ok1),
      ("test 2  reward variance", ok2),
      ("test 3  all tasks ok   ", ok3),
      ("test 4  network shapes ", ok4),
      ("test 5  trained model  ", ok5),
      ("test 6  oracle render  ", ok6),
      ("test 7  oracle solve   ", ok7),
      ("test 8  BC loss curve  ", ok8),
      ("test 9  BC per-task    ", ok9),
   ]:
      if result is True:
         status = "ok"
      elif result is False:
         status = "FAIL"
      else:
         status = "skipped"
      print(f"  {label} : {status}")

   print()
   core = [ok1, ok2, ok3, ok4]
   if all(core):
      print("all core tests passed — environment is healthy.")
   else:
      print("some tests failed — check warnings above.")
