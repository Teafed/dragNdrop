"""
train.py

entry point for goal-conditioned training of the shape manipulation agent.

training uses a curriculum that builds from simple cursor skills up to
the full multi-shape arrangement tasks. see curriculum.py for stage
definitions and advancement logic.

training modes:

   full pipeline (recommended):
      python train.py
      python train.py --timesteps 800000 --bc-episodes 500 --bc-epochs 30

   skip oracle warm-start:
      python train.py --no-oracle

   skip curriculum (all tasks from step 0):
      python train.py --no-curriculum

   start from a specific curriculum stage:
      python train.py --start-stage 3
"""

import argparse
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal, get_embedding
from callbacks import ShapeTaskCallback, CurriculumCallback, TrainingSummaryCallback
from config import MAX_SHAPES, N_ENVS, SUPPORTED_TASKS
from prompt_gen import PromptGenerator

# ---------------------------------------------------------------------------
# training config
# ---------------------------------------------------------------------------

def _save_training_config(save_path: str, curriculum, timesteps: int):
   """write training_config.json alongside the model files."""
   import json
   os.makedirs(save_path, exist_ok=True)
   n_shapes = curriculum.n_shapes_range[1] if curriculum is not None else MAX_SHAPES
   tasks    = curriculum.active_tasks      if curriculum is not None else SUPPORTED_TASKS
   config   = {
      "n_shapes": n_shapes,
      "tasks":    tasks,
   }
   path = os.path.join(save_path, "training_config.json")
   with open(path, "w") as f:
      json.dump(config, f, indent=3)
   print(f"[train] training config saved to {path}")

# ---------------------------------------------------------------------------
# goal-conditioned env factory
# ---------------------------------------------------------------------------

def make_goal_conditioned_env(goal_encoder, curriculum=None, render_mode=None):
   """
   returns a factory function for Monitor-wrapped ShapeEnvs.
   if curriculum is provided, samples task and n_shapes from the current
   stage. otherwise samples uniformly from TASK_POOL with random n_shapes.
   """
   _gen = PromptGenerator()
   def _init():
      if curriculum is not None:
         prompt = curriculum.sample_prompt()
         n_shp  = curriculum.sample_n_shapes()
      else:
         prompt = _gen.sample()
         n_shp  = None   # ShapeEnv samples randomly up to MAX_SHAPES

      goal    = parse_goal(prompt)
      raw_emb = get_embedding(prompt)

      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = goal_encoder(emb_t).squeeze(0).numpy()

      env = ShapeEnv(n_shapes=n_shp, goal=goal)
      env.set_goal_encoding(encoding)
      return Monitor(env)

   return _init


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def build_callbacks(goal_encoder, save_path: str, n_envs: int,
                    curriculum=None) -> CallbackList:
   """
   build the callback stack.

   EvalCallback       — SB3's built-in best-model saver (uses a static env,
                        fine because it only saves checkpoints, doesn't gate
                        curriculum advancement).
   ShapeTaskCallback  — curriculum-aware eval; re-samples a fresh env from
                        the current stage at every evaluation so it always
                        reports metrics for the task being trained now.
   CurriculumCallback — per-task solve rates that gate curriculum advancement.
                        only added when curriculum is active.
   """
   # EvalCallback still needs one env up front — it's only used for checkpoint
   # saving so it's OK that it stays on stage-0 task.
   def _make_static_eval_env():
      _gen = PromptGenerator()
      if curriculum is not None:
         prompt = curriculum.sample_prompt()
         n_shp  = curriculum.sample_n_shapes()
      else:
         prompt =_gen.sample()
         n_shp  = None
      goal    = parse_goal(prompt)
      raw_emb = get_embedding(prompt)
      with torch.no_grad():
         emb_t    = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0)
         encoding = goal_encoder(emb_t).squeeze(0).numpy()
      env = ShapeEnv(n_shapes=n_shp, goal=goal)
      env.set_goal_encoding(encoding)
      return Monitor(env)

   eval_callback = EvalCallback(
      _make_static_eval_env(),
      best_model_save_path=save_path,
      log_path="./logs/",
      eval_freq=max(5000 // n_envs, 1),
      n_eval_episodes=10,
      verbose=1,
   )

   # FIX: pass curriculum + goal_encoder so ShapeTaskCallback can re-sample
   # a fresh env from the current stage at every eval, rather than being
   # frozen on the stage-0 env that was constructed at callback build time.
   task_callback = ShapeTaskCallback(
      curriculum=curriculum,
      goal_encoder=goal_encoder,
      eval_freq=5000,
      n_eval_episodes=10,
      verbose=1,
   )

   callback_list        = [eval_callback, task_callback]
   curriculum_callback  = None

   if curriculum is not None:
      curriculum_callback = CurriculumCallback(
         curriculum=curriculum,
         goal_encoder=goal_encoder,
         eval_freq=5_000,    # was 10k — check more often so gate fires promptly
         n_eval_episodes=30, # was 20 — more episodes = less noisy gate measurement
         verbose=1,
         save_path=save_path,  # saves stageN_checkpoint.zip on each advance
      )
      summary_callback = TrainingSummaryCallback(
         curriculum_cb=curriculum_callback,
         task_cb=task_callback,
         summary_freq=50_000,
      )
      callback_list.append(curriculum_callback)
      callback_list.append(summary_callback)

   return CallbackList(callback_list)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(
   timesteps:      int  = 800_000,
   save_path:      str  = "./models/shape_agent",
   bc_episodes:    int  = 500,
   bc_epochs:      int  = 30,
   use_oracle:     bool = True,
   use_curriculum: bool = True,
   start_stage:    int  = 0,
   resume_model:   str  = None,
):
   """
   train the goal-conditioned agent.

   if use_oracle=True (recommended):
      1. collect oracle demos (or load from disk if available)
      2. train BicameralNetwork via BC
      3. transplant BC weights into PPO with BicameralPolicy
      4. PPO fine-tune with curriculum

   if use_oracle=False:
      skip steps 1-3 and train PPO from random init.

   start_stage: skip directly to this curriculum stage (useful for
   resuming or ablating individual stages).
   """
   from bc_train import (
      GoalEncoder, BicameralPolicy,
      train_bc, build_ppo_from_bc,
   )
   from oracle import collect_demonstrations

   goal_encoder = GoalEncoder()

   if use_curriculum:
      from curriculum import CurriculumManager
      curriculum = CurriculumManager(verbose=True, start_stage=start_stage)
      print(f"\n[curriculum] initial stage: {curriculum.status()}")
   else:
      curriculum = None
      print("\n[curriculum] disabled — training on all tasks from step 0")

   # write config immediately so it exists even if training crashes later
   _save_training_config(save_path, curriculum, timesteps)

   if use_oracle:
      # demos are collected across the full task pool regardless of curriculum
      # stage — bc warms up all tasks, then ppo fine-tunes with curriculum.
      print(f"\n--- collecting {bc_episodes} oracle demonstrations "
            f"across all task pool prompts ---")

      dataset = collect_demonstrations(
         n_episodes=bc_episodes,
         goal_encoder=goal_encoder,
         verbose=True,
      )

      device = "cuda" if torch.cuda.is_available() else "cpu"
      bc_network = train_bc(
         dataset=dataset,
         save_path=save_path,
         epochs=bc_epochs,
         device=device,
      )

      n_envs  = N_ENVS
      vec_env = make_vec_env(
         make_goal_conditioned_env(goal_encoder, curriculum), n_envs=n_envs)

      if resume_model is not None:
         print(f"\n--- resuming from checkpoint: {resume_model} ---")
         model = PPO.load(resume_model, env=vec_env)
      else:
         print("\n--- initialising PPO from BC weights (BicameralPolicy) ---")
         model = build_ppo_from_bc(bc_network, n_shapes=MAX_SHAPES, vec_env=vec_env)

   else:
      n_envs  = N_ENVS
      vec_env = make_vec_env(
         make_goal_conditioned_env(goal_encoder, curriculum), n_envs=n_envs)

      if resume_model is not None:
         print(f"\n--- resuming from checkpoint: {resume_model} ---")
         model = PPO.load(resume_model, env=vec_env)
      else:
         model = PPO(
            BicameralPolicy,
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
         )

   # goal encoder is fixed (random projection, not trained) — set eval mode
   # once here regardless of which branch was taken above.
   goal_encoder.eval()

   callbacks = build_callbacks(goal_encoder, save_path, n_envs, curriculum)

   print(f"\n--- training PPO for {timesteps:,} timesteps ---\n")
   model.learn(total_timesteps=timesteps, callback=callbacks)

   final_path = os.path.join(save_path, "final_model")
   model.save(final_path)
   _save_training_config(save_path, curriculum, timesteps)

   print(f"\n--- done. model saved to {final_path} ---")
   
   return model, goal_encoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
      description="train the goal-conditioned shape manipulation agent."
   )
   parser.add_argument(
      "--timesteps", type=int, default=800_000,
      help="total PPO training timesteps (default: 800000)",
   )
   parser.add_argument(
      "--save", type=str, default="./models/shape_agent",
      help="directory to save model checkpoints and goal encoder",
   )
   parser.add_argument(
      "--no-oracle", action="store_true",
      help="skip oracle warm-start, train PPO from random init",
   )
   parser.add_argument(
      "--no-curriculum", action="store_true",
      help="disable curriculum — train on all tasks from step 0",
   )
   parser.add_argument(
      "--start-stage", type=int, default=0,
      help="start at this curriculum stage (0-6, default: 0)",
   )
   parser.add_argument(
      "--bc-episodes", type=int, default=500,
      help="oracle demo episodes for BC warm-start (default: 500)",
   )
   parser.add_argument(
      "--bc-epochs", type=int, default=30,
      help="BC training epochs (default: 30)",
   )
   parser.add_argument(
      "--resume", type=str, default=None,
      help=(
         "path to a saved model checkpoint to resume from (no .zip suffix needed). "
         "combine with --start-stage to resume at the right curriculum stage and "
         "--no-oracle to skip the bc phase. "
         "example: python train.py --resume ./models/shape_agent/stage_02_checkpoint "
         "--start-stage 3 --no-oracle"
      ),
   )
   args = parser.parse_args()

   train(
      timesteps=args.timesteps,
      save_path=args.save,
      bc_episodes=args.bc_episodes,
      bc_epochs=args.bc_epochs,
      use_oracle=not args.no_oracle,
      use_curriculum=not args.no_curriculum,
      start_stage=args.start_stage,
      resume_model=args.resume,
   )
