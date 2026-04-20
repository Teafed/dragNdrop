"""
train.py

entry point for goal-conditioned training — single-shape, 12-stage curriculum.

python train.py
python train.py --timesteps 1000000 --bc-episodes 600 --bc-epochs 30
python train.py --no-oracle
python train.py --start-stage 5   # skip straight to reach
python train.py --resume ./models/shape_agent/stage_04_checkpoint --start-stage 5 --no-oracle
"""

import argparse
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from shape_env import ShapeEnv
from llm_goal_parser import parse_goal, get_embedding
from callbacks import ShapeTaskCallback, CurriculumCallback, TrainingSummaryCallback
from config import N_ENVS, SUPPORTED_TASKS
from prompt_gen import PromptGenerator


# ---------------------------------------------------------------------------
# env config
# ---------------------------------------------------------------------------

def _save_env_config(save_path: str, curriculum):
   import json
   os.makedirs(save_path, exist_ok=True)
   tasks  = curriculum.active_tasks if curriculum is not None else SUPPORTED_TASKS
   config = {"n_shapes": 1, "tasks": tasks}
   with open(os.path.join(save_path, "env_config.json"), "w") as f:
      json.dump(config, f, indent=3)
   print(f"[train] env config saved to {save_path}/env_config.json")


# ---------------------------------------------------------------------------
# env factory
# ---------------------------------------------------------------------------

def make_goal_conditioned_env(curriculum=None):
   _gen = PromptGenerator()

   def _init():
      if curriculum is not None:
         prompt = curriculum.sample_prompt()
      else:
         prompt = _gen.sample()
      goal    = parse_goal(prompt)
      raw_emb = get_embedding(prompt)
      env     = ShapeEnv(n_shapes=1, goal=goal, goal_embedding=raw_emb)
      return Monitor(env)

   return _init


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def build_callbacks(save_path: str, n_envs: int,
                    curriculum=None) -> CallbackList:
   _gen = PromptGenerator()

   def _make_static_eval_env():
      prompt  = curriculum.sample_prompt() if curriculum else _gen.sample()
      goal    = parse_goal(prompt)
      raw_emb = get_embedding(prompt)
      return Monitor(ShapeEnv(n_shapes=1, goal=goal, goal_embedding=raw_emb))

   eval_callback = EvalCallback(
      _make_static_eval_env(),
      best_model_save_path=save_path,
      log_path="./logs/",
      eval_freq=max(5000 // n_envs, 1),
      n_eval_episodes=50,
      verbose=1,
   )
   task_callback = ShapeTaskCallback(
      curriculum=curriculum, eval_freq=5000, n_eval_episodes=10, verbose=1)

   callback_list = [eval_callback, task_callback]

   if curriculum is not None:
      curriculum_callback = CurriculumCallback(
         curriculum=curriculum,
         eval_freq=5_000,
         n_eval_episodes=50,
         verbose=1,
         save_path=save_path,
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
   timesteps:      int  = 1_000_000,
   save_path:      str  = "./models/shape_agent",
   bc_episodes:    int  = 600,
   bc_epochs:      int  = 30,
   use_oracle:     bool = True,
   use_curriculum: bool = True,
   start_stage:    int  = 0,
   resume_model:   str  = None,
):
   from bc_train import BicameralPolicy, train_bc, build_ppo_from_bc
   from oracle import collect_demonstrations
   from prompt_train import train_prompt

   if use_curriculum:
      from curriculum import CurriculumManager
      curriculum = CurriculumManager(verbose=True, start_stage=start_stage)
      print(f"\n[curriculum] initial stage: {curriculum.status()}")
   else:
      curriculum = None
      print("\n[curriculum] disabled")

   _save_env_config(save_path, curriculum)

   # --- phase 0: prompt pretraining ---
   # auto-retrains if N_TASKS changed (handled inside train_prompt)
   # prompt_trained = train_prompt(save_path=os.path.join(save_path, "phase0"))

   # --- oracle BC warm-start ---
   bc_network = None
   if use_oracle:
      # weight touch and hold_at higher — grip timing is hardest to learn
      task_weights = {
         "move_cardinal": 1.0,
         "move_diagonal": 1.0,
         "click_at":      1.5,
         "hold_at":       2.0,
         "approach":      1.0,
         "reach":         1.0,
         "touch":         2.0,
         "drag":          1.5,
      }

      dataset = collect_demonstrations(
         n_episodes=bc_episodes,
         verbose=True,
         task_weights=task_weights,
      )

      device     = "cuda" if torch.cuda.is_available() else "cpu"
      bc_network = train_bc(
         dataset=dataset,
         save_path=save_path,
         epochs=bc_epochs,
         device=device,
         pretrained_network=None,
      )
   else:
      device = "cuda" if torch.cuda.is_available() else "cpu"

   use_gpu    = (device == "cuda")
   n_envs     = N_ENVS * 4 if use_gpu else N_ENVS
   batch_size = 512         if use_gpu else 128
   vec_cls    = DummyVecEnv

   print(f"\n[train] device={device}  n_envs={n_envs}  batch_size={batch_size}")

   vec_env = make_vec_env(
      make_goal_conditioned_env(curriculum),
      n_envs=n_envs,
      vec_env_cls=vec_cls,
   )

   # --- build PPO ---
   if resume_model is not None:
      print(f"\n--- resuming from checkpoint: {resume_model} ---")
      model = PPO.load(resume_model, env=vec_env)
   elif bc_network is not None:
      print("\n--- initialising PPO from BC weights ---")
      model = build_ppo_from_bc(
         bc_network, n_shapes=1, vec_env=vec_env, batch_size=batch_size,
         ent_coef=0.10,    # high entropy to survive task transitions
         clip_range=0.30,  # looser clip for larger corrections post-collapse
      )
   else:
      model = PPO(
         BicameralPolicy, vec_env,
         learning_rate=3e-4, n_steps=2048, batch_size=batch_size,
         n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.30,
         ent_coef=0.10, vf_coef=0.5, max_grad_norm=0.5,
         verbose=1, tensorboard_log="./logs/tensorboard/",
      )

   callbacks = build_callbacks(save_path, n_envs, curriculum)

   print(f"\n--- training PPO for {timesteps:,} timesteps ---\n")
   model.learn(total_timesteps=timesteps, callback=callbacks)

   final_path = os.path.join(save_path, "final_model")
   model.save(final_path)
   _save_env_config(save_path, curriculum)
   print(f"\n--- done. model saved to {final_path} ---")
   return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--timesteps",    type=int,  default=1_000_000)
   parser.add_argument("--save",         type=str,  default="./models/shape_agent")
   parser.add_argument("--no-oracle",    action="store_true")
   parser.add_argument("--no-curriculum",action="store_true")
   parser.add_argument("--start-stage",  type=int,  default=0)
   parser.add_argument("--bc-episodes",  type=int,  default=600)
   parser.add_argument("--bc-epochs",    type=int,  default=30)
   parser.add_argument("--resume",       type=str,  default=None)
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