"""
curriculum_tester.py

trains and runs models paired with named curriculum configs.

folder layout (under ./models/curriculums/):

   models/
   └── curriculums/
       ├── MyCurriculum/
       │   ├── curriculum.json     ← stage definitions for this curriculum
       │   ├── best_model.zip      ← saved by EvalCallback during training
       │   ├── final_model.zip     ← saved at end of training
       │   └── training_config.json
       └── AnotherCurriculum/
           ├── curriculum.json
           ├── best_model.zip
           ├── final_model.zip
           └── training_config.json

usage:

   # --- define a curriculum as a list of stage dicts and save to json ---
   CurriculumTester.save_curriculum("MyCurriculum", stages=MY_STAGES)

   # --- train ---
   tester = CurriculumTester()
   model = tester.test_curriculum("MyCurriculum", timesteps=300_000)

   # --- run / visualise ---
   tester.run_model("MyCurriculum", n_episodes=5, render=True)

   # --- list all saved curriculums ---
   CurriculumTester.list_curriculums()
"""

import json
import os

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from callbacks import CurriculumCallback, ShapeTaskCallback, TrainingSummaryCallback
from config import MAX_SHAPES, N_ENVS, SUPPORTED_TASKS
from llm_goal_parser import get_embedding, parse_goal
from prompt_gen import PromptGenerator
from shape_env import ShapeEnv

# root folder that holds all curriculum subfolders
_CURRICULUMS_ROOT = os.path.join(".", "models", "curriculums")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _curriculum_dir(name: str) -> str:
    return os.path.join(_CURRICULUMS_ROOT, name)


def _curriculum_json_path(name: str) -> str:
    return os.path.join(_curriculum_dir(name), "curriculum.json")


def _final_model_path(name: str) -> str:
    return os.path.join(_curriculum_dir(name), "final_model")          # .zip added by SB3


def _best_model_path(name: str) -> str:
    return os.path.join(_curriculum_dir(name), "best_model.zip")


def _training_config_path(name: str) -> str:
    return os.path.join(_curriculum_dir(name), "training_config.json")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# CurriculumManager shim — loads stages from a json file instead of the
# hard-coded _STAGES list so each named curriculum is self-contained.
# ---------------------------------------------------------------------------

class _JsonCurriculumManager:
    """
    drop-in replacement for CurriculumManager that reads stage definitions
    from a curriculum.json file rather than the hard-coded _STAGES list.

    the json file must be a list of stage dicts matching the same schema
    used by curriculum.py (name, tasks, n_shapes_min, n_shapes_max,
    gate_task, gate_sr, step_ceiling).  extra keys like notes and
    reward_hints are silently ignored during training but preserved in
    the file for reference.
    """

    def __init__(self, stages: list[dict], verbose: bool = False, start_stage: int = 0):
        if not stages:
            raise ValueError("stages list must not be empty")
        self._stages    = stages
        self._stage_idx = start_stage
        self._verbose   = verbose
        self._gen       = PromptGenerator()

    # ------------------------------------------------------------------
    # compatibility surface expected by the rest of train.py
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> dict:
        return self._stages[self._stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage["name"]

    @property
    def is_final_stage(self) -> bool:
        return self._stage_idx == len(self._stages) - 1

    @property
    def active_tasks(self) -> list[str]:
        return self.current_stage["tasks"]

    @property
    def n_shapes_range(self) -> tuple[int, int]:
        s = self.current_stage
        return s["n_shapes_min"], s["n_shapes_max"]

    @property
    def stage_idx(self) -> int:
        return self._stage_idx

    @property
    def stage(self) -> dict:
        return self._stages[self._stage_idx]
    
    def sample_prompt(self) -> str:
        task = self.sample_task()
        return self._gen._sample_for_task(task)

    def sample_task(self) -> str:
        import random
        return random.choice(self.current_stage["tasks"])

    def sample_n_shapes(self) -> int:
        import random
        s = self.current_stage
        return random.randint(s["n_shapes_min"], s["n_shapes_max"])

    def maybe_advance(self, per_task_solve_rates: dict, current_step: int) -> bool:
        if self.is_final_stage:
            return False
        stage        = self.current_stage
        gate_task    = stage.get("gate_task")
        gate_sr      = stage.get("gate_sr")
        step_ceiling = stage.get("step_ceiling")
        if step_ceiling is not None and current_step >= step_ceiling:
            self._advance()
            return True
        if gate_task and gate_sr is not None:
            if per_task_solve_rates.get(gate_task, 0.0) >= gate_sr:
                self._advance()
                return True
        return False

    def _advance(self):
        if not self.is_final_stage:
            self._stage_idx += 1
            if self._verbose:
                print(f"[curriculum] advanced to {self.stage_name}")

    def status(self) -> str:
        s = self.current_stage
        return (
            f"stage {self._stage_idx}/{len(self._stages)-1}: {s['name']}  |  "
            f"tasks={s['tasks']}  n_shapes={s['n_shapes_min']}-{s['n_shapes_max']}  "
            f"gate={s.get('gate_task')} >= {s.get('gate_sr')}  "
            f"ceiling={s.get('step_ceiling')}"
        )


# ---------------------------------------------------------------------------
# CurriculumTester
# ---------------------------------------------------------------------------

class CurriculumTester:
    """
    trains and runs models paired with named curriculum configs.

    each curriculum lives in  ./models/curriculums/<name>/  and contains:
        curriculum.json      — stage definitions
        final_model.zip      — weights after full training run
        best_model.zip       — best checkpoint saved by EvalCallback
        training_config.json — metadata snapshot written at train time

    public API:
        CurriculumTester.save_curriculum(name, stages)   — write a new curriculum json
        CurriculumTester.list_curriculums()               — print all saved curriculums
        tester.test_curriculum(name, ...)                 — train a model on a curriculum
        tester.run_model(name, ...)                       — load weights and run episodes
    """

    # ------------------------------------------------------------------
    # static helpers — curriculum file management
    # ------------------------------------------------------------------

    @staticmethod
    def save_curriculum(name: str, stages: list[dict]):
        """
        write a list of stage dicts to  ./models/curriculums/<name>/curriculum.json.

        creates the folder if it doesn't exist. safe to call before training.
        stages can include any extra keys (notes, reward_hints, etc.) — they
        are stored in the json and ignored during training.

        example:
            from curriculum import _STAGES
            CurriculumTester.save_curriculum("Baseline", _STAGES)
        """
        path = _curriculum_json_path(name)
        _ensure_dir(_curriculum_dir(name))
        with open(path, "w") as f:
            json.dump(stages, f, indent=3)
        print(f"[CurriculumTester] saved curriculum '{name}' → {path}  ({len(stages)} stages)")

    @staticmethod
    def load_curriculum_stages(name: str) -> list[dict]:
        """load and return the stage list for a named curriculum."""
        path = _curriculum_json_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no curriculum.json found for '{name}' — expected at {path}\n"
                f"create one with CurriculumTester.save_curriculum('{name}', stages)"
            )
        with open(path) as f:
            stages = json.load(f)
        print(f"[CurriculumTester] loaded curriculum '{name}'  ({len(stages)} stages)")
        return stages

    @staticmethod
    def list_curriculums():
        """print all curriculum folders found under the curriculums root."""
        if not os.path.isdir(_CURRICULUMS_ROOT):
            print(f"[CurriculumTester] no curriculums directory found at {_CURRICULUMS_ROOT}")
            return
        entries = sorted(
            e for e in os.listdir(_CURRICULUMS_ROOT)
            if os.path.isdir(os.path.join(_CURRICULUMS_ROOT, e))
        )
        if not entries:
            print(f"[CurriculumTester] no saved curriculums found in {_CURRICULUMS_ROOT}")
            return
        print(f"\n[CurriculumTester] saved curriculums in {_CURRICULUMS_ROOT}:")
        for name in entries:
            d          = _curriculum_dir(name)
            has_json   = os.path.exists(_curriculum_json_path(name))
            has_model  = (
                os.path.exists(_final_model_path(name) + ".zip") or
                os.path.exists(_best_model_path(name))
            )
            tags = []
            if has_json:  tags.append("curriculum.json ✓")
            if has_model: tags.append("model ✓")
            print(f"  {name:<30}  {', '.join(tags) if tags else '(empty)'}")
        print()

    # ------------------------------------------------------------------
    # internal env / callback factory (mirrors train.py helpers but scoped
    # to a specific curriculum and save_path)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_env_factory(curriculum):
        """returns an env factory compatible with make_vec_env."""
        _gen = PromptGenerator()

        def _init():
            if curriculum is not None:
                prompt = curriculum.sample_prompt()
                n_shp  = curriculum.sample_n_shapes()
            else:
                prompt = _gen.sample()
                n_shp  = None

            goal    = parse_goal(prompt)
            raw_emb  = get_embedding(prompt)
            env = ShapeEnv(n_shapes=n_shp, goal=goal, goal_embedding=raw_emb)
            return Monitor(env)

        return _init

    @staticmethod
    def _build_callbacks(save_path: str, n_envs: int,
                         curriculum) -> CallbackList:
        """mirrors build_callbacks from train.py, scoped to save_path."""

        def _static_eval_env():
            _gen = PromptGenerator()
            import numpy as np
            prompt = curriculum.sample_prompt() if curriculum else _gen.sample()
            n_shp  = curriculum.sample_n_shapes() if curriculum else None
            goal   = parse_goal(prompt)
            raw_emb = get_embedding(prompt)
            env = ShapeEnv(n_shapes=n_shp, goal=goal, goal_embedding=raw_emb)
            return Monitor(env)

        eval_cb = EvalCallback(
            _static_eval_env(),
            best_model_save_path=save_path,
            log_path=os.path.join(save_path, "logs"),
            eval_freq=max(5_000 // n_envs, 1),
            n_eval_episodes=10,
            verbose=1,
        )
        task_cb = ShapeTaskCallback(
            curriculum=curriculum,
            eval_freq=5_000,
            n_eval_episodes=10,
            verbose=1,
        )
        cbs = [eval_cb, task_cb]

        if curriculum is not None:
            curr_cb = CurriculumCallback(
                curriculum=curriculum,
                eval_freq=5_000,
                n_eval_episodes=30,
                verbose=1,
                save_path=save_path,
            )
            summary_cb = TrainingSummaryCallback(
                curriculum_cb=curr_cb,
                task_cb=task_cb,
                summary_freq=50_000,
            )
            cbs += [curr_cb, summary_cb]

        return CallbackList(cbs)

    @staticmethod
    def _write_training_config(save_path: str, curriculum, timesteps: int):
        _ensure_dir(save_path)
        n_shapes = curriculum.n_shapes_range[1] if curriculum else MAX_SHAPES
        tasks    = curriculum.active_tasks      if curriculum else SUPPORTED_TASKS
        config   = {"n_shapes": n_shapes, "tasks": tasks, "timesteps": timesteps}
        path     = os.path.join(save_path, "training_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=3)
        print(f"[CurriculumTester] training config saved → {path}")

    # ------------------------------------------------------------------
    # test_curriculum — train a fresh model on a named curriculum
    # ------------------------------------------------------------------

    def test_curriculum(
        self,
        curriculum_name: str,
        timesteps:       int  = 800_000,
        bc_episodes:     int  = 500,
        bc_epochs:       int  = 30,
        use_oracle:      bool = True,
        start_stage:     int  = 0,
        resume_model:    str  = None,
    ):
        """
        train a model on the named curriculum and save everything to
        ./models/curriculums/<curriculum_name>/.

        args:
            curriculum_name : folder name under ./models/curriculums/
                              must contain a curriculum.json file
                              (create one with CurriculumTester.save_curriculum)
            timesteps       : total PPO training steps
            bc_episodes     : oracle demo episodes for BC warm-start
            bc_epochs       : BC training epochs
            use_oracle      : if True, BC warm-start before PPO
            start_stage     : skip to this stage index (useful for resuming)
            resume_model    : path to a .zip checkpoint to resume from
                              (skips BC init if provided)

        returns:
            (model)
        """
        from bc_train import BicameralPolicy, train_bc, build_ppo_from_bc
        from oracle import collect_demonstrations

        save_path = _curriculum_dir(curriculum_name)
        _ensure_dir(save_path)

        print(f"\n{'='*60}")
        print(f"  CurriculumTester — training '{curriculum_name}'")
        print(f"  save path : {save_path}")
        print(f"  timesteps : {timesteps:,}")
        print(f"{'='*60}\n")

        # --- load curriculum stages from json ---
        stages     = CurriculumTester.load_curriculum_stages(curriculum_name)
        curriculum = _JsonCurriculumManager(stages, verbose=True, start_stage=start_stage)
        print(f"[curriculum] initial stage: {curriculum.status()}\n")

        self._write_training_config(save_path, curriculum, timesteps)

        n_envs  = N_ENVS
        vec_env = make_vec_env(
            self._make_env_factory(curriculum), n_envs=n_envs
        )

        # --- model initialisation ---
        if resume_model is not None:
            print(f"[train] resuming from checkpoint: {resume_model}")
            model = PPO.load(resume_model, env=vec_env)

        elif use_oracle:
            print(f"[train] collecting {bc_episodes} oracle demonstrations …")
            dataset = collect_demonstrations(
                n_episodes=bc_episodes,
                verbose=True,
            )
            device     = "cuda" if torch.cuda.is_available() else "cpu"
            bc_network = train_bc(
                dataset=dataset,
                save_path=save_path,
                epochs=bc_epochs,
                device=device,
            )
            print("[train] initialising PPO from BC weights …")
            model = build_ppo_from_bc(bc_network, n_shapes=MAX_SHAPES, vec_env=vec_env)

        else:
            print("[train] initialising PPO from random weights …")
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
                tensorboard_log=os.path.join(save_path, "tensorboard"),
            )


        callbacks = self._build_callbacks(save_path, n_envs, curriculum)

        print(f"\n[train] starting PPO — {timesteps:,} timesteps\n")
        model.learn(total_timesteps=timesteps, callback=callbacks)

        # --- save final model inside the curriculum folder ---
        final_path = _final_model_path(curriculum_name)
        model.save(final_path)
        self._write_training_config(save_path, curriculum, timesteps)

        print(f"\n[CurriculumTester] training complete.")
        print(f"  final model → {final_path}.zip")
        print(f"  best model  → {_best_model_path(curriculum_name)}")

        return model

    # ------------------------------------------------------------------
    # run_model — load a trained model and run episodes
    # ------------------------------------------------------------------

    def run_model(
        self,
        curriculum_name: str,
        n_episodes:      int  = 5,
        prefer_best:     bool = True,
        render:          bool = True,
        verbose:         bool = True,
    ) -> list[dict]:
        """
        load a trained model for the named curriculum and run n_episodes.

        args:
            curriculum_name : folder name under ./models/curriculums/
            n_episodes      : number of episodes to run
            prefer_best     : if True, load best_model.zip; else final_model.zip
            render          : if True, call env.render() each step
            verbose         : if True, print per-episode stats

        returns:
            list of episode result dicts:
            [{"episode": i, "reward": float, "steps": int, "success": bool}, ...]
        """
        import numpy as np

        # --- resolve model path ---
        best_path  = _best_model_path(curriculum_name)
        final_path = _final_model_path(curriculum_name) + ".zip"

        if prefer_best and os.path.exists(best_path):
            model_path = best_path
            label      = "best_model"
        elif os.path.exists(final_path):
            model_path = final_path
            label      = "final_model"
        else:
            raise FileNotFoundError(
                f"no trained model found for curriculum '{curriculum_name}'.\n"
                f"  checked: {best_path}\n"
                f"           {final_path}\n"
                f"  run test_curriculum('{curriculum_name}') first."
            )

        print(f"\n[CurriculumTester] running '{curriculum_name}' ({label})")
        print(f"  model path : {model_path}")
        print(f"  episodes   : {n_episodes}\n")

        # --- load curriculum stages to sample representative tasks ---
        stages     = CurriculumTester.load_curriculum_stages(curriculum_name)
        curriculum = _JsonCurriculumManager(stages, verbose=False,
                                            start_stage=len(stages) - 1)  # final stage

        model = PPO.load(model_path)

        results = []

        for ep in range(n_episodes):
            # build a fresh env for this episode
            prompt  = curriculum.sample_prompt()
            n_shp   = curriculum.sample_n_shapes()
            goal    = parse_goal(prompt)
            raw_emb = get_embedding(prompt)

            env = ShapeEnv(n_shapes=n_shp, goal=goal, goal_embedding=raw_emb)

            obs, _  = env.reset()
            done    = False
            total_r = 0.0
            steps   = 0
            success = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_r += reward
                steps   += 1
                done     = terminated or truncated

                if render:
                    env.render()

                if info.get("success", False):
                    success = True

            result = {"episode": ep + 1, "reward": round(total_r, 3),
                      "steps": steps, "success": success}
            results.append(result)

            if verbose:
                status = "✓ success" if success else "✗ failed"
                print(f"  ep {ep+1:>3} | {status} | reward={total_r:>8.3f} | steps={steps}")

            env.close()

        # --- summary ---
        if verbose and n_episodes > 1:
            n_success  = sum(r["success"] for r in results)
            mean_r     = sum(r["reward"]  for r in results) / n_episodes
            mean_steps = sum(r["steps"]   for r in results) / n_episodes
            print(f"\n  summary over {n_episodes} episodes:")
            print(f"    success rate : {n_success}/{n_episodes} "
                  f"({100*n_success/n_episodes:.0f}%)")
            print(f"    mean reward  : {mean_r:.3f}")
            print(f"    mean steps   : {mean_steps:.1f}\n")

        return results

    # ------------------------------------------------------------------
    # convenience: save performance metrics after a run
    # ------------------------------------------------------------------

    def save_performance_metrics(self, curriculum_name: str, results: list[dict]):
        """
        write episode results from run_model() to a json file alongside
        the curriculum and model in ./models/curriculums/<name>/.

        args:
            curriculum_name : folder name
            results         : list of dicts returned by run_model()
        """
        save_path = _curriculum_dir(curriculum_name)
        _ensure_dir(save_path)

        n          = len(results)
        n_success  = sum(r["success"] for r in results)
        mean_r     = sum(r["reward"]  for r in results) / n if n else 0
        mean_steps = sum(r["steps"]   for r in results) / n if n else 0

        payload = {
            "curriculum":   curriculum_name,
            "n_episodes":   n,
            "success_rate": round(n_success / n, 4) if n else 0,
            "mean_reward":  round(mean_r, 4),
            "mean_steps":   round(mean_steps, 2),
            "episodes":     results,
        }

        path = os.path.join(save_path, "performance_metrics.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=3)
        print(f"[CurriculumTester] performance metrics saved → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from curriculum import _STAGES as DEFAULT_STAGES

    parser = argparse.ArgumentParser(description="CurriculumTester CLI")
    sub    = parser.add_subparsers(dest="command")

    # --- save ---
    p_save = sub.add_parser("save", help="save a curriculum json from the default _STAGES")
    p_save.add_argument("name", help="curriculum name (folder will be created)")

    # --- train ---
    p_train = sub.add_parser("train", help="train a model on a saved curriculum")
    p_train.add_argument("name",                          help="curriculum name")
    p_train.add_argument("--timesteps",  type=int, default=800_000)
    p_train.add_argument("--bc-episodes",type=int, default=500)
    p_train.add_argument("--bc-epochs",  type=int, default=30)
    p_train.add_argument("--no-oracle",  action="store_true")
    p_train.add_argument("--start-stage",type=int, default=0)
    p_train.add_argument("--resume",     type=str, default=None)

    # --- run ---
    p_run = sub.add_parser("run", help="run a trained model")
    p_run.add_argument("name",                            help="curriculum name")
    p_run.add_argument("--episodes",   type=int, default=5)
    p_run.add_argument("--no-render",  action="store_true")
    p_run.add_argument("--final",      action="store_true",
                       help="load final_model instead of best_model")

    # --- list ---
    sub.add_parser("list", help="list all saved curriculums")

    args = parser.parse_args()

    tester = CurriculumTester()

    if args.command == "save":
        CurriculumTester.save_curriculum(args.name, DEFAULT_STAGES)

    elif args.command == "train":
        tester.test_curriculum(
            curriculum_name=args.name,
            timesteps=args.timesteps,
            bc_episodes=args.bc_episodes,
            bc_epochs=args.bc_epochs,
            use_oracle=not args.no_oracle,
            start_stage=args.start_stage,
            resume_model=args.resume,
        )

    elif args.command == "run":
        results = tester.run_model(
            curriculum_name=args.name,
            n_episodes=args.episodes,
            prefer_best=not args.final,
            render=not args.no_render,
        )
        tester.save_performance_metrics(args.name, results)

    elif args.command == "list":
        CurriculumTester.list_curriculums()

    else:
        parser.print_help()