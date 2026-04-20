"""
prompt_train.py

prompt pretraining — teaches the right stream encoder to understand
natural language prompts before BC or PPO training begins.

CLASSIFIER_TASKS now includes all 8 real tasks:
   move_cardinal, move_diagonal, approach,
   click_at, hold_at, reach, touch, drag

right stream input is 389-dim:
   cols [:, :5]   — shape layout (ALL_SHAPES_DIM=5)
   cols [:, 5:]   — raw embedding (EMBEDDING_DIM=384)

transplant copies classifier encoder weights into the embedding columns.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import SUPPORTED_TASKS, POLICY_HIDDEN_SIZE, EMBEDDING_DIM, ALL_SHAPES_DIM
from prompt_gen import PromptGenerator
from llm_goal_parser import get_embedding

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

CLASSIFIER_TASKS = [t for t in SUPPORTED_TASKS if t != "none"]
N_TASKS          = len(CLASSIFIER_TASKS)
TASK_TO_IDX      = {t: i for i, t in enumerate(CLASSIFIER_TASKS)}
IDX_TO_TASK      = {i: t for t, i in TASK_TO_IDX.items()}

RIGHT_SHAPE_COLS = ALL_SHAPES_DIM   # = 5


# ---------------------------------------------------------------------------
# TaskClassifierNetwork
# ---------------------------------------------------------------------------

class TaskClassifierNetwork(nn.Module):
   """
   lightweight classifier: 384-dim embedding → task label.
   encoder mirrors BicameralNetwork.right_encoder for weight transplant.
   """

   def __init__(self, hidden: int = POLICY_HIDDEN_SIZE, n_tasks: int = N_TASKS):
      super().__init__()
      self.hidden  = hidden
      self.n_tasks = n_tasks
      self.encoder = nn.Sequential(
         nn.Linear(EMBEDDING_DIM, hidden),
         nn.Tanh(),
         nn.Linear(hidden, hidden),
         nn.Tanh(),
      )
      self.classifier = nn.Linear(hidden, n_tasks)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.classifier(self.encoder(x))

   def encode(self, x: torch.Tensor) -> torch.Tensor:
      return self.encoder(x)


# ---------------------------------------------------------------------------
# dataset generation
# ---------------------------------------------------------------------------

def build_dataset(
   n_samples:  int  = 10_000,
   verbose:    bool = True,
   cache_path: str  = None,
) -> dict:
   if cache_path is not None and os.path.exists(cache_path):
      if verbose:
         print(f"[prompt_train] loading cached dataset from {cache_path}")
      data = np.load(cache_path, allow_pickle=True)
      # validate cached N_TASKS matches current
      cached_tasks = list(data.get("task_names", []))
      cached_unique = len(set(cached_tasks))
      if cached_unique != N_TASKS:
         if verbose:
            print(f"[prompt_train] cache has {cached_unique} tasks but current "
                  f"N_TASKS={N_TASKS} — regenerating dataset")
      else:
         return {
            "embeddings": data["embeddings"],
            "labels":     data["labels"],
            "task_names": cached_tasks,
         }

   if verbose:
      print(f"\n[prompt_train] generating dataset — {n_samples:,} samples "
            f"across {N_TASKS} tasks")
      print(f"  tasks: {CLASSIFIER_TASKS}")

   gen       = PromptGenerator()
   per_task  = n_samples // N_TASKS
   remainder = n_samples - per_task * N_TASKS

   embeddings = []
   labels     = []
   task_names = []

   for task_idx, task in enumerate(CLASSIFIER_TASKS):
      n = per_task + (1 if task_idx < remainder else 0)
      if verbose:
         print(f"  [{task_idx+1}/{N_TASKS}] {task:<25} generating {n} samples...")
      for _ in range(n):
         prompt    = gen.sample(task)
         embedding = get_embedding(prompt)
         embeddings.append(embedding)
         labels.append(TASK_TO_IDX[task])
         task_names.append(task)

   embeddings = np.array(embeddings, dtype=np.float32)
   labels     = np.array(labels,     dtype=np.int64)

   if verbose:
      print(f"\n  dataset complete: {len(embeddings):,} samples")

   if cache_path is not None:
      os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
      np.savez(cache_path, embeddings=embeddings, labels=labels,
               task_names=np.array(task_names))
      if verbose:
         print(f"  dataset cached → {cache_path}")

   return {"embeddings": embeddings, "labels": labels, "task_names": task_names}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_classifier(
   dataset:    dict,
   hidden:     int   = POLICY_HIDDEN_SIZE,
   epochs:     int   = 40,
   batch_size: int   = 256,
   lr:         float = 3e-4,
   device:     str   = "cpu",
   verbose:    bool  = True,
) -> TaskClassifierNetwork:

   emb_t   = torch.tensor(dataset["embeddings"], dtype=torch.float32).to(device)
   label_t = torch.tensor(dataset["labels"],     dtype=torch.int64).to(device)

   if verbose:
      print(f"\n[prompt_train] training classifier")
      print(f"  samples : {len(emb_t):,}  n_tasks : {N_TASKS}  ({CLASSIFIER_TASKS})")

   network   = TaskClassifierNetwork(hidden=hidden, n_tasks=N_TASKS).to(device)
   optimizer = torch.optim.Adam(network.parameters(), lr=lr)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-5)
   loader    = DataLoader(TensorDataset(emb_t, label_t),
                          batch_size=batch_size, shuffle=True)

   best_acc   = 0.0
   best_state = None

   for epoch in range(1, epochs + 1):
      network.train()
      epoch_loss = 0.0
      n_correct  = n_total = 0
      for emb_batch, label_batch in loader:
         logits = network(emb_batch)
         loss   = F.cross_entropy(logits, label_batch)
         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
         optimizer.step()
         epoch_loss += loss.item()
         preds       = logits.argmax(dim=-1)
         n_correct  += (preds == label_batch).sum().item()
         n_total    += len(label_batch)
      scheduler.step()
      acc = n_correct / n_total
      if acc > best_acc:
         best_acc   = acc
         best_state = {k: v.clone() for k, v in network.state_dict().items()}
      if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == 1):
         print(f"  epoch {epoch:3d}/{epochs}  "
               f"loss: {epoch_loss/len(loader):.4f}  "
               f"acc: {acc:.1%}  best: {best_acc:.1%}")

   if best_state is not None:
      network.load_state_dict(best_state)
      if verbose:
         print(f"\n  best accuracy: {best_acc:.1%}")

   return network.cpu().eval()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
   network: TaskClassifierNetwork,
   dataset: dict,
   verbose: bool = True,
) -> dict:
   network.eval()
   embeddings = torch.tensor(dataset["embeddings"], dtype=torch.float32)
   labels     = torch.tensor(dataset["labels"],     dtype=torch.int64)
   with torch.no_grad():
      preds = network(embeddings).argmax(dim=-1)
   results         = {}
   overall_correct = 0
   if verbose:
      print(f"\n[prompt_train] classifier evaluation")
   for idx, task in IDX_TO_TASK.items():
      mask      = (labels == idx)
      n_total   = mask.sum().item()
      n_correct = (preds[mask] == idx).sum().item()
      acc       = n_correct / max(n_total, 1)
      results[task] = acc
      overall_correct += n_correct
      if verbose:
         bar = "█" * int(acc * 20)
         print(f"  {task:<25} {n_correct:>7,}  {n_total:>7,}  {acc:>7.1%}  {bar}")
   overall = overall_correct / max(len(labels), 1)
   results["overall"] = overall
   if verbose:
      print(f"  OVERALL  {overall_correct:>7,}  {len(labels):>7,}  {overall:>7.1%}\n")
   return results


# ---------------------------------------------------------------------------
# transplant
# ---------------------------------------------------------------------------

def transplant_into_bicameral(
   classifier,
   bc_network = None,
   verbose:   bool = True,
) -> "BicameralNetwork":
   from bc_train import BicameralNetwork
   if bc_network is None:
      bc_network = BicameralNetwork()
      if verbose:
         print("[prompt_train] created fresh BicameralNetwork for transplant")
   if verbose:
      print("\n[prompt_train] transplanting classifier → right stream encoder")

   classifier_enc = classifier.encoder
   right_enc      = bc_network.right_encoder

   with torch.no_grad():
      c_weight_0 = classifier_enc[0].weight   # (hidden, 384)
      c_bias_0   = classifier_enc[0].bias
      r_weight_0 = right_enc[0].weight         # (hidden, 389)

      if c_weight_0.shape[0] != r_weight_0.shape[0]:
         raise ValueError(
            f"hidden mismatch: {c_weight_0.shape[0]} vs {r_weight_0.shape[0]}")

      right_enc[0].weight[:, RIGHT_SHAPE_COLS:].copy_(c_weight_0)
      right_enc[0].bias.copy_(c_bias_0)
      if verbose:
         print(f"  layer 0: ({c_weight_0.shape}) → cols [{RIGHT_SHAPE_COLS}:] "
               f"of ({r_weight_0.shape})  ✓")

      c_weight_2 = classifier_enc[2].weight
      c_bias_2   = classifier_enc[2].bias
      r_weight_2 = right_enc[2].weight
      if c_weight_2.shape != r_weight_2.shape:
         raise ValueError(f"layer 2 mismatch: {c_weight_2.shape} vs {r_weight_2.shape}")
      right_enc[2].weight.copy_(c_weight_2)
      right_enc[2].bias.copy_(c_bias_2)
      if verbose:
         print(f"  layer 2: ({c_weight_2.shape}) → ({r_weight_2.shape})  ✓")

   if verbose:
      print("  transplant complete.")
   return bc_network


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify_transplant(classifier, bc_network, n_probes=100, verbose=True) -> bool:
   gen = PromptGenerator()
   classifier.eval()
   bc_network.eval()
   if verbose:
      print(f"\n[prompt_train] verifying transplant ({n_probes} probes)...")
   max_diff = 0.0
   n_failed = 0
   with torch.no_grad():
      for _ in range(n_probes):
         emb_384  = torch.tensor(
            get_embedding(gen.sample()), dtype=torch.float32).unsqueeze(0)
         cls_out  = classifier.encoder(emb_384)
         inp_389  = torch.zeros(1, 389)
         inp_389[:, RIGHT_SHAPE_COLS:] = emb_384
         right_out = bc_network.right_encoder(inp_389)
         diff      = (cls_out - right_out).abs().max().item()
         max_diff  = max(max_diff, diff)
         if diff > 1e-4:
            n_failed += 1
   passed = n_failed == 0
   if verbose:
      status = "✓ PASSED" if passed else f"✗ FAILED ({n_failed}/{n_probes})"
      print(f"  max diff: {max_diff:.2e}  {status}\n")
   return passed


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save_classifier(classifier, save_path: str, metrics: dict = None):
   os.makedirs(save_path, exist_ok=True)
   torch.save(classifier.state_dict(), os.path.join(save_path, "prompt_classifier.pt"))
   print(f"[prompt_train] classifier saved → {save_path}/prompt_classifier.pt")
   if metrics is not None:
      with open(os.path.join(save_path, "prompt_classifier_metrics.json"), "w") as f:
         json.dump(metrics, f, indent=3)


def load_classifier(save_path: str, hidden: int = POLICY_HIDDEN_SIZE) -> TaskClassifierNetwork:
   weights_path = os.path.join(save_path, "prompt_classifier.pt")
   if not os.path.exists(weights_path):
      raise FileNotFoundError(f"no classifier at {weights_path}")
   network = TaskClassifierNetwork(hidden=hidden, n_tasks=N_TASKS)
   network.load_state_dict(torch.load(weights_path, map_location="cpu"))
   network.eval()
   print(f"[prompt_train] classifier loaded from {weights_path}")
   return network


# ---------------------------------------------------------------------------
# train_prompt — main entry point
# ---------------------------------------------------------------------------

def train_prompt(
   save_path:      str  = "./models/phase0",
   n_samples:      int  = 10_000,
   epochs:         int  = 40,
   hidden:         int  = POLICY_HIDDEN_SIZE,
   device:         str  = None,
   verbose:        bool = True,
   use_cache:      bool = True,
   skip_if_exists: bool = True,
) -> "BicameralNetwork":
   if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
   os.makedirs(save_path, exist_ok=True)

   cache_path = os.path.join(save_path, "dataset_cache.npz") if use_cache else None
   dataset    = build_dataset(n_samples=n_samples, verbose=verbose,
                              cache_path=cache_path)

   weights_path = os.path.join(save_path, "prompt_classifier.pt")

   # check if existing classifier has the right number of output classes
   need_retrain = True
   if skip_if_exists and os.path.exists(weights_path):
      try:
         ckpt = torch.load(weights_path, map_location="cpu")
         # classifier.weight shape: (n_tasks, hidden)
         saved_n_tasks = ckpt["classifier.weight"].shape[0]
         if saved_n_tasks == N_TASKS:
            if verbose:
               print(f"[prompt_train] classifier found ({N_TASKS} tasks) — "
                     f"skipping training")
            need_retrain = False
         else:
            if verbose:
               print(f"[prompt_train] cached classifier has {saved_n_tasks} tasks "
                     f"but N_TASKS={N_TASKS} — retraining")
      except Exception:
         pass

   if need_retrain:
      classifier = train_classifier(dataset=dataset, hidden=hidden,
                                    epochs=epochs, device=device, verbose=verbose)
   else:
      classifier = load_classifier(save_path, hidden=hidden)

   metrics = evaluate_classifier(classifier, dataset, verbose=verbose)
   if metrics["overall"] < 0.70:
      print(f"  ⚠  accuracy {metrics['overall']:.1%} < 70% — "
            f"consider more samples or epochs.")
   save_classifier(classifier, save_path, metrics)

   from bc_train import BicameralNetwork
   bc_network = transplant_into_bicameral(
      classifier=classifier, bc_network=BicameralNetwork(), verbose=verbose)
   verify_transplant(classifier, bc_network, verbose=verbose)

   transplant_path = os.path.join(save_path, "transplanted_bc_network.pt")
   torch.save(bc_network.state_dict(), transplant_path)
   if verbose:
      print(f"[prompt_train] transplanted network saved → {transplant_path}\n")

   return bc_network


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--save",      type=str,  default="./models/phase0")
   parser.add_argument("--samples",   type=int,  default=10_000)
   parser.add_argument("--epochs",    type=int,  default=40)
   parser.add_argument("--hidden",    type=int,  default=POLICY_HIDDEN_SIZE)
   parser.add_argument("--no-cache",  action="store_true")
   parser.add_argument("--retrain",   action="store_true")
   parser.add_argument("--verify",    action="store_true")
   parser.add_argument("--eval-only", action="store_true")
   args   = parser.parse_args()
   device = "cuda" if torch.cuda.is_available() else "cpu"

   if args.verify:
      from bc_train import BicameralNetwork
      clf = load_classifier(args.save, hidden=args.hidden)
      bc  = BicameralNetwork()
      transplant_into_bicameral(clf, bc, verbose=True)
      verify_transplant(clf, bc, verbose=True)
   elif args.eval_only:
      clf     = load_classifier(args.save, hidden=args.hidden)
      cache   = None if args.no_cache else os.path.join(args.save, "dataset_cache.npz")
      dataset = build_dataset(n_samples=2_000, verbose=True, cache_path=cache)
      evaluate_classifier(clf, dataset, verbose=True)
   else:
      train_prompt(
         save_path=args.save, n_samples=args.samples, epochs=args.epochs,
         hidden=args.hidden, device=device,
         use_cache=not args.no_cache, skip_if_exists=not args.retrain,
      )