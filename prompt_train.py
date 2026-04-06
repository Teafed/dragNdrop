"""
prompt_train.py

phase 0 pretraining — teaches the right stream encoder to understand
natural language prompts before BC or PPO training begins.

--- motivation ---
   the right stream encoder in BicameralNetwork receives a 414-dim input:
      obs[14:428] = shape_layout (30-dim) + raw_embedding (384-dim)

   without pretraining, the right stream starts from random weights and
   must simultaneously learn:
      1. what the 384-dim embedding means as a task description
      2. how shapes relate to each other spatially
      3. how to combine both into useful action features
   all from sparse PPO reward alone — very slow.

   phase 0 fixes problem 1 explicitly: we train a small classifier to
   predict task name directly from the 384-dim embedding. the classifier's
   learned weights are then transplanted into the embedding-processing
   columns of the right stream encoder, so BC/PPO starts with a right
   stream that already "speaks the language" of prompts.

--- architecture ---
   TaskClassifierNetwork:
      384-dim embedding
         → Linear(384, hidden)  + Tanh     ← these weights get transplanted
         → Linear(hidden, hidden) + Tanh   ← these weights get transplanted
         → Linear(hidden, n_tasks)          ← classification head, discarded

   transplant target (BicameralNetwork.right_encoder):
      Linear(414, hidden):  weight shape is (hidden, 414)
         columns [:, 30:]   correspond to the 384-dim embedding portion
         columns [:, :30]   correspond to the 30-dim shape layout portion
      transplant copies classifier layer1 weights into columns [:, 30:]
      shape layout columns [:, :30] stay randomly initialized

--- usage ---

   # train the classifier and transplant weights into a BicameralNetwork
   python prompt_train.py

   # control training
   python prompt_train.py --samples 20000 --epochs 50 --hidden 256

   # skip transplant (just train and save classifier weights)
   python prompt_train.py --no-transplant

   # verify transplant worked
   python prompt_train.py --verify

   # in your training pipeline (before bc_train.py):
   from prompt_train import run_phase0
   bc_network = run_phase0()          # returns BicameralNetwork with transplanted weights
   # then pass bc_network to train_bc() as usual

--- integration with train.py ---
   in train():
      from prompt_train import run_phase0
      bc_network = run_phase0(save_path=save_path)
      # pass bc_network into build_ppo_from_bc() or use as BC init
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import SUPPORTED_TASKS, POLICY_HIDDEN_SIZE
from prompt_gen import PromptGenerator
from llm_goal_parser import get_embedding

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# tasks the classifier is trained on — subset of SUPPORTED_TASKS
# excludes "none" since it is not a real manipulation task
CLASSIFIER_TASKS = [t for t in SUPPORTED_TASKS if t != "none"]
N_TASKS          = len(CLASSIFIER_TASKS)
TASK_TO_IDX      = {t: i for i, t in enumerate(CLASSIFIER_TASKS)}
IDX_TO_TASK      = {i: t for t, i in TASK_TO_IDX.items()}

EMBEDDING_DIM    = 384    # all-MiniLM-L6-v2 output dimension
RIGHT_SHAPE_COLS = 30     # shape layout columns in right stream input (obs[14:44])

# ---------------------------------------------------------------------------
# TaskClassifierNetwork
# ---------------------------------------------------------------------------

class TaskClassifierNetwork(nn.Module):
    """
    lightweight classifier: 384-dim embedding → task label.

    the two encoder layers intentionally mirror the right stream encoder
    in BicameralNetwork so their weights can be transplanted directly.
    hidden size defaults to POLICY_HIDDEN_SIZE to guarantee shape match.

    architecture:
        Linear(384, hidden) + Tanh   ← transplant layer 0
        Linear(hidden, hidden) + Tanh ← transplant layer 1
        Linear(hidden, n_tasks)       ← classification head (discarded after training)
    """

    def __init__(self, hidden: int = POLICY_HIDDEN_SIZE, n_tasks: int = N_TASKS):
        super().__init__()
        self.hidden  = hidden
        self.n_tasks = n_tasks

        # encoder layers — match right_encoder structure in BicameralNetwork
        self.encoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # classification head — discarded after training
        self.classifier = nn.Linear(hidden, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 384) → logits: (batch, n_tasks)"""
        features = self.encoder(x)
        return self.classifier(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """return encoder features without classification head."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# dataset generation
# ---------------------------------------------------------------------------

def build_dataset(
    n_samples:   int  = 10_000,
    verbose:     bool = True,
    cache_path:  str  = None,
) -> dict:
    """
    generate (embedding, task_label) pairs using PromptGenerator.

    samples are balanced across tasks — each task gets n_samples // N_TASKS
    samples, then remaining samples are distributed randomly to avoid bias.

    args:
        n_samples:  total number of (embedding, label) pairs to generate
        verbose:    print progress
        cache_path: if set, save/load dataset to/from this .npz file
                    useful for avoiding re-embedding on repeated runs

    returns:
        dict with keys:
            "embeddings": np.ndarray (n_samples, 384) float32
            "labels":     np.ndarray (n_samples,)     int64
            "task_names": list of task name strings (length n_samples)
    """
    # --- load from cache if available ---
    if cache_path is not None and os.path.exists(cache_path):
        if verbose:
            print(f"[phase0] loading cached dataset from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "labels":     data["labels"],
            "task_names": list(data["task_names"]),
        }

    if verbose:
        print(f"\n[phase0] generating dataset — {n_samples:,} samples "
              f"across {N_TASKS} tasks")
        print(f"  tasks: {CLASSIFIER_TASKS}")

    gen            = PromptGenerator()
    per_task       = n_samples // N_TASKS
    remainder      = n_samples - per_task * N_TASKS

    embeddings = []
    labels     = []
    task_names = []

    for task_idx, task in enumerate(CLASSIFIER_TASKS):
        # add one extra sample to first `remainder` tasks
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
        print(f"\n  dataset complete: {len(embeddings):,} samples  "
              f"embedding_dim={embeddings.shape[1]}")

    # --- save to cache ---
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path,
                 embeddings=embeddings,
                 labels=labels,
                 task_names=np.array(task_names))
        if verbose:
            print(f"  dataset cached → {cache_path}")

    return {
        "embeddings": embeddings,
        "labels":     labels,
        "task_names": task_names,
    }


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
    """
    train TaskClassifierNetwork on (embedding, task_label) pairs.

    loss: cross-entropy over CLASSIFIER_TASKS classes.
    scheduler: cosine annealing with warm restarts.
    returns trained network on cpu in eval mode.

    args:
        dataset:    dict from build_dataset()
        hidden:     hidden size — must match POLICY_HIDDEN_SIZE for transplant
        epochs:     training epochs
        batch_size: minibatch size
        lr:         initial learning rate
        device:     "cpu" or "cuda"
        verbose:    print per-epoch metrics
    """
    emb_t   = torch.tensor(dataset["embeddings"], dtype=torch.float32).to(device)
    label_t = torch.tensor(dataset["labels"],     dtype=torch.int64).to(device)

    # --- class balance check ---
    if verbose:
        print(f"\n[phase0] training classifier")
        print(f"  samples    : {len(emb_t):,}")
        print(f"  hidden     : {hidden}")
        print(f"  epochs     : {epochs}")
        print(f"  batch_size : {batch_size}")
        print(f"  lr         : {lr}")
        print(f"  device     : {device}")
        print(f"  n_tasks    : {N_TASKS}  ({CLASSIFIER_TASKS})\n")
        counts = {IDX_TO_TASK[i]: int((label_t == i).sum()) for i in range(N_TASKS)}
        for task, n in counts.items():
            print(f"  {task:<25} {n:>6,} samples")
        print()

    network   = TaskClassifierNetwork(hidden=hidden, n_tasks=N_TASKS).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    loader = DataLoader(
        TensorDataset(emb_t, label_t),
        batch_size=batch_size,
        shuffle=True,
    )

    best_acc  = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        network.train()
        epoch_loss = 0.0
        n_correct  = 0
        n_total    = 0

        for emb_batch, label_batch in loader:
            logits = network(emb_batch)
            loss   = F.cross_entropy(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds      = logits.argmax(dim=-1)
            n_correct  += (preds == label_batch).sum().item()
            n_total    += len(label_batch)

        scheduler.step()

        acc     = n_correct / n_total
        avg_loss = epoch_loss / len(loader)
        cur_lr   = scheduler.get_last_lr()[0]

        # save best checkpoint
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in network.state_dict().items()}

        if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"loss: {avg_loss:.4f}  "
                  f"acc: {acc:.1%}  "
                  f"best: {best_acc:.1%}  "
                  f"lr: {cur_lr:.2e}")

    # restore best weights
    if best_state is not None:
        network.load_state_dict(best_state)
        if verbose:
            print(f"\n  best accuracy: {best_acc:.1%}  (restored best checkpoint)")

    return network.cpu().eval()


# ---------------------------------------------------------------------------
# per-task accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    network: TaskClassifierNetwork,
    dataset: dict,
    verbose: bool = True,
) -> dict:
    """
    evaluate classifier accuracy per task and overall.

    returns dict:
        {task_name: accuracy, ..., "overall": accuracy}
    """
    network.eval()
    embeddings = torch.tensor(dataset["embeddings"], dtype=torch.float32)
    labels     = torch.tensor(dataset["labels"],     dtype=torch.int64)

    with torch.no_grad():
        logits = network(embeddings)
        preds  = logits.argmax(dim=-1)

    results     = {}
    overall_correct = 0

    if verbose:
        print(f"\n[phase0] classifier evaluation")
        print(f"  {'task':<25} {'correct':>8}  {'total':>8}  {'accuracy':>9}")
        print(f"  {'-'*55}")

    for idx, task in IDX_TO_TASK.items():
        mask      = (labels == idx)
        n_total   = mask.sum().item()
        n_correct = (preds[mask] == idx).sum().item()
        acc       = n_correct / max(n_total, 1)
        results[task] = acc
        overall_correct += n_correct

        if verbose:
            bar = "█" * int(acc * 20)
            print(f"  {task:<25} {n_correct:>8,}  {n_total:>8,}  "
                  f"{acc:>8.1%}  {bar}")

    overall = overall_correct / max(len(labels), 1)
    results["overall"] = overall

    if verbose:
        print(f"  {'-'*55}")
        print(f"  {'OVERALL':<25} {overall_correct:>8,}  "
              f"{len(labels):>8,}  {overall:>8.1%}\n")

    return results


# ---------------------------------------------------------------------------
# weight transplant
# ---------------------------------------------------------------------------

def transplant_into_bicameral(
    classifier:    TaskClassifierNetwork,
    bc_network:    "BicameralNetwork" = None,
    verbose:       bool = True,
) -> "BicameralNetwork":
    """
    transplant classifier encoder weights into the right stream encoder
    of a BicameralNetwork.

    the right stream encoder input is 414-dim:
        columns [:, :30]   — shape layout (obs[14:44])
        columns [:, 30:]   — raw embedding (obs[44:428])

    transplant copies classifier encoder weights into the embedding
    columns only, leaving shape layout columns randomly initialized.

    layer 1 (Linear 384→hidden):
        classifier.encoder[0].weight  shape: (hidden, 384)
        right_encoder[0].weight       shape: (hidden, 414)
        copy classifier weights → right_encoder weight[:, 30:]

    layer 1 bias:
        copy directly (same shape: hidden)

    layer 2 (Linear hidden→hidden):
        identical shape — copy directly

    layer 2 bias:
        copy directly

    args:
        classifier: trained TaskClassifierNetwork (cpu, eval mode)
        bc_network: BicameralNetwork to transplant into.
                    if None, creates a fresh BicameralNetwork.
        verbose:    print transplant details

    returns:
        BicameralNetwork with transplanted right stream encoder weights
    """
    from bc_train import BicameralNetwork

    if bc_network is None:
        bc_network = BicameralNetwork()
        if verbose:
            print("[phase0] created fresh BicameralNetwork for transplant")

    if verbose:
        print("\n[phase0] transplanting classifier weights → right stream encoder")

    classifier_enc = classifier.encoder
    right_enc      = bc_network.right_encoder

    with torch.no_grad():

        # --- layer 0: Linear(EMBEDDING_DIM, hidden) → Linear(414, hidden) ---
        # classifier weight: (hidden, 384)
        # right_enc weight:  (hidden, 414)
        # copy into columns [30:] which correspond to the embedding portion
        c_weight_0 = classifier_enc[0].weight   # (hidden, 384)
        c_bias_0   = classifier_enc[0].bias      # (hidden,)

        r_weight_0 = right_enc[0].weight         # (hidden, 414)

        # verify shapes are compatible
        if c_weight_0.shape[0] != r_weight_0.shape[0]:
            raise ValueError(
                f"hidden size mismatch: classifier={c_weight_0.shape[0]} "
                f"vs right_encoder={r_weight_0.shape[0]}. "
                f"make sure --hidden matches POLICY_HIDDEN_SIZE={POLICY_HIDDEN_SIZE}"
            )
        if c_weight_0.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"classifier input dim mismatch: got {c_weight_0.shape[1]}, "
                f"expected {EMBEDDING_DIM}"
            )

        # transplant embedding columns only
        right_enc[0].weight[:, RIGHT_SHAPE_COLS:].copy_(c_weight_0)
        right_enc[0].bias.copy_(c_bias_0)

        if verbose:
            print(f"  layer 0 weights: classifier ({c_weight_0.shape}) → "
                  f"right_encoder cols [{RIGHT_SHAPE_COLS}:] of "
                  f"({r_weight_0.shape})  ✓")

        # --- layer 2: Linear(hidden, hidden) — identical shapes, copy directly ---
        c_weight_2 = classifier_enc[2].weight   # (hidden, hidden)
        c_bias_2   = classifier_enc[2].bias      # (hidden,)

        r_weight_2 = right_enc[2].weight         # (hidden, hidden)

        if c_weight_2.shape != r_weight_2.shape:
            raise ValueError(
                f"layer 2 shape mismatch: classifier={c_weight_2.shape} "
                f"vs right_encoder={r_weight_2.shape}"
            )

        right_enc[2].weight.copy_(c_weight_2)
        right_enc[2].bias.copy_(c_bias_2)

        if verbose:
            print(f"  layer 2 weights: classifier ({c_weight_2.shape}) → "
                  f"right_encoder ({r_weight_2.shape})  ✓")

    if verbose:
        print("  transplant complete.")

    return bc_network


# ---------------------------------------------------------------------------
# transplant verification
# ---------------------------------------------------------------------------

def verify_transplant(
    classifier: TaskClassifierNetwork,
    bc_network: "BicameralNetwork",
    n_probes:   int  = 100,
    verbose:    bool = True,
) -> bool:
    """
    verify that transplanted weights produce consistent encoder outputs.

    generates n_probes random prompts, runs each through:
        1. classifier.encoder(embedding_384)
        2. bc_network.right_encoder(zero_padded_to_414)[embedding portion]

    checks that outputs are numerically identical (within float32 tolerance).

    returns True if verification passes, False otherwise.
    """
    gen = PromptGenerator()
    classifier.eval()
    bc_network.eval()

    if verbose:
        print(f"\n[phase0] verifying transplant with {n_probes} probes...")

    max_diff  = 0.0
    n_failed  = 0

    with torch.no_grad():
        for _ in range(n_probes):
            # sample a random prompt and get its embedding
            prompt    = gen.sample()
            emb_384   = torch.tensor(
                get_embedding(prompt), dtype=torch.float32).unsqueeze(0)

            # run through classifier encoder
            cls_out   = classifier.encoder(emb_384)   # (1, hidden)

            # build a 414-dim input: zeros for shape cols, real embedding for rest
            inp_414   = torch.zeros(1, 414)
            inp_414[:, RIGHT_SHAPE_COLS:] = emb_384   # cols [30:] = embedding

            # run through right_encoder
            right_out = bc_network.right_encoder(inp_414)   # (1, hidden)

            diff = (cls_out - right_out).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-4:
                n_failed += 1

    passed = n_failed == 0

    if verbose:
        status = "✓ PASSED" if passed else f"✗ FAILED ({n_failed}/{n_probes} probes)"
        print(f"  max absolute difference : {max_diff:.2e}")
        print(f"  result                  : {status}\n")

    return passed


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save_classifier(
    classifier: TaskClassifierNetwork,
    save_path:  str,
    metrics:    dict = None,
):
    """save classifier weights and optional metrics to save_path/."""
    os.makedirs(save_path, exist_ok=True)
    weights_path = os.path.join(save_path, "prompt_classifier.pt")
    torch.save(classifier.state_dict(), weights_path)
    print(f"[phase0] classifier weights saved → {weights_path}")

    if metrics is not None:
        metrics_path = os.path.join(save_path, "prompt_classifier_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=3)
        print(f"[phase0] metrics saved → {metrics_path}")


def load_classifier(
    save_path: str,
    hidden:    int = POLICY_HIDDEN_SIZE,
) -> TaskClassifierNetwork:
    """load a previously saved TaskClassifierNetwork."""
    weights_path = os.path.join(save_path, "prompt_classifier.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"no classifier weights found at {weights_path}\n"
            f"run phase 0 training first: python prompt_train.py"
        )
    network = TaskClassifierNetwork(hidden=hidden, n_tasks=N_TASKS)
    network.load_state_dict(torch.load(weights_path, map_location="cpu"))
    network.eval()
    print(f"[phase0] classifier loaded from {weights_path}")
    return network


# ---------------------------------------------------------------------------
# run_phase0 — main entry point for pipeline integration
# ---------------------------------------------------------------------------

def run_phase0(
    save_path:       str   = "./models/phase0",
    n_samples:       int   = 10_000,
    epochs:          int   = 40,
    hidden:          int   = POLICY_HIDDEN_SIZE,
    device:          str   = None,
    verbose:         bool  = True,
    use_cache:       bool  = True,
    skip_if_exists:  bool  = True,
) -> "BicameralNetwork":
    """
    full phase 0 pipeline:
        1. build / load dataset
        2. train classifier (or load if already trained)
        3. evaluate per-task accuracy
        4. transplant weights into a fresh BicameralNetwork
        5. verify transplant
        6. save classifier weights

    args:
        save_path:      directory to save classifier weights and metrics
        n_samples:      dataset size
        epochs:         training epochs
        hidden:         hidden size — must equal POLICY_HIDDEN_SIZE
        device:         "cuda" / "cpu" — auto-detected if None
        verbose:        print progress
        use_cache:      cache generated dataset to disk to avoid re-embedding
        skip_if_exists: if classifier weights already exist in save_path,
                        skip training and load them directly

    returns:
        BicameralNetwork with right stream encoder pre-initialized from
        classifier weights. pass this directly to train_bc() or
        build_ppo_from_bc() in your training pipeline.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PHASE 0 — prompt classification pretraining")
        print(f"  save_path : {save_path}")
        print(f"  n_samples : {n_samples:,}")
        print(f"  epochs    : {epochs}")
        print(f"  hidden    : {hidden}  (must equal POLICY_HIDDEN_SIZE)")
        print(f"  device    : {device}")
        print(f"{'='*60}\n")

    os.makedirs(save_path, exist_ok=True)

    # --- step 1: build or load dataset ---
    cache_path = os.path.join(save_path, "dataset_cache.npz") if use_cache else None
    dataset    = build_dataset(
        n_samples=n_samples,
        verbose=verbose,
        cache_path=cache_path,
    )

    # --- step 2: train or load classifier ---
    weights_path = os.path.join(save_path, "prompt_classifier.pt")
    if skip_if_exists and os.path.exists(weights_path):
        if verbose:
            print(f"[phase0] classifier weights found at {weights_path} "
                  f"— skipping training (pass skip_if_exists=False to retrain)")
        classifier = load_classifier(save_path, hidden=hidden)
    else:
        classifier = train_classifier(
            dataset=dataset,
            hidden=hidden,
            epochs=epochs,
            device=device,
            verbose=verbose,
        )

    # --- step 3: evaluate ---
    metrics = evaluate_classifier(classifier, dataset, verbose=verbose)

    # warn if accuracy is too low to be useful for transplant
    if metrics["overall"] < 0.70:
        print(f"  ⚠  overall accuracy {metrics['overall']:.1%} is below 70% — "
              f"consider increasing --samples or --epochs before transplanting.")

    # --- step 4: save classifier ---
    save_classifier(classifier, save_path, metrics)

    # --- step 5: transplant into BicameralNetwork ---
    from bc_train import BicameralNetwork
    bc_network = transplant_into_bicameral(
        classifier=classifier,
        bc_network=BicameralNetwork(),
        verbose=verbose,
    )

    # --- step 6: verify ---
    verify_transplant(classifier, bc_network, verbose=verbose)

    # save the transplanted network weights too so they can be reloaded
    transplant_path = os.path.join(save_path, "transplanted_bc_network.pt")
    torch.save(bc_network.state_dict(), transplant_path)
    if verbose:
        print(f"[phase0] transplanted BicameralNetwork saved → {transplant_path}")
        print(f"\n[phase0] done. BicameralNetwork ready for BC/PPO training.\n")

    return bc_network


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="phase 0 — pretrain prompt classifier and transplant "
                    "weights into BicameralNetwork right stream encoder"
    )
    parser.add_argument(
        "--save", type=str, default="./models/phase0",
        help="directory to save classifier weights and metrics (default: ./models/phase0)",
    )
    parser.add_argument(
        "--samples", type=int, default=10_000,
        help="number of (prompt, label) pairs to generate (default: 10000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="classifier training epochs (default: 40)",
    )
    parser.add_argument(
        "--hidden", type=int, default=POLICY_HIDDEN_SIZE,
        help=f"hidden size — must equal POLICY_HIDDEN_SIZE={POLICY_HIDDEN_SIZE} "
             f"for transplant to work (default: {POLICY_HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="do not cache the generated dataset to disk",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="retrain even if classifier weights already exist",
    )
    parser.add_argument(
        "--no-transplant", action="store_true",
        help="train classifier only, skip transplant into BicameralNetwork",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="load existing classifier and verify transplant only (no training)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="load existing classifier and print per-task accuracy (no training)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- verify only ---
    if args.verify:
        from bc_train import BicameralNetwork
        classifier = load_classifier(args.save, hidden=args.hidden)
        bc_network = BicameralNetwork()
        transplant_into_bicameral(classifier, bc_network, verbose=True)
        verify_transplant(classifier, bc_network, verbose=True)

    # --- eval only ---
    elif args.eval_only:
        classifier = load_classifier(args.save, hidden=args.hidden)
        cache_path = (None if args.no_cache
                      else os.path.join(args.save, "dataset_cache.npz"))
        dataset    = build_dataset(n_samples=2_000, verbose=True,
                                   cache_path=cache_path)
        evaluate_classifier(classifier, dataset, verbose=True)

    # --- no transplant ---
    elif args.no_transplant:
        cache_path = (None if args.no_cache
                      else os.path.join(args.save, "dataset_cache.npz"))
        dataset    = build_dataset(
            n_samples=args.samples, verbose=True, cache_path=cache_path)
        classifier = train_classifier(
            dataset=dataset, hidden=args.hidden,
            epochs=args.epochs, device=device, verbose=True)
        metrics    = evaluate_classifier(classifier, dataset, verbose=True)
        save_classifier(classifier, args.save, metrics)

    # --- full pipeline ---
    else:
        run_phase0(
            save_path=args.save,
            n_samples=args.samples,
            epochs=args.epochs,
            hidden=args.hidden,
            device=device,
            use_cache=not args.no_cache,
            skip_if_exists=not args.retrain,
        )