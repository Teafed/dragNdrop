# dragNdrop

A reinforcement learning agent that controls a cursor to manipulate shapes on
a 2D canvas, conditioned on natural-language prompts. Built as a proof of
concept for language-conditioned action.

The agent reads instructions like "drag the red square to the left side"
or "click on the yellow circle" through a sentence embedding, then produces
continuous cursor movement and grip actions to satisfy the goal.

## Quickstart

### Installation
Requires Python 3.10+.
1. Clone or download and unzip the repository
2. Create and activate a virtual environment:
```bash
   # windows
   python -m venv .venv
   .venv\Scripts\activate

   # mac/linux
   python -m venv .venv
   source .venv/bin/activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```

**Demo:**
```bash
python demo.py
python demo.py --prompt "drag the green circle to the bottom"
python demo.py --task drag
python demo.py --model ./path/to/model      # demo a different model
python demo.py --oracle                     # demo oracle
```

**Evaluation:**
```bash
python demo.py --headless --episodes 200
python demo.py --task reach --headless --episodes 500
```

**Training:**
```bash
# from scratch
python train.py
python train.py --timesteps 800000 --bc-episodes 600 --bc-epochs 30
# resume from checkpoint
python train.py --resume ./models/shape_agent/stage_04_checkpoint --start-stage 5 --no-oracle
# specify save directory
python train.py --save ./different/path
```

## System overview

The system has three pieces: an environment that simulates the canvas, a
policy network that maps observations and prompts to actions, and a training
pipeline that warm-starts via behavior cloning before refining with PPO on
a curriculum.

### Environment

`shape_env.py` implements a Gymnasium environment with canvas containing
one shape and a cursor. Each step, the agent produces a continuous
`(dx, dy, grip)` action; the cursor moves and grips the shape. Episodes
terminate on success or after a step budget. Tasks supported: reach,
touch, drag, plus several rudimentary "scaffolding" tasks (move-to-zone,
click-at, hold-at) used during early curriculum stages.

### Bicameral network

The architecture is a two-stream network with cross-attention:

- A **left stream** reads cursor-local information (cursor position,
  grabbed shape, nearest shape).
- A **right stream** reads scene-global information (all shapes plus a
  384-dim sentence embedding of the prompt).
- A **cross-attention** block lets the right (global) stream attend to
  the left (cursor-local) stream when forming a decision.
- Three output heads produce movement, grip, and a direction-bias for
  drag tasks. The drag-direction head is gated by the holding bit so
  it contributes nothing during reach and touch.

### Training

Training runs in two phases:

1. **Behavior cloning** on demonstrations from a hand-coded oracle that
   has access to ground-truth state. The oracle solves any task we have
   defined and produces 600 episodes worth of `(observation, action)`
   pairs. We train the bicameral network on these with a mixture of
   regression and supervised classification losses.

2. **PPO refinement** picks up the BC weights and continues training
   on environment reward through a 12-stage curriculum that progressively
   introduces task complexity: rudimentary navigation first, then
   single-task focus stages, then composite tasks.

## Repository layout

```
config.py            shared constants (canvas dims, observation layout, task list)
shape_env.py         Gymnasium environment
llm_goal_parser.py   prompt parser + sentence embedding
prompt_gen.py        template-based prompt generator
oracle.py            hard-coded expert policy + demonstration collection
bc_train.py          bicameral network architecture + behavior cloning trainer
curriculum.py        12-stage curriculum manager with weighted task sampling
train.py             top-level training entry point
callbacks.py         SB3 callbacks for evaluation, curriculum advancement, shutdown
demo.py              pygame demo: trained agent, oracle, or human control
debug.py             diagnostic harness (env sanity, oracle solve rates, BC loss)
```
