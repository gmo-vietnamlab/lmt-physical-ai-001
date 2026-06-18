# Gym-Robotics Demo — Physical AI from A to Z

Physical AI demos for 3 robotics tasks on the MuJoCo simulator:

1. **FetchReach** — robot moves the gripper to a target. Reinforcement Learning
   (SAC), easy, solves 100% in ~2 min.
2. **FetchPush** — robot pushes an object to a target. Two approaches: a scripted
   controller (~70%, no training, instant) or RL + HER (training, intermediate).
3. **Shadow Hand with 92 touch sensors** — 24-DOF robot hand (hard, needs millions
   of steps; the script only demos the environment with a random policy).

---

## Requirements

| Item | Requirement |
|---|---|
| Python | 3.10+ |
| OS | macOS / Linux (Windows via WSL2) |
| Disk | ~2GB |
| GPU | Optional (MuJoCo is CPU-bound) |

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/gmo-vietnamlab/lmt-physical-ai-001.git
cd lmt-physical-ai-001

# 2. Create a virtual environment
python -m venv rl-env
source rl-env/bin/activate          # macOS / Linux
# rl-env\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Demo 1 — FetchReach (RL)

`demos/fetch_reach.py` trains a SAC agent that moves the gripper to a target.

```bash
python demos/fetch_reach.py
```

| Option | Description | Time |
|---|---|---|
| 1 | Train FetchReach — solve 100% (RL) | ~2 min |
| 2 | Render trained model (view 3D) | Instant |
| 3 | Reset (delete saved model) | Instant |

**Recommended start:** Option `1` → train (~2 min), then Option `2` → watch the
robot move in 3D MuJoCo. The trained model is saved in `saved_models/`.

---

## Demo 2 — FetchPush (scripted or RL + HER)

`demos/fetch_push.py` offers two ways to solve the push task: a hand-coded
scripted controller (instant) or RL + HER training.

```bash
python demos/fetch_push.py
```

| Option | Description | Time |
|---|---|---|
| 1 | Train FetchPush — RL + HER, 50k steps (re-run to accumulate) | ~5 min/run |
| 2 | Render trained RL model (view 3D) | Instant |
| 3 | Scripted controller — run, no training (~70%) | Instant |
| 4 | Render scripted controller (view 3D) | Instant |
| 5 | Reset (delete saved push model + replay buffer) | Instant |

**Quick demo:** Option `3` runs the scripted controller instantly (~70% success).
For the RL route, Option `1` trains with HER (sparse reward); success is typically
0% for the first one or two runs and climbs after ~200-300k accumulated steps.
Each training run auto-saves the model + replay buffer in `saved_models/` and the
next run continues from there.

---

## Demo 3 — Shadow Hand with Touch Sensors

```bash
python demos/manipulate_block_touch_sensors_example.py
```

The menu lets you choose:
- **Continuous Touch Sensors** — sensors return normal force values (float >= 0).
- **Boolean Touch Sensors** — sensors return 0/1 (contact / no contact).

Observation: **153 dims** = 61 robot state + 92 touch sensors.
Action: **20 dims** (20 motor commands for 20 actuated joints).

> ⚠️ This task needs **millions of steps** to solve — the script only demos the
> environment + a random policy.

---

## Standard RL project flow

```
Train (sim) → Save (file) → Load (anytime) → Inference (control the robot)
```

In production: train once (slow), use many times (~0.5ms/inference).

---

## Troubleshooting

- **No GUI window on macOS** → try `mjpython demos/<script>.py` instead of `python`.
- **Running on a headless server (no display)** → change `render_mode="human"` to `render_mode="rgb_array"` in the code.
- **Training is slow** → this is normal, ~175 steps/sec on a MacBook M-series. A GPU does not help much (MuJoCo is CPU-bound).
- **FetchPush RL success is 0% at first** → normal: HER needs accumulated data. Run training (Option 1) a few times (~200-300k steps total), or use the scripted controller (Option 3) for an instant demo.
- **Warning `Overriding environment in registry`** → ignore it, harmless.

---

## Directory structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── demos/
    ├── fetch_reach.py                            # FetchReach (RL: train + render)
    ├── fetch_push.py                             # FetchPush (scripted + RL/HER)
    └── manipulate_block_touch_sensors_example.py # Shadow Hand 92 touch sensors
```

`saved_models/` is created automatically during training (ignored by git).

---

## References

- [Gymnasium-Robotics (Farama Foundation)](https://robotics.farama.org/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Soft Actor-Critic paper](https://arxiv.org/abs/1801.01290)
- [OpenAI - Solving Rubik's Cube with a Robot Hand](https://openai.com/index/solving-rubiks-cube/)

---

## License

MIT (or a license of your choice — add a `LICENSE` file before going public).
