# Gym-Robotics Demo — Physical AI from A to Z

Physical AI demos for 3 robotics tasks using Reinforcement Learning (SAC + HER)
on the MuJoCo simulator:

1. **FetchReach** — robot moves the gripper to a target (easy, solves 100% in ~2 min).
2. **FetchPush** — robot pushes an object to a target (intermediate, solvable after a few training runs).
3. **Shadow Hand with 92 touch sensors** — 24-DOF robot hand (hard, needs millions of steps).

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

## Demo 1 + 2 — FetchReach and FetchPush

The script `demos/fetch_slide_demo.py` bundles the 3 Fetch tasks
(Reach / Push / Slide) behind an interactive menu.

```bash
python demos/fetch_slide_demo.py
```

Menu options:

| Option | Description | Time |
|---|---|---|
| 1 | FetchReach — train + solve 100% | ~2 min |
| 2 | FetchSlide — hard task, needs many runs | ~5 min/run |
| 3 | Render FetchReach (view 3D) | Instant |
| 4 | Render FetchSlide (view 3D) | Instant |
| 5 | Compare Random vs Trained | ~2 min |
| 6 | FetchPush — intermediate, solvable | ~5 min |
| 7 | Render FetchPush (view 3D) | Instant |
| 8 | Reset model (delete all, start over) | Instant |

**Recommended start:**
- Option `1` → train FetchReach (~2 min).
- Option `3` → watch the robot move in 3D MuJoCo.
- Option `6` → try FetchPush (run 2-3 times to accumulate ~100k-150k steps).

Each run auto-saves the model + replay buffer in `saved_models/`. The next run
loads them and continues training.

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

- **No GUI window on macOS** → try `mjpython demos/fetch_slide_demo.py` instead of `python`.
- **Running on a headless server (no display)** → change `render_mode="human"` to `render_mode="rgb_array"` in the code.
- **Training is slow** → this is normal, ~175 steps/sec on a MacBook M-series. A GPU does not help much (MuJoCo is CPU-bound).
- **FetchSlide does not solve right away** → it needs ~300k-500k steps total (Option 2 about 6-10 times). It is a hard task.
- **Warning `Overriding environment in registry`** → ignore it, harmless.

---

## Directory structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── demos/
    ├── fetch_slide_demo.py                       # FetchReach + FetchPush + FetchSlide
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
