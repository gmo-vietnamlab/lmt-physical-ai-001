#!/usr/bin/env python
"""
Demo: FetchPush - two ways to solve the same task.

FetchPush - push an object across the table to a target position. This is
harder than FetchReach because the gripper must interact with an object,
not just move through free space. Two approaches are offered:

  - Scripted controller (no training): a hand-coded policy that lines up
    behind the object and pushes it toward the goal. ~70% success, runs
    instantly - good for a quick live demo.
  - RL + HER (training): SAC with a Hindsight Experience Replay buffer on
    the sparse-reward FetchPush-v4. HER relabels failed goals into
    successes, which is what makes the sparse task learnable. Success
    improves gradually over ~200-300k steps - re-run training a few times
    to accumulate steps. Training is CPU-bound and slow; the scripted
    option exists so the task can still be demoed instantly.

Menu:
  1. Train RL + HER  - train 50k steps, save model + replay buffer
  2. Render trained RL model in 3D
  3. Scripted controller - run, no training (~70%)
  4. Render scripted controller in 3D
  5. Reset - delete the saved push model + replay buffer

Run: python demos/fetch_push.py
"""

import os
import time as time_module
import warnings
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# Suppress "Overriding environment already in registry" warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*")

gymnasium_robotics.register_robotics_envs()

MODEL_DIR = "saved_models"
PUSH_MODEL = os.path.join(MODEL_DIR, "fetch_push_sac")
PUSH_BUFFER = os.path.join(MODEL_DIR, "fetch_push_sac_buffer")


# ============================================================
# Scripted controller (no training, ~70% success)
# ============================================================
def fetch_push_action(obs):
    """Hand-coded push policy (no learning, ~70% success).

    Strategy: line the gripper up behind the object on the object->goal line,
    drop to table height, then push the object toward the goal.
    obs["observation"] layout: [0:3] gripper, [3:6] object.
    """
    o = obs["observation"]
    grip, obj, goal = o[0:3], o[3:6], obs["desired_goal"]
    push = goal[:2] - obj[:2]
    dist = np.linalg.norm(push)
    u = push / dist if dist > 1e-6 else np.array([1.0, 0.0])  # unit push dir
    perp = np.array([-u[1], u[0]])
    rel = grip[:2] - obj[:2]
    behind = np.dot(rel, u) < 0           # gripper is on the far side from goal
    lateral = abs(np.dot(rel, perp))      # off-line distance
    contact = obj[:2] - u * 0.065         # aim point just behind the object
    at_table = grip[2] < obj[2] + 0.02

    if behind and lateral < 0.025 and at_table:
        target, gain = np.array([goal[0], goal[1], obj[2]]), 2.0       # push
    elif np.linalg.norm(grip[:2] - contact) < 0.025 and not at_table:
        target, gain = np.array([contact[0], contact[1], obj[2]]), 8.0  # descend
    else:
        target, gain = np.array([contact[0], contact[1], obj[2] + 0.07]), 8.0  # go above contact

    a = (target - grip) * gain
    return np.clip([a[0], a[1], a[2], -1.0], -1.0, 1.0)


def run_push_scripted(render=False):
    """Run the scripted FetchPush controller; optionally render in 3D."""
    print("\n" + "=" * 60)
    print("FetchPush (scripted controller - no training)")
    print("=" * 60)
    print("Hand-coded policy: line up behind the object, push toward the goal")
    print()

    n_episodes = 5 if render else 20
    env = gym.make("FetchPush-v4", render_mode="human" if render else None)
    if render:
        print("Close the MuJoCo window to stop.\n")

    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(fetch_push_action(obs))
            done = terminated or truncated
            if render:
                time_module.sleep(0.03)
        ok = info.get("is_success", 0.0)
        successes += ok
        if render:
            print(f"  Episode {ep+1}: {'OK' if ok else 'X'}")

    print(f"\nSuccess: {successes:.0f}/{n_episodes} = {successes/n_episodes:.0%}")
    env.close()


# ============================================================
# RL + HER training (sparse reward FetchPush-v4)
# ============================================================
def evaluate(model, env_id, n_episodes=20):
    """Evaluate a model, return success rate and average reward."""
    env = gym.make(env_id)
    successes = []
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        successes.append(info.get("is_success", 0.0))
        rewards.append(ep_reward)
    env.close()
    return np.mean(successes), np.mean(rewards)


def train_push_her(steps=50_000):
    """Train SAC + HER on sparse FetchPush-v4. Resumes from a saved model.

    Re-run this a few times to accumulate ~200-300k steps; success rate is
    typically 0% for the first one or two runs (HER needs accumulated data)
    and then climbs.
    """
    print("=" * 60)
    print("FetchPush (RL + HER, sparse reward)")
    print("=" * 60)
    print("Sparse reward + HER: failed goals get relabeled into successes")
    print()

    env = gym.make("FetchPush-v4")

    if os.path.exists(PUSH_MODEL + ".zip"):
        print("Loading existing model to continue training...")
        model = SAC.load(PUSH_MODEL, env=env)
        if os.path.exists(PUSH_BUFFER + ".pkl"):
            model.load_replay_buffer(PUSH_BUFFER)
            print(f"  Replay buffer restored ({model.replay_buffer.size()} transitions)")
    else:
        print("Creating new model...")
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            learning_rate=1e-3,
            batch_size=512,
            gamma=0.95,
            tau=0.05,
            learning_starts=1000,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            verbose=0,
        )

    print(f"Training {steps:,} steps...")
    t0 = time_module.time()
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    elapsed = time_module.time() - t0
    print(f"Training done in {elapsed:.0f}s ({steps / elapsed:.0f} steps/s)")

    success_rate, avg_reward = evaluate(model, "FetchPush-v4", n_episodes=20)
    print(f"\nResult: success={success_rate:.0%}, avg_reward={avg_reward:.1f}")
    if success_rate < 0.2:
        print("   Re-run training a few more times (total ~200-300k steps).")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(PUSH_MODEL)
    model.save_replay_buffer(PUSH_BUFFER)
    print(f"\nModel saved: {PUSH_MODEL}.zip (+ replay buffer)")

    env.close()
    return model


def render_push_trained(n_episodes=5):
    """Load the saved RL push model and render it acting in 3D."""
    print("\n" + "=" * 60)
    print("Render: FetchPush (trained RL model)")
    print("=" * 60)

    if not os.path.exists(PUSH_MODEL + ".zip"):
        print(f"No saved model at {PUSH_MODEL}.zip")
        print("Run option 1 first to train FetchPush with RL + HER.")
        return

    model = SAC.load(PUSH_MODEL)
    env = gym.make("FetchPush-v4", render_mode="human")
    print("Close the MuJoCo window to stop.\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            time_module.sleep(0.03)
        ok = info.get("is_success", 0.0)
        print(f"  Episode {ep+1}: {'OK' if ok else 'X'}")

    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  FETCH PUSH DEMO")
    print("  Scripted (instant) or RL + HER (training)")
    print("=" * 60)

    print("\nChoose:")
    print("  1. Train FetchPush - RL + HER, 50k steps (re-run to accumulate)")
    print("  2. Render trained RL model in 3D")
    print("  3. Scripted controller - run, no training (~70%)")
    print("  4. Render scripted controller in 3D")
    print("  5. Reset (delete saved push model + replay buffer)")

    choice = input("\nEnter (1-5): ").strip()

    if choice == "1":
        train_push_her()
    elif choice == "2":
        render_push_trained()
    elif choice == "3":
        run_push_scripted(render=False)
    elif choice == "4":
        run_push_scripted(render=True)
    elif choice == "5":
        for f in [PUSH_MODEL + ".zip", PUSH_BUFFER + ".pkl"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Deleted: {f}")
        print("Done! Push model reset.")
    else:
        run_push_scripted(render=False)
