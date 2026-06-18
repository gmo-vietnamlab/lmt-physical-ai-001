#!/usr/bin/env python
"""
Demo: FetchReach with reinforcement learning (practical on CPU).

FetchReach - move the gripper to a target point. Solves 100% with RL (SAC)
in ~2 minutes - a clean example of reinforcement learning working on a CPU.

Menu:
  1. Train  - learn the policy from scratch (~2 min), save the model
  2. Render - load the saved model and watch it act in 3D
  3. Reset  - delete the saved model

Run: python demos/fetch_reach.py
"""

import os
import time as time_module
import warnings
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC

# Suppress "Overriding environment already in registry" warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*")

gymnasium_robotics.register_robotics_envs()

MODEL_DIR = "saved_models"
REACH_MODEL = os.path.join(MODEL_DIR, "fetch_reach_sac")


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


def train_fetch_reach():
    """
    FetchReach: robot moves the gripper to a target position.
    The simplest task, solves 100% with just 20k steps (~2 minutes).
    """
    print("=" * 60)
    print("FetchReach (easy - solves 100%)")
    print("=" * 60)
    print("Task: move the gripper to the target point (red)")
    print("Action: 4 dims (dx, dy, dz, gripper)")
    print("Observation: 10 dims (gripper position + velocity)")
    print()

    env = gym.make("FetchReachDense-v4")

    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        batch_size=256,
        learning_starts=500,
        verbose=0,
    )

    print("Training 20k steps...")
    t0 = time_module.time()
    model.learn(total_timesteps=20_000)
    elapsed = time_module.time() - t0
    print(f"Done in {elapsed:.0f}s")

    success_rate, avg_reward = evaluate(model, "FetchReachDense-v4", n_episodes=20)
    print(f"\nResult: success={success_rate:.0%}, avg_reward={avg_reward:.1f}")
    print("-> The agent learned to move the gripper to the target!")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(REACH_MODEL)
    print(f"\nModel saved: {REACH_MODEL}.zip")

    env.close()
    return model


def render_fetch_reach(n_episodes=5):
    """Load the saved FetchReach model and render it acting in 3D."""
    print("\n" + "=" * 60)
    print("Render: FetchReach (trained RL model)")
    print("=" * 60)

    if not os.path.exists(REACH_MODEL + ".zip"):
        print(f"No saved model at {REACH_MODEL}.zip")
        print("Run option 1 first to train FetchReach.")
        return

    model = SAC.load(REACH_MODEL)
    env = gym.make("FetchReachDense-v4", render_mode="human")
    print("Close the MuJoCo window to stop.\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            time_module.sleep(0.03)
        ok = info.get("is_success", 0.0)
        print(f"  Episode {ep+1}: {'OK' if ok else 'X'} reward={ep_reward:.1f}")

    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  FETCH REACH DEMO (RL)")
    print("=" * 60)

    print("\nChoose:")
    print("  1. Train FetchReach - solve 100% with RL (~2 min)")
    print("  2. Render trained model in 3D")
    print("  3. Reset (delete saved model)")

    choice = input("\nEnter (1-3): ").strip()

    if choice == "1":
        train_fetch_reach()
    elif choice == "2":
        render_fetch_reach()
    elif choice == "3":
        f = REACH_MODEL + ".zip"
        if os.path.exists(f):
            os.remove(f)
            print(f"  Deleted: {f}")
        print("Done! Reach model reset.")
    else:
        train_fetch_reach()
