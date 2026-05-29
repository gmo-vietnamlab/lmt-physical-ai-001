#!/usr/bin/env python
"""
RL training demo for FetchSlide - robot slides an object toward a target.

FetchSlide is harder than FetchReach because:
  - The robot must slide a puck across the table to the goal position
  - The goal lies OUTSIDE the robot's reach -> it must push hard enough for
    the puck to slide all the way there
  - It requires precise control of force and direction

Fetch task difficulty (easy -> hard):
  FetchReach (100% @ 20k steps) -> FetchPush -> FetchSlide -> FetchPickAndPlace

FetchSlide needs ~100k-500k steps before clear improvement shows up.
This script lets you train in batches and track progress.

Run: python demos/fetch_slide_demo.py
"""

import os
import time as time_module
import warnings
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

# Suppress "Overriding environment already in registry" warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*")

gymnasium_robotics.register_robotics_envs()

MODEL_DIR = "saved_models"
SLIDE_MODEL = os.path.join(MODEL_DIR, "fetch_slide_sac")
REACH_MODEL = os.path.join(MODEL_DIR, "fetch_reach_sac")


def evaluate(model, env_id, n_episodes=10):
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


# ============================================================
# Demo 1: FetchReach - quick solve to understand the flow
# ============================================================
def demo_fetch_reach():
    """
    FetchReach: robot moves the gripper to a target position.
    The simplest task, solves 100% with just 20k steps (~2 minutes).
    """
    print("=" * 60)
    print("DEMO 1: FetchReach (easy - solves 100%)")
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

    # Evaluate
    success_rate, avg_reward = evaluate(model, "FetchReachDense-v4", n_episodes=20)
    print(f"\nResult: success={success_rate:.0%}, avg_reward={avg_reward:.1f}")
    print("-> The agent learned to move the gripper to the target!")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(REACH_MODEL)
    print(f"\nModel saved: {REACH_MODEL}.zip")

    env.close()
    return model


# ============================================================
# Demo 2: FetchSlide - harder, needs longer training
# ============================================================
def demo_fetch_slide(timesteps=50_000):
    """
    FetchSlide: Robot pushes puck to slide toward a target position.
    Much harder than FetchReach, requires longer training.

    Uses SAC + HER (Hindsight Experience Replay) for efficient learning.

    Key fix: save/load replay buffer so continued training actually works.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: FetchSlide (hard - needs many steps)")
    print("=" * 60)
    print("Task: slide a puck across the table to the target position")
    print("Hard because: goal is out of reach, needs correct force + direction")
    print(f"Training {timesteps:,} steps (estimated ~{timesteps/175:.0f}s)...")
    print()

    env = gym.make("FetchSlide-v4")  # Sparse reward + HER

    buffer_path = SLIDE_MODEL + "_buffer"

    # Check for existing model AND buffer to continue training properly
    if os.path.exists(SLIDE_MODEL + ".zip"):
        print("Found existing model, continuing training...")
        model = SAC.load(SLIDE_MODEL, env=env)
        # Load replay buffer if available (critical for HER to work)
        if os.path.exists(buffer_path + ".pkl"):
            print("Loading existing replay buffer (keep collected experience)...")
            model.load_replay_buffer(buffer_path)
            print(f"  Buffer size: {model.replay_buffer.size()} transitions")
        else:
            print("  ! No existing buffer found, starting to collect again.")
    else:
        print("Creating new model...")
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=1e-3,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=1000,
            tau=0.005,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            # HER - critical for sparse reward
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            verbose=0,
        )

    t0 = time_module.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed = time_module.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({timesteps/elapsed:.0f} steps/s)")

    # Save model AND replay buffer
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(SLIDE_MODEL)
    model.save_replay_buffer(buffer_path)
    print(f"Model + Buffer saved (buffer: {model.replay_buffer.size()} transitions)")

    # Evaluate
    success_rate, avg_reward = evaluate(model, "FetchSlide-v4", n_episodes=10)
    print(f"\nResult: success={success_rate:.0%}, avg_reward={avg_reward:.1f}")

    if success_rate < 0.1:
        print("\nLow success rate is normal for FetchSlide!")
        print(f"   It needs ~300k-500k steps in total.")
        print(f"   Re-run option 2 multiple times (the buffer is preserved).")
        print(f"   Or try option 6: FetchPush (easier, solvable).")
    else:
        print(f"\nThe agent is learning! Success rate: {success_rate:.0%}")

    env.close()
    return model


# ============================================================
# Demo 6: FetchPush - intermediate, solvable
# ============================================================
PUSH_MODEL = os.path.join(MODEL_DIR, "fetch_push_sac")


def demo_fetch_push(timesteps=50_000):
    """
    FetchPush: Robot pushes object to target position (within reach).
    Easier than FetchSlide, can be solved with ~50k-100k steps.
    Good middle ground to demonstrate successful training.
    """
    print("\n" + "=" * 60)
    print("DEMO 6: FetchPush (intermediate - solvable!)")
    print("=" * 60)
    print("Task: push an object to the target position (within reach)")
    print("Easier than FetchSlide because the goal is within the robot's reach")
    print(f"Training {timesteps:,} steps...")
    print()

    env = gym.make("FetchPush-v4")  # Sparse reward
    buffer_path = PUSH_MODEL + "_buffer"

    if os.path.exists(PUSH_MODEL + ".zip"):
        print("Found existing model, continuing training...")
        model = SAC.load(PUSH_MODEL, env=env)
        if os.path.exists(buffer_path + ".pkl"):
            model.load_replay_buffer(buffer_path)
            print(f"  Buffer loaded: {model.replay_buffer.size()} transitions")
    else:
        print("Creating new model...")
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=1e-3,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=1000,
            tau=0.005,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            verbose=0,
        )

    t0 = time_module.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed = time_module.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({timesteps/elapsed:.0f} steps/s)")

    # Save model + buffer
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(PUSH_MODEL)
    model.save_replay_buffer(buffer_path)

    # Evaluate
    success_rate, avg_reward = evaluate(model, "FetchPush-v4", n_episodes=10)
    print(f"\nResult: success={success_rate:.0%}, avg_reward={avg_reward:.1f}")

    if success_rate > 0:
        print(f"\nThe agent learned! Success rate: {success_rate:.0%}")
    else:
        print(f"\n   Re-run option 6 once or twice more (total ~100k-150k steps).")

    env.close()
    return model


# ============================================================
# Render a trained agent
# ============================================================
def demo_render(env_id, model_path):
    """Render a trained agent."""
    if not os.path.exists(model_path + ".zip"):
        print(f"No model found at {model_path}.zip")
        print("Train it first!")
        return

    print(f"\nRender: {env_id}")
    print("Close the MuJoCo window to stop.\n")

    env = gym.make(env_id, render_mode="human")
    model = SAC.load(model_path, env=env)

    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            time_module.sleep(0.04)  # ~25 FPS, slowed down to observe
        success = "OK" if info.get("is_success", False) else "X"
        print(f"  Episode {ep+1}: {success} reward={ep_reward:.1f} ({steps} steps)")

    env.close()


# ============================================================
# Compare before/after training
# ============================================================
def demo_compare():
    """Compare random vs trained on FetchReach."""
    print("\n" + "=" * 60)
    print("COMPARE: Random vs Trained (FetchReach)")
    print("=" * 60)

    env = gym.make("FetchReachDense-v4")

    # Random
    print("\n[Random Policy]")
    random_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        r = 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            r += reward
            done = terminated or truncated
        random_rewards.append(r)
    print(f"  Avg reward: {np.mean(random_rewards):.2f}")

    # Trained
    print("\n[Training 20k steps...]")
    model = SAC("MultiInputPolicy", env, learning_starts=500, verbose=0)
    model.learn(total_timesteps=20_000)

    print("\n[Trained Policy]")
    trained_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        r = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            r += reward
            done = terminated or truncated
        trained_rewards.append(r)
    print(f"  Avg reward: {np.mean(trained_rewards):.2f}")

    print(f"\n  Improvement: {np.mean(trained_rewards) - np.mean(random_rewards):+.2f}")
    print(f"  (Closer to 0 is best, since reward = -distance)")
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  FETCH ROBOTICS DEMO")
    print("  FetchReach (easy) -> FetchPush (intermediate) -> FetchSlide (hard)")
    print("=" * 60)

    print("\nChoose:")
    print("  1. FetchReach - Train + solve 100% (~2 min)")
    print("  2. FetchSlide - Train 50k steps (~5 min, run multiple times)")
    print("  3. Render FetchReach (train first)")
    print("  4. Render FetchSlide (train first)")
    print("  5. Compare Random vs Trained")
    print("  6. FetchPush - Intermediate, solvable (~5 min)")
    print("  7. Render FetchPush (train first)")
    print("  8. Reset model (delete old models, start over)")

    choice = input("\nEnter (1-8): ").strip()

    if choice == "1":
        demo_fetch_reach()
    elif choice == "2":
        demo_fetch_slide(timesteps=50_000)
    elif choice == "3":
        demo_render("FetchReachDense-v4", REACH_MODEL)
    elif choice == "4":
        demo_render("FetchSlide-v4", SLIDE_MODEL)
    elif choice == "5":
        demo_compare()
    elif choice == "6":
        demo_fetch_push(timesteps=50_000)
    elif choice == "7":
        demo_render("FetchPush-v4", PUSH_MODEL)
    elif choice == "8":
        # Reset all models
        for f in [SLIDE_MODEL + ".zip", SLIDE_MODEL + "_buffer.pkl",
                  PUSH_MODEL + ".zip", PUSH_MODEL + "_buffer.pkl",
                  REACH_MODEL + ".zip"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Deleted: {f}")
        print("Done! All models reset.")
    else:
        demo_fetch_reach()
