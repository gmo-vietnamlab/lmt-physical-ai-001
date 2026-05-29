#!/usr/bin/env python
"""
Example using the HandManipulateBlock Touch Sensors environment
from Gymnasium-Robotics (Farama Foundation).

This environment simulates the Shadow Dexterous Hand manipulating a block,
with 92 touch sensors distributed over the palm and finger phalanges.

Run: python demos/manipulate_block_touch_sensors_example.py

Reference:
    https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block_touch_sensors/
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Register the robotics environments into the gymnasium registry
gymnasium_robotics.register_robotics_envs()


def run_continuous_touch_sensor_example():
    """
    Example with continuous touch sensors returning force values.
    Each sensor returns a float >= 0 (total normal contact force).
    """
    print("=" * 60)
    print("Example: HandManipulateBlock with Continuous Touch Sensors")
    print("=" * 60)

    # Create the environment with continuous touch sensors
    # Use render_mode="human" to show the MuJoCo GUI
    # Switch to "rgb_array" if running headless (no display)
    env = gym.make(
        "HandManipulateBlock_ContinuousTouchSensors-v1",
        render_mode="human",
    )

    # Reset the environment
    observation, info = env.reset()

    print(f"\n--- Observation Space Info ---")
    print(f"Observation keys: {list(observation.keys())}")
    print(f"Observation shape: {observation['observation'].shape}")  # (153,)
    print(f"Desired goal shape: {observation['desired_goal'].shape}")
    print(f"Achieved goal shape: {observation['achieved_goal'].shape}")

    # The last 92 values in the observation are the touch sensor data
    touch_data = observation["observation"][-92:]
    print(f"\nTouch sensor data (92 sensors): shape={touch_data.shape}")
    print(f"  Min force: {touch_data.min():.4f}")
    print(f"  Max force: {touch_data.max():.4f}")
    print(f"  Mean force: {touch_data.mean():.4f}")

    print(f"\n--- Action Space Info ---")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")

    # Run a few steps with random actions
    print(f"\n--- Running 200 steps with random actions ---")
    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print info every 50 steps
        if (step + 1) % 50 == 0:
            touch_data = observation["observation"][-92:]
            active_sensors = np.sum(touch_data > 0.01)
            print(
                f"  Step {step + 1:3d} | "
                f"Reward: {reward:.3f} | "
                f"Active sensors: {active_sensors}/92 | "
                f"Success: {info.get('is_success', False)}"
            )

    print(f"\nTotal reward after 200 steps: {total_reward:.3f}")
    env.close()


def run_boolean_touch_sensor_example():
    """
    Example with boolean touch sensors.
    True if the sensor detects contact force, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Example: HandManipulateBlock with Boolean Touch Sensors")
    print("=" * 60)

    env = gym.make(
        "HandManipulateBlock_BooleanTouchSensors-v1",
        render_mode="human",
    )

    observation, info = env.reset()

    # Boolean touch sensor data (0 or 1)
    touch_data = observation["observation"][-92:]
    print(f"\nBoolean touch sensor data:")
    print(f"  Sensors detecting contact: {int(touch_data.sum())}/92")

    # Run one episode
    print(f"\n--- Running 1 episode ---")
    total_reward = 0
    step = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 100 == 0:
            touch_data = observation["observation"][-92:]
            contacts = int(touch_data.sum())
            print(
                f"  Step {step:4d} | "
                f"Reward: {reward:.3f} | "
                f"Contacts: {contacts}/92 | "
                f"Success: {info.get('is_success', False)}"
            )

    print(f"\nEpisode ended after {step} steps")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Success: {info.get('is_success', False)}")
    env.close()


def run_no_render_example():
    """
    Fast headless example (no render) - suitable for training.
    """
    print("\n" + "=" * 60)
    print("Example: Headless run (no render) - suitable for training")
    print("=" * 60)

    # Do not set render_mode for faster execution
    env = gym.make("HandManipulateBlock_ContinuousTouchSensors-v1")

    observation, info = env.reset()
    print(f"\nObservation shape: {observation['observation'].shape}")
    print(f"  - Robot state (61 dims): joint positions, velocities, etc.")
    print(f"  - Touch sensors (92 dims): force readings from 92 sensors")
    print(f"  - Total: 61 + 92 = 153 dims")

    # Breakdown of touch sensor structure
    print(f"\n--- Distribution of 92 sensors on the hand ---")
    print(f"  Proximal phalanx (4 fingers x 7 sensors) = 28 sensors")
    print(f"  Middle phalanx   (4 fingers x 5 sensors) = 20 sensors")
    print(f"  Fingertip        (4 fingers x 5 sensors) = 20 sensors")
    print(f"  Thumb            (3 phalanges x 5 sensors) = 15 sensors")
    print(f"  Palm             (1 x 9 sensors)         =  9 sensors")
    print(f"  {'-' * 38}")
    print(f"  Total                                     = 92 sensors")

    # Quick run of 1000 steps
    print(f"\n--- Running 1000 steps (headless) ---")
    total_reward = 0
    for step in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()

    print(f"  Completed 1000 steps")
    print(f"  Total reward: {total_reward:.3f}")
    env.close()


def run_rotate_variations():
    """
    Try the different rotation variants of the environment.
    """
    print("\n" + "=" * 60)
    print("ManipulateBlock Touch Sensors environment variants")
    print("=" * 60)

    variations = [
        ("HandManipulateBlock_ContinuousTouchSensors-v1", "Hold block (no rotation)"),
        ("HandManipulateBlockRotateZ_ContinuousTouchSensors-v1", "Rotate around Z axis"),
        ("HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1", "Rotate parallel"),
        ("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1", "Free XYZ rotation"),
        ("HandManipulateBlockFull_ContinuousTouchSensors-v1", "Full (position + rotation)"),
    ]

    for env_id, description in variations:
        print(f"\n  {description}:")
        print(f"    ID: {env_id}")
        try:
            env = gym.make(env_id)
            obs, _ = env.reset()
            print(f"    Observation shape: {obs['observation'].shape}")
            print(f"    Desired goal shape: {obs['desired_goal'].shape}")
            print(f"    Action shape: {env.action_space.shape}")
            env.close()
        except Exception as e:
            print(f"    Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Gymnasium-Robotics: HandManipulateBlock Touch Sensors")
    print("  Shadow Dexterous Hand with 92 touch sensors")
    print("=" * 60)

    print("\nChoose an example:")
    print("  1. Continuous Touch Sensors (render GUI)")
    print("  2. Boolean Touch Sensors (render GUI)")
    print("  3. Headless - no render (fast, suitable for training)")
    print("  4. View environment variants")
    print("  5. Run all")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        run_continuous_touch_sensor_example()
    elif choice == "2":
        run_boolean_touch_sensor_example()
    elif choice == "3":
        run_no_render_example()
    elif choice == "4":
        run_rotate_variations()
    elif choice == "5":
        run_no_render_example()
        run_rotate_variations()
        run_continuous_touch_sensor_example()
        run_boolean_touch_sensor_example()
    else:
        print("Invalid choice. Running headless example...")
        run_no_render_example()
