import os
import gymnasium as gym
from gymnasium.utils.play import play

ENV_ID = "LunarLander-v3"
MODEL = "models/ppo_lunarlander_final"


mode = input("[1] Human\n[2] AI\n> ").strip()

if mode == "1":
    env = gym.make(ENV_ID, render_mode="rgb_array")

    play(
        env,
        keys_to_action={
            "w": 2,
            "a": 3,
            "d": 1
        },
        noop=0
    )

    env.close()

else:
    from stable_baselines3 import PPO

    path = input(f"Model path [{MODEL}]: ").strip() or MODEL

    if not (os.path.isfile(path) or os.path.isfile(path + ".zip")):
        print("Model not found.")
        quit()

    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(path)

    obs, _ = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        env.close()