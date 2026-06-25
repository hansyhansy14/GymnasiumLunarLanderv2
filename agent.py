import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

ENV_ID          = "LunarLander-v3"
MODEL_DIR       = "models"
LOG_DIR         = "logs"
FINAL_MODEL     = os.path.join(MODEL_DIR, "ppo_lunarlander_final")
TOTAL_TIMESTEPS = 200_000
N_ENVS          = 8


def make_env(render_mode=None):
    return Monitor(gym.make(ENV_ID, render_mode=render_mode))


def build_model(env, load_path=None):
    if load_path:
        return PPO.load(load_path, env=env)

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )


def train(load_path=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    eval_env = make_env()

    model = build_model(train_env, load_path)

    callbacks = [
        CheckpointCallback(
            save_freq=10_000,
            save_path=MODEL_DIR,
            name_prefix="ppo_lunarlander",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, "best"),
            log_path=LOG_DIR,
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=load_path is None,
        tb_log_name="PPO",
    )

    model.save(FINAL_MODEL)
    train_env.close()
    eval_env.close()


def play(load_path):
    env = make_env(render_mode="human")
    model = build_model(env, load_path)

    obs, _ = env.reset()
    episode = 1
    reward_sum = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward

        if terminated or truncated:
            print(f"Episode {episode} reward {reward_sum:.1f}")
            reward_sum = 0.0
            episode += 1
            obs, _ = env.reset()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "play"], default="train")
    p.add_argument("--load", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train(args.load)

    elif args.mode == "play":
        play(args.load)


if __name__ == "__main__":
    main()