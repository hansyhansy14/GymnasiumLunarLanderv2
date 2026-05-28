import argparse
import os
import pygame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


ENV_ID          = "LunarLander-v3"
MODEL_DIR       = "models"
LOG_DIR         = "logs"
FINAL_MODEL     = os.path.join(MODEL_DIR, "ppo_lunarlander_final")
TOTAL_TIMESTEPS = 100_000
N_ENVS          = 16
CHECKPOINT_FREQ = 500_000

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
        gamma=0.999,
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
            save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_lunarlander",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, "best"),
            log_path=LOG_DIR,
            eval_freq=max(10_000 // N_ENVS, 1),
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


def play(load_path, fullscreen=False):
    env = make_env(render_mode="human")
    model = build_model(env, load_path)

    env.reset()
    env.render()

    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        sw, sh = screen.get_size()
        scale = min(sw / 600, sh / 400)
        new_w, new_h = int(600 * scale), int(400 * scale)
        pygame.display.set_mode((new_w, new_h))

    obs, _ = env.reset()
    ep_reward, ep_num = 0.0, 1

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                print(f"Episode {ep_num}  reward = {ep_reward:+.1f}")
                ep_reward = 0.0
                ep_num += 1
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def evaluate(load_path, n_episodes=20):
    env = make_env()
    model = build_model(env, load_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True)
    print(f"Mean reward over {n_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "play", "eval"], default="train")
    p.add_argument("--load", default=None, metavar="PATH")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--fullscreen", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train(load_path=args.load)
    elif args.mode == "play":
        if not args.load:
            print(f"--load PATH required. Try: py agent.py --mode play --load {FINAL_MODEL}")
            raise SystemExit(1)
        play(load_path=args.load, fullscreen=args.fullscreen)
    elif args.mode == "eval":
        if not args.load:
            print("--load PATH required for eval mode.")
            raise SystemExit(1)
        evaluate(load_path=args.load, n_episodes=args.episodes)


if __name__ == "__main__":
    main()