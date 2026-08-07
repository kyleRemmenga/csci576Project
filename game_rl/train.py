#!/usr/bin/env python3
"""Train a masked conv policy on the shapez build env.

    PYTHONPATH=. .venv/bin/python train.py --timesteps 200000
"""

import argparse
import os
import sys

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from shapez_rl.encoding import DEFAULT_BUILDINGS, ROTATIONS
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.expert import HUB_TILE, ScriptedMiner
from shapez_rl.fake_api import FakeShapezServer
from shapez_rl.policy import SpatialMaskablePolicy
from shapez_rl.wrappers import FlatAction


def bounds_arg(text):
    try:
        x, y, w, h = (int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected four integers: x,y,w,h")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return {"x": x, "y": y, "w": w, "h": h}


def base_urls_arg(text):
    urls = tuple(url.strip().rstrip("/") for url in text.split(",") if url.strip())
    if not urls:
        raise argparse.ArgumentTypeError("expected at least one URL")
    if len(set(urls)) != len(urls):
        # Two envs sharing one game would interleave their episodes into one world.
        raise argparse.ArgumentTypeError("each env needs its own game; URLs must be distinct")
    return urls


def buildings_arg(text):
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one building")
    unknown = [name for name in names if name not in DEFAULT_BUILDINGS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown: {', '.join(unknown)}; valid: {', '.join(DEFAULT_BUILDINGS)}"
        )
    return names


def make_env(base_url, budget, ticks, target_shape, servers, bounds, buildings,
             log_path=None, goal_level=None):
    """Env factory. With no base_url, each env gets its own fake server."""

    def _init():
        url = base_url
        if url is None:
            server = FakeShapezServer().start()
            servers.append(server)
            url = server.base_url

        env = ShapezBuildEnv(
            base_url=url,
            bounds=bounds,
            buildings=buildings,
            placement_budget=budget,
            run_ticks=ticks,
            target_shape=target_shape,
            goal_level=goal_level,
        )
        # info_keywords lands the reward-ladder rungs in the CSV alongside reward.
        return Monitor(
            FlatAction(env),
            filename=log_path,
            info_keywords=("mined", "progress", "delivered", "waste_routed", "placements_succeeded"),
        )

    return _init


def build_model(venv, args):
    example = venv.get_attr("env")[0]  # unwrap Monitor -> FlatAction

    return MaskablePPO(
        SpatialMaskablePolicy,
        venv,
        policy_kwargs=dict(
            n_buildings=len(example.env.buildings),
            n_rotations=len(ROTATIONS),
            feature_channels=args.channels,
        ),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=args.target_kl,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.tensorboard,
        device=args.device,
    )


def collect_expert(env_fn, episodes, target_shape, seed_start=0):
    """Roll out the scripted expert, recording (observation, flat action) pairs.

    Builds its own env rather than borrowing one from the vec env, because
    SubprocVecEnv keeps its envs in other processes where the expert cannot
    reach the live map cache it needs.
    """
    env = env_fn()
    flat_env = env.env
    inner = flat_env.env

    observations, actions, returns, skipped = [], [], [], 0
    try:
        for index in range(episodes):
            obs, _info = env.reset(seed=seed_start + index)
            agent = ScriptedMiner(inner, target_shape=target_shape)
            if agent.reset(inner._map_cache) == 0:
                skipped += 1
                continue

            episode_rewards = []
            terminated = False
            while not terminated:
                flat = flat_env.flatten_action(agent.act())
                observations.append(obs)
                actions.append(flat)
                obs, reward, terminated, _truncated, _info = env.step(flat)
                episode_rewards.append(reward)

            # gamma is 1.0, so the target for each step is the reward still to come.
            total = 0.0
            to_go = []
            for reward in reversed(episode_rewards):
                total += reward
                to_go.append(total)
            returns.extend(reversed(to_go))
    finally:
        env.close()

    if skipped:
        print(f"  {skipped} episode(s) had no routable patch and were skipped")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(returns, dtype=np.float32),
    )


def behavior_clone(model, observations, actions, returns, epochs, batch_size=64):
    """Supervised pretraining so PPO starts from a policy that already delivers.

    Fits the critic alongside the actor: leaving the value head at its random
    initialisation makes PPO's first advantages meaningless, and those updates
    undo the imitated policy before the critic ever catches up.
    """
    policy = model.policy
    obs_t = th.as_tensor(observations, device=policy.device)
    act_t = th.as_tensor(actions, device=policy.device)
    ret_t = th.as_tensor(returns, device=policy.device)
    count = len(act_t)

    policy.set_training_mode(True)
    for epoch in range(epochs):
        order = th.randperm(count, device=policy.device)
        action_total, value_total = 0.0, 0.0
        for start in range(0, count, batch_size):
            batch = order[start : start + batch_size]
            distribution = policy.get_distribution(obs_t[batch])
            action_loss = -distribution.log_prob(act_t[batch]).mean()
            values = policy.predict_values(obs_t[batch]).flatten()
            value_loss = th.nn.functional.mse_loss(values, ret_t[batch])
            loss = action_loss + model.vf_coef * value_loss

            policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            policy.optimizer.step()
            action_total += action_loss.item() * len(batch)
            value_total += value_loss.item() * len(batch)

        accuracy = _clone_accuracy(policy, obs_t, act_t)
        print(
            f"  bc epoch {epoch + 1}/{epochs}  loss {action_total / count:.4f}"
            f"  value {value_total / count:.3f}  match {accuracy:.0%}"
        )


def _clone_accuracy(policy, obs_t, act_t, limit=2048):
    with th.no_grad():
        sample = obs_t[:limit]
        predicted = policy.get_distribution(sample).distribution.probs.argmax(dim=-1)
        return (predicted == act_t[: len(sample)]).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        type=base_urls_arg,
        default=None,
        help="Real API; comma-separate one URL per env. Omit to use the fake",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--budget", type=int, default=24, help="Buildings placed per episode")
    parser.add_argument("--ticks", type=int, default=3000, help="Ticks in the run phase")
    parser.add_argument("--target-shape", default=None, help="Score one shape, e.g. CuCuCuCu")
    parser.add_argument(
        "--goal-level",
        type=int,
        default=None,
        help="Hub level to start at; 2 unlocks the cutter, 3 the balancer",
    )
    parser.add_argument(
        "--bounds",
        type=bounds_arg,
        default=None,
        help="Play area as x,y,w,h; default -16,-16,32,32",
    )
    parser.add_argument(
        "--buildings",
        type=buildings_arg,
        default=DEFAULT_BUILDINGS,
        help="Comma-separated subset to allow, e.g. miner,belt",
    )
    parser.add_argument("--channels", type=int, default=64, help="Conv trunk width")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=None, help="Rollout per env; default 10 episodes")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: one episode")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--bc-episodes",
        type=int,
        default=0,
        help="Expert episodes to imitate before PPO; 0 disables the warm start",
    )
    parser.add_argument("--bc-epochs", type=int, default=10)
    # Stops an update once the policy has moved too far, which prevents collapse.
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard", default=None, help="Log dir; omit to disable")
    parser.add_argument("--save", default="runs/ppo_shapez", help="Checkpoint path, no extension")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Timesteps between snapshots; 0 to keep only the final save",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/logs",
        help="Per-episode CSV dir; empty string to disable",
    )
    args = parser.parse_args()

    if args.n_steps is None:
        args.n_steps = args.budget * 10
    if args.batch_size is None:
        args.batch_size = args.budget

    if args.base_url is None:
        print("no --base-url given, using the offline fake API\n")
    elif len(args.base_url) != args.n_envs:
        parser.error(
            f"got {len(args.base_url)} URL(s) but --n-envs {args.n_envs}; "
            "pass one running game per env"
        )

    if args.bounds is not None:
        hub_x, hub_y = HUB_TILE
        inside = (
            args.bounds["x"] <= hub_x < args.bounds["x"] + args.bounds["w"]
            and args.bounds["y"] <= hub_y < args.bounds["y"] + args.bounds["h"]
        )
        if not inside:
            # Without the hub in frame nothing can be delivered, so reward stays 0.
            print(f"warning: bounds exclude the hub at {HUB_TILE}; reward is unreachable\n")

    servers = []
    log_dir = args.log_dir or None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        print(f"per-episode CSV -> {os.path.join(log_dir, '<n>.monitor.csv')}\n")

    def env_fn(index, log_path):
        return make_env(
            args.base_url[index] if args.base_url else None,
            args.budget,
            args.ticks,
            args.target_shape,
            servers,
            args.bounds,
            args.buildings,
            log_path,
            args.goal_level,
        )

    env_fns = [
        env_fn(i, os.path.join(log_dir, str(i)) if log_dir else None)
        for i in range(args.n_envs)
    ]

    # DummyVecEnv steps envs one after another, so real games only overlap under
    # SubprocVecEnv. The fake API is fast enough that process overhead costs more.
    parallel = args.n_envs > 1 and args.base_url is not None
    if parallel:
        print(f"stepping {args.n_envs} games in parallel via SubprocVecEnv\n")
    venv = SubprocVecEnv(env_fns) if parallel else DummyVecEnv(env_fns)

    callbacks = []
    if args.checkpoint_freq > 0:
        # save_freq counts steps per env, so scale it to mean total timesteps.
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // args.n_envs),
                save_path=os.path.dirname(args.save) or ".",
                name_prefix=os.path.basename(args.save),
            )
        )

    try:
        model = build_model(venv, args)
        if args.bc_episodes > 0:
            print(f"collecting {args.bc_episodes} expert episodes")
            # No log path: this env's episodes are demonstrations, not training.
            observations, actions, returns = collect_expert(
                env_fn(0, None), args.bc_episodes, args.target_shape
            )
            print(f"behaviour cloning on {len(actions)} expert actions")
            behavior_clone(model, observations, actions, returns, args.bc_epochs)
            # Baseline for "did PPO add anything on top of imitation?"
            model.save(f"{args.save}_bc")
            print(f"saved {args.save}_bc.zip")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks or None,
            progress_bar=False,
        )
        model.save(args.save)
        print(f"\nsaved {args.save}.zip")
    finally:
        venv.close()
        for server in servers:
            server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
