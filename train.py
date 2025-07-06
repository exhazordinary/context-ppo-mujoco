import torch
import gymnasium as gym
import json
import os
import numpy as np
from envs.protocol_wrapper import ProtocolWrapper
from models.context_ppo import PPOAgent
from config.protocols import PROTOCOLS


def make_env(protocol_name):
    base_env = gym.make("HalfCheetah-v5")
    protocol = PROTOCOLS[protocol_name]
    wrapped_env = ProtocolWrapper(base_env, protocol)
    return wrapped_env


def train(protocol_names, total_episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 17  # for HalfCheetah-v5
    action_dim = 6
    context_dim = len(protocol_names)

    agent = PPOAgent(obs_dim, action_dim, context_dim, device)

    for protocol_idx, protocol_name in enumerate(protocol_names):
        env = make_env(protocol_name)
        context_vec = np.eye(context_dim)[protocol_idx]  # one-hot

        for ep in range(total_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.get_action(obs, context_vec)
                scaled_action = np.clip(action * env.action_space.high, env.action_space.low, env.action_space.high)
                next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
                done = terminated or truncated

                agent.store_transition(obs, action, reward, next_obs, done, context_vec)
                obs = next_obs
                total_reward += reward

            agent.learn()
            if ep % 10 == 0:
                print(f"Protocol: {protocol_name}, Episode {ep}, Reward: {total_reward:.2f}")

        env.close()

    torch.save(agent.model.state_dict(), "checkpoints/context_ppo.pth")
    print("✅ Model saved to checkpoints/context_ppo.pth")

if __name__ == "__main__":
    train(protocol_names=list(PROTOCOLS.keys()))
