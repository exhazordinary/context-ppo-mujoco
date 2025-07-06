import torch
import gymnasium as gym
import numpy as np
from envs.protocol_wrapper import ProtocolWrapper
from models.context_ppo import PPOAgent
from config.protocols import PROTOCOLS


def evaluate(agent, env, context_vec, episodes=3):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(obs, context_vec)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 17
    action_dim = 6
    context_dim = len(PROTOCOLS)

    agent = PPOAgent(obs_dim, action_dim, context_dim, device)
    # Normally you would load a trained model here
    agent.model.load_state_dict(torch.load("checkpoints/context_ppo.pth", map_location=device))

    for idx, protocol_name in enumerate(PROTOCOLS):
        print(f"\nEvaluating protocol: {protocol_name}")
        env = gym.make("HalfCheetah-v5", render_mode="human")
        env = ProtocolWrapper(env, PROTOCOLS[protocol_name])
        context_vec = np.eye(context_dim)[idx]
        evaluate(agent, env, context_vec)
        env.close()


if __name__ == "__main__":
    main()