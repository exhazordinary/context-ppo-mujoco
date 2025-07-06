# Contextual PPO for Multi-Protocol HalfCheetah

This project implements a Contextual Proximal Policy Optimization (PPO) agent for the HalfCheetah-v5 environment using multiple protocols. The agent is trained to adapt to different task protocols by leveraging context vectors, enabling robust performance across a variety of environment modifications.

## Features

- **Contextual PPO**: Trains a single agent to handle multiple protocols using context vectors.
- **Custom Protocol Wrappers**: Easily extend or modify environment dynamics via protocol wrappers.
- **Training and Evaluation Scripts**: Simple CLI scripts for training and evaluating the agent.
- **Checkpointing**: Model weights are saved for later evaluation or resuming training.

## Installation

1. **Python Version**: Python 3.8 or higher is recommended.
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the Contextual PPO agent across all defined protocols:
```bash
python train.py
```
- Model checkpoints will be saved to `checkpoints/context_ppo.pth`.

### Evaluation

To evaluate the trained agent on all protocols:
```bash
python evaluate.py
```
- The script loads the checkpoint and prints episode rewards for each protocol.

## File Structure

- `train.py` — Main training script.
- `evaluate.py` — Evaluation script for trained agents.
- `config/` — Protocol definitions and configuration.
- `envs/` — Custom environment wrappers.
- `models/` — PPO agent implementation.
- `utils/` — Logging and utility functions.
- `assets/` — (Optional) Generated assets such as GIFs.
- `checkpoints/` — Saved model weights.

## Notes

- The project uses the `HalfCheetah-v5` environment from `gymnasium[mujoco]`. Ensure you have the necessary MuJoCo dependencies installed (handled by `gymnasium[mujoco]`).
- For custom protocols, edit or extend `config/protocols.py`.

## License

[Add your license information here, if applicable.]
