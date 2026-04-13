"""Train a TB-GFN on a 2D HyperGrid with asymmetric corner rewards.

Designed to produce a checkpoint that visualizes well in the web demo:
- 16x16 grid (small enough for fast training, large enough to see structure)
- One dominant mode (big reward) + several smaller modes
- Standard sampling collapses to the big mode; CapRel should spread across all.

Outputs:
    webdemo/trained_policy.json — state_dim/n_actions/hidden weights for JS
    webdemo/reward_map.json     — per-state reward for visualization
"""
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environments.hypergrid import HyperGrid
from src.algorithms.tb_gfn import TrajectoryBalanceGFN

H = 16
NDIM = 2
# Asymmetric reward map: one dominant mode + 4 small modes
MODES = {
    (H - 1, H - 1): 10.0,   # giant mode (top-right corner)
    (0, H - 1):     1.5,    # small mode (top-left corner)
    (H - 1, 0):     1.5,    # small mode (bottom-right corner)
    (5,  H - 1):    1.0,    # small mode (top edge)
    (H - 1, 5):     1.0,    # small mode (right edge)
}
R0 = 1e-3


def build_env():
    env = HyperGrid(ndim=NDIM, height=H, R0=R0, reward_cos=True)

    # Override reward with our asymmetric map.
    def custom_reward(state):
        return float(MODES.get(tuple(state), R0))

    env.get_reward = custom_reward
    return env


def train(env, n_iter=4000, lr=1e-3, hidden_dim=128):
    torch.manual_seed(42); np.random.seed(42)
    gfn = TrajectoryBalanceGFN(
        env, hidden_dim=hidden_dim, lr=lr, batch_size=64, epsilon=0.15,
    )
    print(f"Training TB-GFN: H={H}, NDIM={NDIM}, n_iter={n_iter}")
    print(f"Modes: {MODES}")

    log_every = 200
    losses = []
    for step in range(n_iter):
        loss = gfn.train_step()
        losses.append(loss)
        if (step + 1) % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"  step {step+1}/{n_iter}: loss = {avg:.4f}", flush=True)
    print(f"Final log_Z: {gfn.log_Z.item():.3f}")
    return gfn


def export_weights(gfn, out_path):
    """Serialize forward_policy weights to JSON for browser inference."""
    state = {
        'state_dim': gfn.state_dim,
        'n_actions': gfn.n_actions,
        'height': gfn.env.height,
        'ndim': gfn.env.ndim,
        'log_Z': float(gfn.log_Z.item()),
        'layers': [],
    }
    # PolicyNetwork has self.net = Sequential(Linear, LeakyReLU, Linear, LeakyReLU, Linear)
    for module in gfn.forward_policy.net:
        if isinstance(module, torch.nn.Linear):
            state['layers'].append({
                'type': 'linear',
                'W': module.weight.detach().cpu().numpy().tolist(),  # [out, in]
                'b': module.bias.detach().cpu().numpy().tolist(),    # [out]
            })
        elif isinstance(module, torch.nn.LeakyReLU):
            state['layers'].append({'type': 'leaky_relu',
                                    'slope': float(module.negative_slope)})
    with open(out_path, 'w') as f:
        json.dump(state, f)
    n_params = sum(
        len(L.get('b', [])) + sum(len(r) for r in L.get('W', []))
        for L in state['layers']
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path}  ({n_params} params, {size_kb:.1f} KB)")


def export_reward_map(out_path):
    """Per-state reward grid for visualization."""
    grid = []
    for x in range(H):
        row = []
        for y in range(H):
            row.append(MODES.get((x, y), R0))
        grid.append(row)
    payload = {
        'H': H, 'NDIM': NDIM,
        'modes': [{'x': k[0], 'y': k[1], 'r': v} for k, v in MODES.items()],
        'reward_grid': grid,   # [H][H], indexed [x][y]
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    env = build_env()
    gfn = train(env, n_iter=4000)
    out_dir = Path(__file__).resolve().parent
    export_weights(gfn, out_dir / 'trained_policy.json')
    export_reward_map(out_dir / 'reward_map.json')
    print("DONE.")
