# Congestion-Aware GFlowNet Sampling — Web Demo

Interactive 2D HyperGrid demo for the paper
*Congestion-Aware Inference-Time Sampling for Frozen GFlowNets*.

## Run

```bash
# Any static file server works:
python -m http.server --directory webdemo 8000
# then open http://localhost:8000
```

Or just `open webdemo/index.html` directly (file:// works for this demo).

## What it shows

Two panels side-by-side, both sampling trajectories on a 10×10 grid with
4 modes (star markers):

- **Left**: standard sampling — policy forward only.
- **Right**: capacity-relative (CapRel) sampling — same policy but with
  the congestion penalty
  φ(c, n·p) = λ · max(0, (c / (n·p_base))^β − t^β)
  subtracted from child-action logits at each step.

Heatmaps show visit counts per cell; yellow stars fill as modes are discovered.
Recent trajectories are traced in pink.

## Controls

| Control | Meaning |
|---|---|
| λ    | congestion penalty weight |
| t    | capacity threshold (penalty fires when visit ratio > t) |
| β    | BPR-style exponent on the visit ratio |
| Batch size | trajectories drawn per "Step" / "Play" tick |
| Base-policy temperature | softness of the pretend-trained policy |

Compare `modes found` between the two panels as you let it run.
At default settings (λ=5, t=1, β=2), CapRel fills all 4 modes
meaningfully faster than standard.

## Implementation notes

- **Single file** (`index.html`, ~350 LoC): no build step, no framework.
- The "pretrained" policy is a soft-greedy toward the nearest mode (a
  heuristic stand-in for a trained TB-GFN). The congestion mechanism
  itself is identical to the paper's implementation.
- Per-parent child visit counts are maintained online; the capacity
  baseline is `n·p_base` where `n` = parent visits and `p_base` = base-policy
  probability of that child.
- MDP: actions = {up, right, stop}; trajectory ends on stop or at boundary.

## Limitations

- Proxy policy, not a real trained GFN. If you want real weights, export
  a small TB-GFN to JSON and replace `basePolicy()`.
- 2D only. Extending to 3D/4D is straightforward but visualization becomes
  less readable (slice views needed).
