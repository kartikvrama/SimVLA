# VLA Interview Practice Notebooks

This folder contains a progressive set of notebooks for robotics interview prep
focused on VLA-style modeling, attention coding, diffusion/flow loops, and
training loop engineering.

## Recommended Order

1. `00_setup_tensor_drills.ipynb`
2. `01_attention_from_scratch.ipynb`
3. `02_transformer_block_conditioning.ipynb`
4. `03_reverse_loops_flow_matching.ipynb`
5. `04_vla_processing_and_collation.ipynb`
6. `05_action_spaces_and_norm.ipynb`
7. `06_training_loop_engineering.ipynb`
8. `07_mini_vla_end_to_end.ipynb`

## How To Use

- Each notebook has:
  - short concept recap,
  - provided helper functions,
  - TODO cells with `raise NotImplementedError`,
  - tests that should pass when TODOs are complete.
- Keep `common.py` in the same folder to reuse deterministic helpers.
- Use CPU and small synthetic data only; all exercises are designed to run fast.

## Interview Practice Tips

- Time-box each notebook (20-40 minutes, final notebook 45-60 minutes).
- Explain your implementation out loud while coding.
- After passing tests, discuss complexity, failure modes, and extensions.
