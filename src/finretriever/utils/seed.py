"""Determinism helpers.

Seeds Python's ``random``, NumPy, and PyTorch so that training and evaluation
runs are reproducible from the same config.
"""

# TODO(v1): `set_seed(seed: int)` covering random, numpy, torch (CPU + CUDA),
#           and PYTHONHASHSEED via os.environ.
# TODO(v1): note: cuBLAS determinism flags should be opt-in — they slow
#           training on GPU and the v1 fine-tune doesn't need them.
