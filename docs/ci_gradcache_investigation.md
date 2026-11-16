# CI investigation: the two failing GradCache tests

**Date:** 2026-11-16. **Outcome:** the failures were float32 rounding differences that depend on the CPU's kernels
and the PyTorch version, not a gradient error. The exactness tests now run in float64 with a tolerance about a
million times stricter, and CI installs a pinned environment. The trainer was not changed.

## The failure

The only GitHub Actions run (run 42760457003, 2026-11-10, commit `f3b2ff9`, Ubuntu 24.04, Python 3.11.18) failed 2
of 1,051 tests. CI installed whatever was newest that day, which was torch 2.15.0+cpu and numpy 2.4.6. The
development machine had torch 2.10.0 and numpy 1.26.4.

| Test | Compared | Observed | Old tolerance |
|---|---|---|---|
| `test_without_dropout_it_equals_the_plain_full_batch_loss` | GradCache loss (mini-batches of 2) vs one full-batch pass of 7 pairs | 0.6561450362 vs 0.6561452746 (Δ 2.4e-7, about 4 float32 ULPs) | abs 1e-7 |
| `test_gradients_are_scaled_and_unscaled_exactly_as_in_plain_training` | Scaled gradients (×4096), mini-batches of 3 vs full batch | one embedding-gradient element 6.4306 vs 6.4309, in a row whose largest entries are about 220 | rtol 1e-5, atol 1e-4 |

## What was ruled in and out

I measured GradCache against the plain full-batch computation for mini-batch sizes 1, 2, 3 and 7. Each case was
run in float32 and in float64, with the same tiny encoder the tests use, in six environments.

| Environment | PyTorch | Old GradCache tests |
|---|---|---|
| macOS arm64 (Apple M2), the development machine | 2.10.0 | all pass |
| macOS arm64 | 2.15.0 | the scaled-gradient test fails |
| Linux arm64 (Docker, native) | 2.10.0+cpu | a third test fails: the dropout replay with mini-batch size 1 |
| Linux arm64 (Docker, native) | 2.15.0+cpu | all pass |
| Linux x86_64 (Docker, emulated) | 2.10.0+cpu | the two CI tests fail |
| Linux x86_64 (GitHub runner) | 2.15.0+cpu | the two CI tests fail (the CI log) |

- **CUDA versus CPU: not involved.** Every failing run is CPU-only (GitHub runners have no GPU).
- **A gradient correctness problem: no.**
  - With a float64 model, GradCache and the plain full-batch computation agree to at most **4.3e-16** of the
    largest gradient in every environment and every mini-batch size, and their losses are bitwise identical.
  - In float32, GradCache's gradient differs from the float64 result by 3.9e-7 to 2.2e-6 (relative norm). Plain
    float32 training differs by 5.2e-7 to 2.0e-6: the same size.
  - With one mini-batch the size of the full batch, the float32 results are bitwise identical.
- **Floating-point rounding: yes.** GradCache encodes mini-batches of 1–3 rows and accumulates parameter gradients
  over several backward passes. The plain computation encodes all 7 rows at once in a single backward pass. The
  sums are mathematically equal but run in a different order, so in float32 they round differently: by 0 to about
  4 ULPs here.
- **PyTorch version: yes.** On the same Mac, torch 2.10.0 passes and torch 2.15.0 fails the scaled-gradient test.
- **Hardware: yes.** The same torch build (2.15.0+cpu, x86_64) gives a 2.4e-7 loss difference on the GitHub runner
  and 6.0e-8 under emulation, whose CPU features differ. Linux arm64 and macOS arm64 also differ.

The old tests compared individual float32 elements with tolerances of about 1e-7. That is one or two units in the
last place, so whether they passed depended on the machine's summation order.

## The fix

- **Exactness is tested in float64** (`tests/unit/test_gradcache.py`). GradCache must match the reference to within
  **1e-12 of the largest gradient magnitude**. The old float32 tolerances were about 1e-7 absolute or 1e-6 relative,
  so this is about a million times stricter. Rounding in float64 is about 1e-16, while a wrong term, scale or
  replay changes gradients by far more than 1e-12.
  - The dropout-replay, plain full-batch and GradScaler tests all run in float64.
  - The full-batch comparison now covers mini-batch sizes 1, 2, 3 and 7.
  - Power-of-two loss scaling is checked with exact equality.
- **A float32 accuracy test was added.** For mini-batch sizes 1–3, GradCache's float32 gradient must be as close to
  the float64 gradient as plain float32 training is: within 4 times the plain error, or 4 float32 epsilons if
  that is larger.
- **The new tests catch errors the old ones missed.** I multiplied GradCache's own loss by 1 + 2⁻²³ (one float32
  ULP, the smallest change float32 can represent) and left the reference untouched. Both new exactness tests fail.
  The old full-batch test passed with this error.
- **Results.** The new tests pass in all six environments above.
  - Unchanged and still passing: the BERT replay test (dropout and eager attention, float32), the
    `gradcache_check` preflight test, and the end-to-end training test.
  - The trainer (`biencoder_trainer.py`) and its preflight thresholds were not changed, so the recorded training
    runs remain the code's behaviour.
- **CI is pinned.** CI installs `requirements/lock-linux-x86_64-cpu.txt`: torch 2.10.0+cpu and every other package
  at the analysis environment's version ([environment.md](environment.md)). It no longer picks up whatever PyTorch
  is newest.

## Not yet verified

The new tests and the pinned CI environment have not run on a GitHub runner, because that needs a push. A local
replica of the whole workflow (Docker, Linux x86_64 emulated, a clean copy of the files that would be committed,
the pinned lock) passes: `pip check`, ruff, and 1,281 tests passed, 41 skipped, 0 failed. The first push will
show whether the GitHub runner agrees.
