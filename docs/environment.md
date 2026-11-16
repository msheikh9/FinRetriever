# Environment and installation

FinRetriever is tested on **Python 3.11** only. The published results were produced on macOS arm64 (Apple M2,
CPU) with Python 3.11.5. CI runs on Linux x86_64 with CPU-only PyTorch. Model training ran separately on a Kaggle
T4 GPU, in the Kaggle image's own CUDA PyTorch ([kaggle_training.md](kaggle_training.md)).

## What to install

| File | Contents | Use it for |
|---|---|---|
| `requirements/lock-macos-arm64.txt` | Every package, direct and transitive (63), at the version that produced the published results | Reproducing the results on a Mac |
| `requirements/lock-linux-x86_64-cpu.txt` | The same 63 packages at the same versions, with `torch==2.10.0+cpu` | Linux without a GPU; this is what CI installs |
| `requirements.txt` | The 13 direct runtime dependencies, pinned to the same versions | A shorter install that lets pip choose the transitive versions |
| `requirements-dev.txt` | `requirements.txt` plus pytest, pytest-cov, ruff, black and mypy, pinned | Development |
| `requirements-ci.txt` | `requirements-dev.txt` with CPU PyTorch; the readable source of the Linux lock | Regenerating the Linux lock |
| `requirements-kaggle.txt` | What the Kaggle GPU notebook added to its image | Historical record of the training runs; leave unchanged |
| `pyproject.toml` | Compatible version ranges: the core dependencies, plus the `dense` and `dev` extras | `pip install -e ".[dense,dev]"` when exact pins are not wanted |

Both lock files were generated with `uv pip compile` and constrained to the analysis environment's installed
versions. Every version in them is one that was actually used. The analysis environment itself is not published,
because it also contains unrelated packages.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install-locked            # picks requirements/lock-macos-arm64.txt on macOS, the Linux CPU lock elsewhere
# equivalent: pip install -r requirements/lock-<platform>.txt && pip install -e . --no-deps
make test                      # the full suite; no data or model download needed
```

- **Linux with an NVIDIA GPU.** Install the CUDA build of torch 2.10.0 that matches your driver
  (https://pytorch.org) first. Then run `pip install -r requirements.txt`, which keeps the installed torch 2.10.0.
- **Other platforms** (Windows, Intel macOS): `pip install -r requirements-dev.txt`. These platforms are untested.
- **Without exact pins:** `pip install -e ".[dense,dev]"` installs within the declared ranges. Each range starts at
  the tested version; anything newer is untested.

## Dependency roles

| Group | Packages | Needed for |
|---|---|---|
| Core | numpy, pydantic, PyYAML, requests, beautifulsoup4, lxml, rank-bm25 | EDGAR download, parsing and chunking, configs, BM25, metrics, evaluation |
| Dense (`[dense]`) | torch, transformers, sentence-transformers, tokenizers, safetensors, huggingface-hub | Dense retrieval, fine-tuning, checkpoint verification, the v1 analysis, the v2 experiments, the demo |
| Development (`[dev]`) | pytest, pytest-cov, ruff, black, mypy | Tests, lint, formatting, type checks |
| GPU training | CUDA PyTorch | Fine-tuning at the published scale (a T4 or larger); the trainer's tests run on CPU |
| Optional | orjson | Faster JSON. `finretriever.utils.io` falls back to the standard library with the same output, and the published results were produced without it |

Unused, and no longer listed: pandas, tqdm, pydantic-settings and pytrec_eval. The code never imports them, and
the metrics are computed in `finretriever.evaluation.metrics`.

## Network access

- The test suite needs none. It builds tiny local models and mocks the Hugging Face Hub. CI sets
  `HF_HUB_OFFLINE=1`.
- `make download` needs SEC EDGAR access and an `SEC_USER_AGENT` ([data_pipeline.md](data_pipeline.md)).
- Question generation needs a Groq API key (`.env.example`) and spends tokens.
- Downloading `BAAI/bge-base-en-v1.5` needs the Hugging Face Hub once; it is cached afterwards.

## Regenerating the locks

Use `uv pip compile`. The commands are in each lock file's header. Pass the analysis environment's `pip freeze` as
`-c`; without it, uv picks the newest versions and the lock no longer describes the tested environment.
