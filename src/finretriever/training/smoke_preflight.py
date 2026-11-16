"""Preflight for a trainer run on a GPU (the smoke test or a validated fine-tuning run): everything the run depends
on is checked; nothing is trained.

Run from the bundle directory (``scripts/smoke_preflight.py``). Each check passes, warns, fails or is
skipped:

1. ``bundle``: every file has its manifest SHA-256 and nothing unlisted is present (``smoke_bundle``).
2. ``config``: the trainer config is one the bundle was built for, unchanged.
3. ``corpus_and_preparation``: the frozen corpus and the preparation verify (file hashes, evidence
   index rebuilt and compared with its recorded digest) and are the ones recorded in the bundle.
4. ``candidates``: every training candidate verifies again (``candidates.load_candidates``), and
   ``provenance`` traces each to its source pair, accepting decision and generation record.
5. ``smoke_subset`` and ``schedule``: the config selects the expected pairs and optimizer steps,
   and every batch holds only allowed negatives.
6. ``evaluation_isolation``: no evaluation file exists in the bundle, and none was opened.
7. ``model_and_tokenizer``: the pinned revision resolves, the tokenizer is the one that defined the
   preparation's visible text, every query fits, truncating each positive passage keeps the pair's
   visible text (``biencoder_trainer.check_lengths``), and (unless skipped) the base weights load.
8. ``cuda_fp16`` and ``gpu_memory``: CUDA works with fp16 autocast and loss scaling, and the free
   memory (measured now, not assumed) covers the peak: the one measured by ``gradcache`` when there
   is one, else the ESTIMATED peak.

For a validated fine-tuning run, also:

* ``validation_set``: the validation set loads, re-verified and isolated from the pairs trained on
  (``trainer_setup.prepare_inputs``), and is the bundled one; it only selects checkpoints;
* ``schedule``: every seed's schedule is the bundled one;
* ``gradcache`` (with ``mini_batch_size``): on the loaded base model (eager attention, dropout on),
  GradCache's gradients equal those of the same forward passes without caching, on a few real
  pairs (``biencoder_trainer.gradcache_check``); on CUDA, the peak memory of one cached step of a
  full mini-batch is measured, plus AdamW's state (estimated). Gradients are cleared and no
  optimizer step is taken.

The base model is loaded in evaluation mode without gradients, except for the ``gradcache`` check;
no optimizer is built and no model file is written. The report goes to ``smoke_outputs/preflight.json``.
Exit codes: 0 ready; 1 a check failed; 3 nothing failed but checks were skipped (not ready for training).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TextIO

from finretriever.training.candidates import verify_candidate_provenance
from finretriever.training.smoke_bundle import (
    MANIFEST_NAME,
    evaluation_sources,
    trace_opened,
    under,
    verify_bundle,
)
from finretriever.training.trainer_setup import (
    estimate_gpu_memory_gib,
    load_trainer_config,
    plan_steps,
    prepare_inputs,
)

PREFIX = "[smoke_preflight]"
DEFAULT_CONFIG = "configs/train/biencoder_bge_base_smoke.yaml"
FALLBACK_CONFIG = "configs/train/biencoder_bge_base_smoke_gc.yaml"
DEFAULT_REPORT = "smoke_outputs/preflight.json"
MEMORY_MARGIN = 1.15  # free memory must exceed the estimate by 15% to pass; at least the estimate to warn
GRADCACHE_PAIRS = 4  # real pairs in the GradCache equivalence check, in mini-batches of 2
GRADCACHE_TOLERANCE = {"fp32": 1e-5, "fp16": 1e-3}  # relative difference of the gradient vectors


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail | skipped
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy", "pydantic"):
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "unknown"


def _guard(name: str, run: Callable[[], Check], checks: list[Check]) -> Check:
    try:
        check = run()
    except Exception as exc:  # a check that cannot complete fails; the others still run
        check = Check(name, "fail", f"{type(exc).__name__}: {exc}")
    checks.append(check)
    return check


def run_preflight(
    config_path: str = DEFAULT_CONFIG,
    *,
    root: Path = Path("."),
    offline: bool = False,
    skip_model_load: bool = False,
    skip_cuda: bool = False,
    bundle: bool = True,
    encoder_loader: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Run every check; see the module docstring. ``root`` must be the working directory: the
    configs' paths are relative to it."""
    if root.resolve() != Path.cwd().resolve():
        raise ValueError(f"run the preflight from {root} (the bundle directory)")
    started = time.time()
    checks: list[Check] = []
    manifest: dict[str, Any] = {}
    expected: dict[str, Any] = {}

    def check_bundle() -> Check:
        nonlocal manifest
        if not bundle:
            return Check("bundle", "skipped", "run from a repository checkout (--no-bundle)")
        report = verify_bundle(root)
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        return Check("bundle", "pass", f"{report['files']} files match their SHA-256; nothing unlisted", report)

    _guard("bundle", check_bundle, checks)
    expected = manifest.get("expected", {})
    config_box: dict[str, Any] = {}

    def check_config() -> Check:
        config, sha = load_trainer_config(root / config_path)
        config_box.update(config=config, sha=sha)
        if not bundle:
            return Check("config", "pass", f"{config_path} loads (sha256 {sha[:12]})")
        if not manifest:
            return Check("config", "fail", "no verified bundle manifest to check the config against")
        wanted = expected.get("configs", {}).get(config_path)
        if wanted is None:
            return Check("config", "fail", f"{config_path} is not a config this bundle was built for")
        if wanted["sha256"] != sha:
            return Check("config", "fail", f"{config_path} changed since the bundle was built")
        runs = ", ".join(config.for_seed(seed).run_name for seed in config.seed_list)
        return Check("config", "pass", f"{config_path} is unchanged (runs {runs})", {"sha256": sha})

    _guard("config", check_config, checks)
    config = config_box.get("config")
    if config is None:
        return _finish(checks, config_path, started, ready=False)

    state: dict[str, Any] = {}
    with trace_opened(root) as opened:
        def check_inputs() -> Check:
            inputs = prepare_inputs(config.for_seed(config.seed_list[0]), config_box["sha"])
            state["inputs"] = inputs
            identity = inputs.identity
            found = {
                "corpus": identity["corpus"],
                "passages": len(inputs.prepared.rows),
                "preparation": {k: identity["preparation"][k] for k in ("version", "manifest_sha256",
                                                                        "evidence_index_digest")},
            }
            if expected:
                if identity["corpus"] != expected["corpus"]:
                    return Check("corpus_and_preparation", "fail", "the corpus is not the bundled one", found)
                for key in ("version", "manifest_sha256", "evidence_index_digest"):
                    if identity["preparation"][key] != expected["preparation"][key]:
                        return Check("corpus_and_preparation", "fail", f"preparation {key} differs", found)
            return Check(
                "corpus_and_preparation", "pass",
                f"corpus {identity['corpus']['version']} ({found['passages']} passages, sha256 "
                f"{identity['corpus']['sha256'][:12]}) and preparation {identity['preparation']['version']} verify; "
                "evidence index rebuilt and matches its digest", found,
            )

        _guard("corpus_and_preparation", check_inputs, checks)
        inputs = state.get("inputs")

        def check_candidates() -> Check:
            assert inputs is not None
            available = inputs.identity["candidates"]["available"]
            data = {"version": config.data.candidates, "verified": available,
                    "manifest_sha256": inputs.identity["candidates"]["manifest_sha256"]}
            wanted = expected.get("candidates", {})
            if wanted and (available, data["manifest_sha256"]) != (wanted["available"], wanted["manifest_sha256"]):
                return Check("candidates", "fail", "the candidate dataset differs from the bundled one", data)
            return Check("candidates", "pass", f"all {available} training candidates re-verified (contract, "
                         "validator, evidence leakage, equivalents)", data)

        def check_provenance() -> Check:
            assert inputs is not None
            found = verify_candidate_provenance(inputs.prepared, config.data.candidates)
            wanted = expected.get("candidates", {}).get("provenance")
            if wanted and found != wanted:
                return Check("provenance", "fail", f"provenance counts {found} differ from {wanted}", found)
            return Check("provenance", "pass", f"{found['candidates']} candidates traced to their source pairs, "
                         f"accepting decisions and generation records {found['by_source']}", found)

        def check_subset() -> Check:
            assert inputs is not None
            pairs = inputs.identity["pairs"]
            data = {"pairs": pairs["count"], "pair_ids_sha256": pairs["pair_ids_sha256"]}
            wanted = expected.get("configs", {}).get(config_path)
            if wanted and (wanted["pairs"], wanted["pair_ids_sha256"]) != (data["pairs"], data["pair_ids_sha256"]):
                return Check("smoke_subset", "fail", f"the config selects {pairs['count']} pairs, not the expected "
                             f"{wanted['pairs']}", data)
            return Check("smoke_subset", "pass", f"{pairs['count']} of {inputs.identity['candidates']['available']} "
                         f"candidates selected (subset seed {pairs['subset']['seed']}), as expected", data)

        def check_schedule() -> Check:
            assert inputs is not None
            schedule = plan_steps(inputs)
            state["schedule"] = schedule
            data = {"steps": len(schedule.steps), "warmup_steps": schedule.warmup_steps,
                    "batches_per_epoch": list(schedule.batches_per_epoch), "sha256": schedule.sha256,
                    "dropped_single_pair_batches": schedule.dropped_batches}
            wanted = expected.get("configs", {}).get(config_path)
            if wanted and (wanted["steps"], wanted["schedule_sha256"]) != (data["steps"], data["sha256"]):
                return Check("schedule", "fail", "the schedule differs from the bundled one", data)
            if config.training.seeds is None:
                return Check("schedule", "pass", f"{data['steps']} optimizer steps; every batch holds only allowed "
                             "negatives", data)
            data = {"seeds": {}, "batching": schedule.batching}
            for seed in config.seed_list:
                run = plan_steps(inputs.for_seed(seed))
                data["seeds"][str(seed)] = {"steps": len(run.steps), "sha256": run.sha256,
                                            "dropped_single_pair_batches": run.dropped_batches,
                                            "in_batch_negatives_per_query": run.negatives}
                bundled = (wanted or {}).get("seeds", {}).get(str(seed))
                if wanted and (bundled is None or (bundled["steps"], bundled["schedule_sha256"]) != (
                        len(run.steps), run.sha256)):
                    return Check("schedule", "fail", f"seed {seed}'s schedule differs from the bundled one", data)
            steps = sorted({v["steps"] for v in data["seeds"].values()})
            return Check("schedule", "pass", f"seeds {list(config.seed_list)}: {steps} optimizer steps each; every "
                         "batch holds only allowed negatives", data)

        def check_validation() -> Check:
            assert inputs is not None
            found = {k: inputs.identity["validation"][k] for k in ("version", "manifest_sha256", "pairs",
                                                                  "pair_ids_sha256")}
            wanted = expected.get("configs", {}).get(config_path, {}).get("validation")
            if expected and found != wanted:
                return Check("validation_set", "fail", "the validation set differs from the bundled one", found)
            return Check("validation_set", "pass", f"{found['version']}: {found['pairs']} synthetic questions (AI "
                         f"review, not gold) re-verified and isolated from the {len(inputs.pairs)} pairs trained on; "
                         "used only to select checkpoints", found)

        if inputs is not None:
            named = [("candidates", check_candidates), ("provenance", check_provenance),
                     ("smoke_subset", check_subset), ("schedule", check_schedule)]
            if config.validation is not None:
                named.append(("validation_set", check_validation))
            for name, run in named:
                _guard(name, run, checks)

    def check_isolation() -> Check:
        sources = evaluation_sources(str(root / config.data.data_prep_config))
        present = [p for p in ("data/eval", *sources) if (root / p).exists()]
        touched = sorted(p for p in opened if under(p, ("data", "eval"))
                         or any(p == s or under(p, Path(s).parts) for s in sources))
        data = {"evaluation_sources_named_by_config": len(sources), "present": present, "opened": touched,
                "files_opened": len(opened)}
        if present or touched:
            return Check("evaluation_isolation", "fail", "evaluation data is present or was opened", data)
        return Check("evaluation_isolation", "pass", f"data/eval and all {len(sources)} evaluation sources named by "
                     f"the data-preparation config are absent; none of the {len(opened)} files read is one", data)

    _guard("evaluation_isolation", check_isolation, checks)

    def check_model() -> Check:
        if inputs is None:
            return Check("model_and_tokenizer", "fail", "inputs did not load")
        from huggingface_hub import hf_hub_download

        from finretriever.training.biencoder_trainer import (
            HFEncoder,
            TrainerError,
            _tokenizer_sha256,
            check_lengths,
            load_base_encoder,
        )

        model = config.base_model
        path = Path(hf_hub_download(model.name, "config.json", revision=model.revision, local_files_only=offline))
        commit = path.parent.name
        data: dict[str, Any] = {"name": model.name, "revision": model.revision, "resolved_commit": commit}
        if commit != model.revision:
            return Check("model_and_tokenizer", "fail", f"revision resolved to {commit}", data)
        if skip_model_load:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model.name, revision=model.revision, local_files_only=offline)
            sha = _tokenizer_sha256(tokenizer)
            encoder: Any = HFEncoder(None, tokenizer, model.pooling)  # the tokenizer only: no weights
        else:
            gradient_cache = config.training.mini_batch_size is not None
            loader = encoder_loader or (
                lambda m: load_base_encoder(m, local_files_only=offline, gradient_cache=gradient_cache))
            encoder = loader(model)
            if hasattr(encoder, "eval"):
                encoder.eval()  # weights are only read
            state["encoder"] = encoder
            described = encoder.describe()
            sha = described.get("tokenizer_sha256")
            data["model"] = {k: described.get(k) for k in ("class", "hidden_size", "parameters", "pooling")}
        expected_sha = inputs.identity["preparation"]["visible_text"]["tokenizer_sha256"]
        data.update(tokenizer_sha256=sha, preparation_tokenizer_sha256=expected_sha,
                    query_max_length=config.training.query_max_length,
                    passage_max_length=config.training.passage_max_length)
        if sha != expected_sha:
            return Check("model_and_tokenizer", "fail", "the tokenizer is not the one that defined the visible text",
                         data)
        try:
            data.update(check_lengths(inputs, encoder))  # as the trainer checks them before its first step
        except TrainerError as exc:
            return Check("model_and_tokenizer", "fail", str(exc), data)
        loaded = "weights not loaded (--skip-model-load)" if skip_model_load else (
            f"weights load ({_count(data['model']['parameters'])} parameters, hidden "
            f"{data['model']['hidden_size'] or 'unknown'})")
        return Check("model_and_tokenizer", "pass", f"{model.name} at {model.revision[:12]}; tokenizer matches the "
                     f"preparation; longest query {data['longest_query_tokens']} tokens; "
                     f"{data['passages_truncated_to_visible_length']} of {data['positive_passages']} positives "
                     f"truncated at {data['passage_max_length']} tokens, every visible text kept; {loaded}", data)

    _guard("model_and_tokenizer", check_model, checks)
    memory_box: dict[str, Any] = {}

    def check_cuda() -> Check:
        if skip_cuda:
            return Check("cuda_fp16", "skipped", "skipped (--skip-cuda: local testing only)")
        import torch

        if not torch.cuda.is_available():
            return Check("cuda_fp16", "fail", "CUDA is not available: enable the GPU accelerator")
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        memory_box.update(free=free, total=total)
        x = torch.randn(256, 256, device=device, requires_grad=True)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = x @ x.T
        loss = y.float().pow(2).mean()
        scaler.scale(loss).backward()
        finite = bool(torch.isfinite(x.grad).all())
        data = {"device": props.name, "capability": f"{props.major}.{props.minor}",
                "total_gib": round(total / 1024**3, 2), "free_gib": round(free / 1024**3, 2),
                "autocast_dtype": str(y.dtype), "fp16_gradients_finite": finite,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()), "torch_cuda": torch.version.cuda}
        del x, y, loss
        torch.cuda.empty_cache()
        if (props.major, props.minor) < (5, 3) or data["autocast_dtype"] != "torch.float16" or not finite:
            return Check("cuda_fp16", "fail", "fp16 autocast or loss scaling does not work on this GPU", data)
        if config.training.fp16 is False:
            return Check("cuda_fp16", "warn", "CUDA works, but the config trains in fp32", data)
        return Check("cuda_fp16", "pass", f"{props.name} (compute {data['capability']}): fp16 autocast and loss "
                     "scaling work", data)

    _guard("cuda_fp16", check_cuda, checks)

    def check_gradcache() -> Check:
        import torch

        from finretriever.training.biencoder_trainer import (
            cached_contrastive_backward,
            gradcache_check,
        )

        encoder, schedule = state.get("encoder"), state.get("schedule")
        if encoder is None or schedule is None or inputs is None:
            return Check("gradcache", "skipped", "the model was not loaded (--skip-model-load) or the inputs did not")
        cfg = config.training
        cuda = "free" in memory_box
        device = torch.device("cuda" if cuda else "cpu")
        fp16 = cfg.fp16 and cuda
        pairs = [inputs.by_id[i] for i in schedule.steps[0].batches[0]]
        queries = [inputs.queries[p.pair_id] for p in pairs]
        positives = [inputs.passages[p.positive_passage_id] for p in pairs]
        settings = {"temperature": cfg.temperature, "query_max_length": cfg.query_max_length,
                    "passage_max_length": cfg.passage_max_length, "device": device, "fp16": fp16}
        encoder.to(device)
        report = gradcache_check(encoder, queries[:GRADCACHE_PAIRS], positives[:GRADCACHE_PAIRS], mini_batch_size=2,
                                 **settings)
        data: dict[str, Any] = {"device": str(device), "fp16": fp16, "pairs": min(GRADCACHE_PAIRS, len(pairs)),
                                **report}
        if cuda:  # one cached step over a full mini-batch: the activations of one mini-batch at a time
            n = min(cfg.mini_batch_size or len(pairs), len(pairs))
            torch.cuda.reset_peak_memory_stats(device)
            encoder.train()
            cached_contrastive_backward(encoder, queries[:n], positives[:n], mini_batch_size=n, scale=lambda x: x * 1024,
                                        **settings)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            optimizer = 8 * sum(p.numel() for p in encoder.parameters()) / 1024**3  # AdamW's two fp32 moments
            encoder.zero_grad(set_to_none=True)
            encoder.eval()
            memory_box["measured_gib"] = round(peak + optimizer, 2)
            data.update(measured_step_peak_gib=round(peak, 2), adamw_state_gib_estimated=round(optimizer, 2),
                        mini_batch_pairs=n)
            torch.cuda.empty_cache()
        tolerance = GRADCACHE_TOLERANCE["fp16" if fp16 else "fp32"]
        if report["relative_difference"] > tolerance or report["cosine"] < 0.999:
            return Check("gradcache", "fail", f"GradCache gradients differ from the uncached passes (relative "
                         f"{report['relative_difference']:.2e} > {tolerance:.0e})", data)
        memory = (f"; one cached step over {data['mini_batch_pairs']} pairs peaked at {data['measured_step_peak_gib']} "
                  "GiB (measured)" if cuda else "")
        return Check("gradcache", "pass", f"GradCache gradients equal the uncached passes on {data['pairs']} real "
                     f"pairs with dropout (relative difference {report['relative_difference']:.1e}, {device}"
                     f"{', fp16' if fp16 else ''}){memory}", data)

    if config.training.mini_batch_size is not None:
        _guard("gradcache", check_gradcache, checks)

    def check_memory() -> Check:
        cfg = config.training
        batch = cfg.mini_batch_size or cfg.batch_size
        estimate = estimate_gpu_memory_gib(batch_size=batch, query_tokens=cfg.query_max_length,
                                           passage_tokens=cfg.passage_max_length, fp16=cfg.fp16,
                                           gradient_checkpointing=cfg.gradient_checkpointing)
        fallback = estimate_gpu_memory_gib(batch_size=batch, query_tokens=cfg.query_max_length,
                                           passage_tokens=cfg.passage_max_length, fp16=cfg.fp16,
                                           gradient_checkpointing=True)
        data = {"estimate_gib (not measured)": estimate, "estimate_with_gradient_checkpointing_gib": fallback}
        if "free" not in memory_box:
            return Check("gpu_memory", "skipped", f"ESTIMATED peak {estimate['total_gib']} GiB; free GPU memory not "
                         "measured (no CUDA check)", data)
        free_gib = memory_box["free"] / 1024**3
        data["free_gib (measured now)"] = round(free_gib, 2)
        if "measured_gib" in memory_box:  # measured by the gradcache check: one step's peak plus AdamW's state
            need = memory_box["measured_gib"]
            data["peak_gib (measured step + estimated optimizer state)"] = need
            if free_gib >= need * MEMORY_MARGIN:
                return Check("gpu_memory", "pass", f"{free_gib:.1f} GiB free (measured) against a {need} GiB peak "
                             "(measured step plus optimizer state)", data)
            return Check("gpu_memory", "warn" if free_gib >= need else "fail", f"{free_gib:.1f} GiB free against a "
                         f"{need} GiB peak (measured step plus optimizer state); lower mini_batch_size", data)
        need = estimate["total_gib"]
        if free_gib >= need * MEMORY_MARGIN:
            return Check("gpu_memory", "pass", f"{free_gib:.1f} GiB free (measured) against an ESTIMATED {need} GiB "
                         "peak", data)
        hint = "" if cfg.gradient_checkpointing else f"; use {FALLBACK_CONFIG} (ESTIMATED {fallback['total_gib']} GiB)"
        if free_gib >= need:
            return Check("gpu_memory", "warn", f"only {free_gib:.1f} GiB free against an ESTIMATED {need} GiB "
                         f"peak{hint}", data)
        return Check("gpu_memory", "fail", f"{free_gib:.1f} GiB free is less than the ESTIMATED {need} GiB "
                     f"peak{hint}", data)

    _guard("gpu_memory", check_memory, checks)
    return _finish(checks, config_path, started, ready=True)


def _finish(checks: list[Check], config_path: str, started: float, *, ready: bool) -> dict[str, Any]:
    failed = [c.name for c in checks if c.status == "fail"]
    skipped = [c.name for c in checks if c.status == "skipped"]
    return {
        "kind": "finretriever_smoke_preflight",
        "config": config_path,
        "checks": [asdict(c) for c in checks],
        "failed": failed,
        "skipped": skipped,
        "ready_for_training": ready and not failed and not skipped,
        "trained": False,
        "model_weights_written": False,
        "seconds": round(time.time() - started, 1),
        "environment": _versions(),
    }


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="smoke_preflight.py", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"Trainer config (default {DEFAULT_CONFIG}).")
    parser.add_argument("--report", default=DEFAULT_REPORT, help=f"Where to write the JSON report ({DEFAULT_REPORT}).")
    parser.add_argument("--offline", action="store_true", help="Use only the local Hugging Face cache.")
    parser.add_argument("--skip-model-load", action="store_true", help="Check the tokenizer and revision only.")
    parser.add_argument("--skip-cuda", action="store_true", help="Local testing only: skip the GPU checks.")
    parser.add_argument("--no-bundle", action="store_true", help="Run from a repository checkout, not a bundle.")
    args = parser.parse_args(argv)
    report = run_preflight(args.config, offline=args.offline, skip_model_load=args.skip_model_load,
                           skip_cuda=args.skip_cuda, bundle=not args.no_bundle)
    for check in report["checks"]:
        print(f"{PREFIX} {check['status'].upper():7s} {check['name']}: {check['detail']}", file=out)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["failed"]:
        print(f"{PREFIX} NOT READY: failed {report['failed']} (report: {path})", file=out)
        return 1
    if report["skipped"]:
        print(f"{PREFIX} NOT READY FOR TRAINING: skipped {report['skipped']} (report: {path})", file=out)
        return 3
    print(f"{PREFIX} READY FOR TRAINING (nothing was trained; report: {path})", file=out)
    return 0


__all__ = ["Check", "main", "run_preflight"]
