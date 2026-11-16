"""Command line for training-data preparation; backs ``scripts/02_generate_queries.py``.

::

    prepare       overlap groups, evaluation blocklist, train/validation split, visible text, manifest (offline)
    sources       how many approved passages each source-quality rule excludes (offline, writes nothing)
    plan          write a versioned sampling plan for a pilot (offline; plans are never changed);
                  --exclude-evidence-of skips passages that may answer a training candidate's question,
                  --exclude-validation-set the units of an existing validation set's passages; --one-per group
                  allows several groups of one split unit; --min-isolated-share keeps and prefers passages
                  whose sentences no passage outside the split holds; --max-per-company-year and
                  --max-per-unit cap the passages of one company and fiscal year and of one split unit;
                  --predictor scores passages by a calibrated evidence-pass prediction, --score-priority ranks
                  by it within the year balance, --after-generated skips the groups of passages already
                  generated and counts them toward the caps, --exclude-candidate-groups skips the groups of
                  every passage a training-candidate dataset uses, --administrative-rule adds that source rule
    calibrate-predictor
                  fit the evidence-pass predictor on validation pilots' records and AI-review decisions
                  (offline, write-once JSON)
    generate      synthetic queries for one split through the configured API; without --confirm-api-use it
                  only prints the plan, and it needs the API key in the environment variable the config names;
                  --continue-run resumes a run whose request settings (token limit, pacing, retries) changed;
                  a stopped plan is refused before any credential is read
    stop-plan     close a plan after its last generated position: a stop record beside the plan, after which
                  generate --plan refuses it (offline, write-once)
    revalidate    re-run the current candidate checks on recorded candidates (offline, append-only)
    build-pairs   pairs from the generation records, global and quality checks, pairs manifest; written
                  once: pairs that would differ from those present are refused
    check         re-run the quality checks on the written pairs
    carry-over    re-check an earlier preparation's pairs against this one and write the eligible ones
                  as development training candidates under <preparation>/carryover/ (offline)
    build-candidates
                  a versioned training-candidate dataset: the pairs of the configured sources that an AI
                  quality review accepted and that pass every current check, with a log of every pair
                  and a manifest, under <preparation>/candidates/<version>/ (offline, write-once)
    build-validation
                  a versioned validation-pair set from one validation plan's records and carried validation
                  sets: questions an AI quality review accepted, whose evidence no passage outside validation
                  holds and which share nothing with the training data, under
                  <preparation>/validation_sets/<version>/ (offline, write-once)
    pin-split     the split pins a new preparation keeps (this preparation's training data and validation
                  passages), written once to --out (offline)
    screen-questions
                  the semantic leakage screen: training questions (of candidate datasets and/or the built
                  train pairs) against the permitted evaluation question texts, with the preparation's pinned
                  zero-shot encoder, under <preparation>/screening/<name>/ (offline, write-once); a candidates
                  config that names it (semantic_screening) excludes flagged and unscreened pairs

Exit codes: 0 done; 1 refused or a quality check failed (nothing written); 2 configuration, corpus or
credential error. The API key is never printed, logged or written.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from finretriever.eval_set.annotation_store import load_corpus_passages
from finretriever.evaluation.provenance import (
    CorpusIntegrityError,
    CorpusMismatchError,
    sha256_file,
)
from finretriever.training.blocklist import BlocklistError
from finretriever.training.candidates import (
    CANDIDATE_FILES,
    CandidateError,
    SourcePreparation,
    build_candidates,
    candidates_dir,
    load_candidates,
    load_candidates_config,
    read_candidates_manifest,
    write_candidates,
)
from finretriever.training.carryover import (
    CARRYOVER_DIRNAME,
    CarryOverError,
    carry_over,
    load_quality_decisions,
    write_carry_over,
)
from finretriever.training.data_prep import (
    DataPrepError,
    load_data_prep_config,
    open_prepared,
    prepare,
    prepared_status,
    verify_frozen_corpus,
    write_prepared,
)
from finretriever.training.evidence_prediction import (
    PredictionError,
    build_calibration,
    describe,
    load_calibration,
    observe,
    predict,
    write_calibration,
)
from finretriever.training.pair_quality import (
    GENERATION_DIRNAME,
    PAIRS_DIRNAME,
    PAIRS_META_FILE,
    PairQualityError,
    aliases_for,
    build_pairs,
    check_pairs,
    evidence_locator,
    load_evaluation_questions,
    read_pairs,
    write_pairs,
)
from finretriever.training.pinning import build_split_pins
from finretriever.training.query_generation import (
    VALIDATOR_VERSION,
    CredentialError,
    GenerationError,
    GroqClient,
    ModelUnavailable,
    QueryGenerator,
    RequestTooLarge,
    Transport,
    api_key_from_env,
    check_run,
    continuations_path,
    describe_changes,
    latest_records,
    prompt_sha256,
    read_records,
    records_path,
    revalidate_records,
    sample_sources,
)
from finretriever.training.sampling_plan import (
    BALANCED_METHODS,
    METHODS,
    PlanError,
    load_plan,
    make_plan,
    make_stop_record,
    plan_path,
    read_stop_record,
    stop_path,
    write_plan,
    write_stop_record,
)
from finretriever.training.semantic_screen import (
    ScreeningError,
    Subject,
    build_screening,
    default_encoder,
    read_screening,
    screening_dir,
    write_screening,
)
from finretriever.training.source_quality import filings_per_group, filter_sources
from finretriever.training.split_pins import SplitPinsError, write_split_pins
from finretriever.training.splits import SplitError
from finretriever.training.validation_set import (
    ISOLATION_METHOD,
    PILOT_GENERATOR_ACCEPTANCE,
    PILOT_REVIEW_ACCEPTANCE,
    CarriedSource,
    ValidationSetError,
    build_validation_set,
    isolated_evidence_share,
    load_validation_config,
    load_validation_decisions,
    read_validation_manifest,
    records_prefix,
    validation_dir,
    write_validation_set,
)
from finretriever.training.validation_set import LOG_FILE as VALIDATION_LOG_FILE
from finretriever.training.visible_text import VisibleTextError, load_tokenizer

PREFIX = "[02_generate_queries]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="02_generate_queries.py", description="Leakage-safe training-data preparation."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default="configs/train/data_prep_v1.yaml", help="Data-preparation YAML config."
    )
    sub.add_parser(
        "prepare",
        parents=[common],
        help="Overlap groups, blocklist, split and visible text (offline).",
    )
    p = sub.add_parser(
        "sources",
        parents=[common],
        help="Report what the source-quality filter keeps and excludes (offline).",
    )
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--administrative-rule", action="store_true",
                   help="Add the administrative_disclosure source rule to the config's settings.")
    p = sub.add_parser(
        "plan", parents=[common], help="Write a versioned sampling plan for a pilot (offline)."
    )
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--name", required=True, help="Plan name, e.g. pilot_v2 (never reused).")
    p.add_argument("--size", required=True, type=int, help="Number of passages.")
    p.add_argument("--seed", type=int, help="Selection seed (default: the split seed).")
    p.add_argument("--method", choices=METHODS, default="company_year_round_robin")
    p.add_argument(
        "--no-source-filter",
        action="store_true",
        help="Select from every approved passage (as the original pilot did).",
    )
    p.add_argument(
        "--max-company-share",
        type=float,
        default=0.25,
        help="balanced: the largest share of the plan any one company may take (default 0.25).",
    )
    p.add_argument(
        "--exclude-carryover",
        action="append",
        default=[],
        metavar="NAME",
        help="balanced: skip the split units of this carry-over's eligible pairs (repeatable).",
    )
    p.add_argument(
        "--exclude-evidence-of",
        action="append",
        default=[],
        metavar="CANDIDATES",
        help="balanced: skip the split units of passages of the split that hold, exactly or possibly, the "
        "evidence of a pair of this training-candidate dataset (repeatable).",
    )
    p.add_argument(
        "--exclude-validation-set",
        action="append",
        default=[],
        metavar="CONFIG:VERSION",
        help="balanced: skip the split units of every passage of this validation set (its plan's passages, carried "
        "pairs and their equivalents), built in the preparation of CONFIG (repeatable).",
    )
    p.add_argument(
        "--one-per",
        choices=("unit", "group"),
        default="unit",
        help="balanced: at most one passage per split unit (default) or per overlap group only.",
    )
    p.add_argument(
        "--min-isolated-share",
        type=float,
        help="balanced: only passages with at least this share of visible sentences that no passage outside the "
        "split holds, preferring higher shares (validation_set.isolated_evidence_share).",
    )
    p.add_argument(
        "--max-per-company-year",
        type=int,
        metavar="N",
        help="balanced: at most N passages per company and fiscal year (one 10-K each here).",
    )
    p.add_argument(
        "--max-per-unit",
        type=int,
        metavar="N",
        help="balanced, with --one-per group: at most N passages per split unit.",
    )
    p.add_argument(
        "--predictor",
        metavar="CALIBRATION",
        help="balanced: score passages by the calibrated evidence-pass predictor (evidence_prediction) in this "
        "calibration file, preferring higher scores.",
    )
    p.add_argument(
        "--min-predicted-pass",
        type=float,
        metavar="P",
        help="with --predictor: only passages whose predicted evidence-pass probability is at least P.",
    )
    p.add_argument(
        "--score-priority",
        action="store_true",
        help="balanced, with scores: rank by score right after the fiscal-year balance, not as the last tie-break.",
    )
    p.add_argument(
        "--after-generated",
        nargs="*",
        metavar="CONFIG",
        help="balanced: skip the overlap groups of every passage of the split with a generation record in this "
        "preparation and in each CONFIG's (another preparation of the corpus), and count those passages toward "
        "--max-per-unit and --max-per-company-year.",
    )
    p.add_argument(
        "--exclude-equivalents-of",
        action="append",
        default=[],
        metavar="CONFIG:VERSION",
        help="balanced: skip the overlap groups of every question of this validation set and of its equivalents "
        "(its units stay open, within --max-per-unit), built in the preparation of CONFIG (repeatable).",
    )
    p.add_argument(
        "--exclude-candidate-groups",
        action="append",
        default=[],
        metavar="CANDIDATES",
        help="balanced: skip the overlap groups of every passage a pair of this training-candidate dataset uses (its "
        "positive and its exact and uncertain equivalents), so no new question is asked of them (repeatable).",
    )
    p.add_argument(
        "--administrative-rule",
        action="store_true",
        help="Add the administrative_disclosure source rule (insider trading, trading plans, cybersecurity "
        "governance) to the config's source-quality settings for this plan.",
    )
    p = sub.add_parser(
        "calibrate-predictor",
        parents=[common],
        help="Fit the evidence-pass predictor on validation pilots (offline, write-once).",
    )
    p.add_argument("--out", required=True, help="Calibration file to write (JSON).")
    p.add_argument(
        "--pilot",
        action="append",
        nargs=4,
        required=True,
        metavar=("CONFIG", "PLAN", "DECISIONS", "POSITIONS"),
        help="A validation pilot: its data-preparation config, plan name, AI-review decisions file and how many "
        "plan positions (from the first) were generated and reviewed (repeatable).",
    )
    p = sub.add_parser(
        "generate", parents=[common], help="Generate synthetic queries for one split (resumable)."
    )
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--limit", type=int, help="At most this many passages in this run.")
    p.add_argument("--plan", help="Use the passages of this sampling plan, in plan order.")
    p.add_argument(
        "--sample",
        type=int,
        help="Legacy pilot selection: this many approved passages spread over companies, unfiltered "
        "(the original pilot; prefer --plan).",
    )
    p.add_argument("--sample-seed", type=int, help="Seed for --sample (default: the split seed).")
    p.add_argument(
        "--max-requests",
        type=int,
        help="Hard cap on HTTP requests in this run, retries included; the run stops cleanly at it.",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also retry passages whose earlier attempt failed.",
    )
    p.add_argument(
        "--continue-run",
        action="store_true",
        help="If only request settings (token limit, pacing, retries) differ from the recorded run, record a "
        "continuation and resume in the same directory; earlier records are kept and skipped.",
    )
    p.add_argument(
        "--confirm-api-use",
        action="store_true",
        help="Send requests; without it only the plan is printed.",
    )
    p = sub.add_parser(
        "stop-plan",
        parents=[common],
        help="Close a plan after its last generated position; generate refuses it afterwards (offline, write-once).",
    )
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--plan", required=True, help="The plan to close.")
    p.add_argument("--reason", required=True, help="Why generation stops (recorded).")
    p = sub.add_parser(
        "revalidate",
        parents=[common],
        help="Re-run the current candidate checks on recorded candidates (no API use).",
    )
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument(
        "--dry-run", action="store_true", help="Report what would change; write nothing."
    )
    sub.add_parser(
        "build-pairs",
        parents=[common],
        help="Build and check the pairs from the generation records.",
    )
    sub.add_parser(
        "check", parents=[common], help="Re-run the quality checks on the written pairs."
    )
    p = sub.add_parser(
        "carry-over",
        parents=[common],
        help="Re-check an earlier preparation's pairs against this one (offline, write-once).",
    )
    p.add_argument("--from-config", required=True, help="Config of the earlier preparation.")
    p.add_argument("--decisions", required=True, help="Quality decisions (JSONL) for its pairs.")
    p.add_argument("--name", required=True, help="Name of the carry-over, e.g. pilot_v2.")
    p = sub.add_parser(
        "build-candidates",
        parents=[common],
        help="Build a versioned training-candidate dataset from accepted pairs (offline, write-once).",
    )
    p.add_argument(
        "--candidates", required=True, help="Candidates YAML config (its data_prep_config is --config)."
    )
    p = sub.add_parser(
        "build-validation",
        parents=[common],
        help="Build a versioned validation-pair set from a validation plan's records (offline, write-once).",
    )
    p.add_argument(
        "--validation", required=True, help="Validation-set YAML config (its data_prep_config is --config)."
    )
    p = sub.add_parser(
        "pin-split",
        parents=[common],
        help="Write the split pins a new preparation keeps from this one (offline, write-once).",
    )
    p.add_argument("--out", required=True, help="Pins file to write (JSON), e.g. configs/train/split_pins_v3.json.")
    p.add_argument("--candidates", action="append", default=[], metavar="VERSION",
                   help="Pin the train pairs of this candidate dataset and every train passage holding their evidence.")
    p.add_argument("--pairs", action="store_true",
                   help="Pin the built train pairs and every train passage holding their evidence.")
    p.add_argument("--plan", action="append", default=[], metavar="NAME",
                   help="Pin the passages of this train plan (repeatable).")
    p.add_argument("--keep-validation", action="store_true", help="Pin every validation passage to validation.")
    p = sub.add_parser(
        "screen-questions",
        parents=[common],
        help="Screen training questions against the permitted evaluation question texts by meaning (offline, "
        "write-once).",
    )
    p.add_argument("--name", required=True, help="Name of the screening record, e.g. train_v3_p1_batch1.")
    p.add_argument("--candidates", action="append", default=[], metavar="VERSION",
                   help="Screen the training questions of this candidate dataset (repeatable).")
    p.add_argument("--pairs", action="store_true", help="Screen the built train pairs of this preparation.")
    p.add_argument("--allow-download", action="store_true",
                   help="Let the encoder download from Hugging Face (default: the local cache only).")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    tokenizer_loader: Callable[..., Any] = load_tokenizer,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    out: TextIO | None = None,
    err: TextIO | None = None,
    screen_encoder: Callable[[Any], Any] | None = None,
) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    args = build_parser().parse_args(argv)

    def say(text: str) -> None:
        print(f"{PREFIX} {text}", file=out)

    try:
        config, config_sha256 = load_data_prep_config(args.config)
        corpus = verify_frozen_corpus(config)
        passages = load_corpus_passages(corpus)
        if args.command == "prepare":
            files = prepare(
                config,
                config_path=args.config,
                config_sha256=config_sha256,
                corpus=corpus,
                passages=passages,
                tokenizer=tokenizer_loader(config.visible_text),
            )
            status = prepared_status(config.output_dir, files)
            written = write_prepared(config.output_dir, files)
            prepared = open_prepared(config.output_dir, corpus, passages)
            m = prepared.manifest
            o, b, s = m["overlap"]["stats"], m["blocklist"]["stats"], m["split"]["stats"]
            outcome = {
                "absent": "wrote",
                "identical": "unchanged (identical copy present)",
                "config_edited": "unchanged (identical data and settings; the config file changed only in "
                "sections the preparation does not use, so its recorded SHA-256 is kept)",
            }[status]
            say(f"{outcome} preparation {config.version} at {config.output_dir}")
            say(
                f"overlap groups: {o['groups']} ({o['singleton_groups']} single passages, largest {o['largest_group']}); "
                f"edges {o['edges_by_kind']}"
            )
            say(
                f"blocked: {b['blocked']} passages ({b['direct']} evaluation candidates + {b['via_overlap_group']} in their groups)"
            )
            if "evidence_reuse" in b:
                r = m["reuse"]
                say(
                    f"evidence reuse: {b['evidence_reuse']} more passages reuse evaluation evidence {b['evidence_reuse_by_rule']}, "
                    f"+ {b['via_evidence_reuse_group']} in their groups; {r['split_links']} links join groups for the split"
                )
                v = r["verification"]
                say(
                    f"verified: no remaining passage substantially reuses evaluation evidence; no link crosses the split; "
                    f"residual sub-threshold reuse ({v['residual_passages_sharing_units_with_evaluation']} passages with evaluation, "
                    f"{v['residual_train_passages_sharing_units_with_validation']} train passages with validation) is checked per pair"
                )
            say(
                f"split: {s['passages']} passages, validation share of usable {s['validation_share_of_usable']}"
            )
            say(
                f"generation sources (>= {config.min_source_words} visible words): {m['generation_sources']}"
            )
            return 0
        prepared = open_prepared(config.output_dir, corpus, passages)
        if args.command == "pin-split":
            doc = build_split_pins(
                prepared, passages, config, config_path=args.config, candidates=args.candidates, pairs=args.pairs,
                plans=args.plan, keep_validation=args.keep_validation,
            )
            written = write_split_pins(args.out, doc)
            say(f"{'wrote' if written else 'unchanged (identical pins present)'} {args.out}")
            say(f"pinned passages: {doc['counts']}; sources: candidates {args.candidates or 'none'}, built pairs "
                f"{'yes' if args.pairs else 'no'}, plans {args.plan or 'none'}, validation "
                f"{'all' if args.keep_validation else 'none'}")
            return 0
        if args.command == "generate":
            return _generate(
                args, config, prepared, passages, transport=transport, sleep=sleep, say=say
            )
        if args.command == "stop-plan":
            return _stop_plan(args, config, prepared, passages, say=say)
        if args.command in ("sources", "plan"):
            return _plan(args, config, prepared, passages, corpus=corpus, say=say)
        if args.command == "calibrate-predictor":
            return _calibrate(args, config, corpus, passages, say=say)
        if args.command == "revalidate":
            return _revalidate(args, config, prepared, passages, say=say)
        if args.command == "screen-questions":
            return _screen_questions(args, config, prepared, screen_encoder, say=say)
        questions = load_evaluation_questions(config.leakage.evaluation_question_files)
        if args.command == "carry-over":
            source_config, _ = load_data_prep_config(args.from_config)
            if (source_config.corpus_sha256, source_config.corpus_version) != (
                config.corpus_sha256,
                config.corpus_version,
            ):
                raise CarryOverError(f"{args.from_config} was prepared from another corpus")
            carried = carry_over(
                name=args.name,
                source=open_prepared(source_config.output_dir, corpus, passages),
                target=prepared,
                target_config=config,
                passages=passages,
                decisions=load_quality_decisions(args.decisions),
                decisions_path=args.decisions,
                evaluation_questions=questions,
            )
            written, path = write_carry_over(prepared, carried)
            stats = carried.meta["stats"]
            say(f"{'wrote' if written else 'unchanged (identical copy present)'} {path}")
            say(
                f"{stats['eligible']} of {stats['pairs']} pairs eligible as development training candidates"
            )
            say(f"ineligible by reason: {stats['ineligible_by_reason'] or 'none'}")
            say(f"eligible by company: {stats['eligible_by_ticker']}")
            say(f"eligible by fiscal year: {stats['eligible_by_fiscal_year']}")
            return 0
        if args.command == "build-candidates":
            return _build_candidates(
                args, config, config_sha256, corpus, prepared, passages, questions, say=say
            )
        if args.command == "build-validation":
            return _build_validation(args, config, config_sha256, corpus, prepared, passages, questions, say=say)
        if args.command == "build-pairs":
            pair_set = build_pairs(config, prepared, passages, questions)
            written, paths = write_pairs(
                pair_set, config, prepared, config_path=args.config, config_sha256=config_sha256
            )
            say(f"{'wrote' if written else 'unchanged (identical pairs present)'} {paths[PAIRS_META_FILE].parent}")
            for split, stats in pair_set.stats["by_split"].items():
                say(f"{split}: {stats}")
            say(f"rejected by reason: {pair_set.stats['rejected_by_reason'] or 'none'}")
            say("quality checks passed; pairs are training data only, never evaluation data")
            return 0
        violations = check_pairs(read_pairs(prepared), config, prepared, passages, questions)
        for violation in violations:
            print(f"{PREFIX} violation: {violation}", file=err)
        say("quality checks passed" if not violations else f"{len(violations)} violations")
        return 1 if violations else 0
    except (
        CredentialError,
        ModelUnavailable,
        RequestTooLarge,
        VisibleTextError,
        CorpusIntegrityError,
        CorpusMismatchError,
    ) as exc:
        print(f"{PREFIX} error: {exc}", file=err)
        return 2
    except (
        DataPrepError,
        BlocklistError,
        SplitError,
        PairQualityError,
        GenerationError,
        PlanError,
        CarryOverError,
        CandidateError,
        ValidationSetError,
        SplitPinsError,
        PredictionError,
        ValueError,
    ) as exc:
        print(f"{PREFIX} refused: {exc}", file=err)
        return 1


def _build_candidates(
    args: argparse.Namespace,
    config: Any,
    config_sha256: str,
    corpus: Any,
    prepared: Any,
    passages: Any,
    questions: Any,
    *,
    say: Callable[[str], None],
) -> int:
    candidates_config, candidates_sha256 = load_candidates_config(args.candidates)
    if Path(candidates_config.data_prep_config).resolve() != Path(args.config).resolve():
        raise CandidateError(
            f"{args.candidates} is for {candidates_config.data_prep_config}, not {args.config}"
        )
    sources = []
    for source in candidates_config.sources:
        if source.data_prep_config is None:
            sources.append(SourcePreparation(source, prepared))
            continue
        earlier, _ = load_data_prep_config(source.data_prep_config)
        if (earlier.corpus_sha256, earlier.corpus_version) != (
            config.corpus_sha256,
            config.corpus_version,
        ):
            raise CandidateError(f"{source.data_prep_config} was prepared from another corpus")
        sources.append(
            SourcePreparation(source, open_prepared(earlier.output_dir, corpus, passages))
        )
    screening = None
    if candidates_config.semantic_screening is not None:
        screening = read_screening(prepared, candidates_config.semantic_screening)
    built = build_candidates(
        candidates_config,
        config_path=args.candidates,
        config_sha256=candidates_sha256,
        target=prepared,
        target_config=config,
        target_config_sha256=config_sha256,
        sources=sources,
        passages=passages,
        evaluation_questions=questions,
        screening=screening,
    )
    written, directory = write_candidates(prepared, built)
    stats = built.manifest["stats"]
    say(f"{'wrote' if written else 'unchanged (identical copy present)'} {directory}")
    for name, s in stats["by_source"].items():
        say(f"{name}: {s['included']} of {s['pairs']} pairs included; excluded by reason: "
            f"{s['excluded_by_reason'] or 'none'}")
    say(f"candidates: {stats['included']} {stats['included_by_split']}; largest company share "
        f"{stats['largest_company_share']}")
    say(f"by company: {stats['included_by_ticker']}")
    say(f"by fiscal year: {stats['included_by_fiscal_year']}")
    say(f"by question type: {stats['included_by_question_type']}")
    say(f"by passage kind: {stats['included_by_passage_kind']}")
    say("accepted by an AI quality review: not human-validated, not gold labels, never evaluation data")
    return 0


def _screen_questions(
    args: argparse.Namespace,
    config: Any,
    prepared: Any,
    encoder_loader: Callable[[Any], Any] | None,
    *,
    say: Callable[[str], None],
) -> int:
    """``screen-questions``: see ``finretriever.training.semantic_screen``."""
    subjects: list[Subject] = []
    sources: list[dict[str, Any]] = []
    for version in args.candidates:
        _, manifest_sha256 = read_candidates_manifest(prepared, version)  # its files are those recorded
        path = candidates_dir(prepared, version) / CANDIDATE_FILES["train"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        subjects += [Subject(f"candidates:{version}", r["pair_id"], r["query"]) for r in rows]
        sources.append({"kind": "candidates", "version": version, "manifest_sha256": manifest_sha256,
                        "questions": len(rows)})
    if args.pairs:
        rows = read_pairs(prepared)["train"]  # checked against its pairs manifest
        subjects += [Subject(f"pairs:{prepared.manifest['version']}", r["pair_id"], r["query"]) for r in rows]
        sources.append({"kind": "pairs", "split": "train", "questions": len(rows),
                        "pairs_meta_sha256": sha256_file(prepared.directory / PAIRS_DIRNAME / PAIRS_META_FILE)})
    if not subjects:
        raise ScreeningError("nothing to screen: name --candidates VERSION or --pairs")
    load = encoder_loader or (lambda c: default_encoder(c, local_files_only=not args.allow_download))
    files = build_screening(args.name, prepared=prepared, config=config, subjects=subjects, sources=sources,
                            encoder=load(config))
    written = write_screening(prepared, args.name, files)
    manifest = json.loads(files["screening.json"])
    stats = manifest["stats"]
    say(f"{'wrote' if written else 'unchanged (identical copy present)'} {screening_dir(prepared, args.name)}")
    say(f"{manifest['subjects']['questions']} distinct training questions against "
        f"{sum(f['questions'] for f in manifest['evaluation_questions']['files'])} permitted evaluation question "
        f"texts ({', '.join(f['path'] for f in manifest['evaluation_questions']['files'])}; query_id and text only)")
    say(f"flagged at cosine >= {manifest['threshold']}: {stats['flagged']}; highest similarity {stats['max_similarity']}")
    say("no qrels, relevance labels, judged passages, pools or audits were read; the record is used only when "
        "candidates are built, never by training or checkpoint selection")
    return 0


def _calibrate(args: argparse.Namespace, config: Any, corpus: Any, passages: Any, *, say: Callable[[str], None]) -> int:
    """``calibrate-predictor``: each pilot's observations in its own preparation, then the fit, written once."""
    observations, sources = [], []
    for path, plan_name, decisions_path, positions in args.pilot:
        other, digest, prepared = _other_preparation(path, config, corpus, passages)
        if not positions.isdigit() or int(positions) < 1:
            raise PlanError(f"--pilot POSITIONS must be a positive number, not {positions!r}")
        count = int(positions)
        approved = prepared.source_passages("validation", passages, min_words=other.min_source_words)
        _, planned = load_plan(prepared.directory, "validation", plan_name, {p.passage_id: p for p in approved},
                               prepared_manifest_sha256=prepared.manifest_sha256)
        if count > len(planned):
            raise PlanError(f"plan {plan_name} has {len(planned)} passages, not {count}")
        records_file = records_path(prepared.directory / GENERATION_DIRNAME, "validation")
        state = latest_records(read_records(records_file)) if records_file.is_file() else {}
        if missing := [p.passage_id for p in planned[:count] if state.get(p.passage_id, {}).get("status") != "done"]:
            raise PlanError(f"{len(missing)} of plan {plan_name}'s first {count} passages have no done generation record")
        locator = evidence_locator(prepared, passages)
        found = observe(
            pilot=plan_name,
            planned=planned[:count],
            records=state,
            decisions=load_validation_decisions(decisions_path),
            locator=locator,
            rows=prepared.rows,
            sentence_shares=isolated_evidence_share(planned[:count], locator, prepared.rows),
        )
        if undecided := [o["plan_position"] for o in found if o["generated"] and o["review"] is None]:
            raise PlanError(f"{decisions_path} decides no question of {plan_name} positions {undecided[:5]}")
        lines, records_sha256 = records_prefix(records_file)
        observations.extend(found)
        sources.append({
            "config": path,
            "config_sha256": digest,
            "prepared_manifest_sha256": prepared.manifest_sha256,
            "plan": plan_name,
            "plan_sha256": sha256_file(plan_path(prepared.directory, "validation", plan_name)),
            "positions": count,
            "records": {"path": str(records_file), "lines": lines, "sha256": records_sha256},
            "decisions": {"path": decisions_path, "sha256": sha256_file(decisions_path)},
        })
    doc = build_calibration(observations, sources)
    written = write_calibration(args.out, doc)
    say(f"{'wrote' if written else 'unchanged (identical calibration present)'} {args.out}")
    for line in describe(doc):
        say(line)
    return 0


def _other_preparation(path: str, config: Any, corpus: Any, passages: Any) -> tuple[Any, str, Any]:
    """Another preparation of the same corpus: its config, the config's SHA-256 and the opened preparation."""
    other, digest = load_data_prep_config(path)
    if (other.corpus_sha256, other.corpus_version) != (config.corpus_sha256, config.corpus_version):
        raise DataPrepError(f"{path} was prepared from another corpus")
    return other, digest, open_prepared(other.output_dir, corpus, passages)


def _build_validation(
    args: argparse.Namespace,
    config: Any,
    config_sha256: str,
    corpus: Any,
    prepared: Any,
    passages: Any,
    questions: Any,
    *,
    say: Callable[[str], None],
) -> int:
    validation_config, validation_sha256 = load_validation_config(args.validation)
    if Path(validation_config.data_prep_config).resolve() != Path(args.config).resolve():
        raise ValidationSetError(
            f"{args.validation} is for {validation_config.data_prep_config}, not {args.config}"
        )
    carried = []
    for entry in validation_config.carry_over:
        other, digest, other_prepared = _other_preparation(entry.data_prep_config, config, corpus, passages)
        carried.append(CarriedSource(entry, other_prepared, other, digest))
    built = build_validation_set(
        validation_config,
        config_path=args.validation,
        config_sha256=validation_sha256,
        prepared=prepared,
        prep_config=config,
        prep_config_sha256=config_sha256,
        passages=passages,
        evaluation_questions=questions,
        carried=carried,
    )
    written, directory = write_validation_set(prepared, built)
    stats = built.manifest["stats"]
    say(f"{'wrote' if written else 'unchanged (identical copy present)'} {directory}")
    for entry in validation_config.carry_over:
        rows = [r for r in built.log if r.get("carried_from") == entry.version]
        say(f"carried {entry.version}: {sum(r['included'] for r in rows)} of {len(rows)} questions included")
        for r in rows:
            say(f"  {r['pair_id']}: {'included' if r['included'] else 'excluded: ' + ', '.join(r['reasons'])}")
    if validation_config.plan is not None:
        planned = sum(r["included"] for r in built.log if "carried_from" not in r)
        prefix = "" if validation_config.plan_positions is None else f" (its first {validation_config.plan_positions})"
        say(f"plan {validation_config.plan}: {stats['plan_passages']} passages{prefix}, {stats['accepted_by_generator']} "
            f"questions accepted by the generator, {planned} included; largest company share "
            f"{stats['largest_company_share']}")
    else:
        say(f"{stats['included']} questions included; largest company share {stats['largest_company_share']}")
    say(f"excluded by reason (a question can have several): {stats['excluded_by_reason'] or 'none'}")
    say(f"by company: {stats['included_by_ticker']}")
    say(f"by fiscal year: {stats['included_by_fiscal_year']}")
    say(f"by question type: {stats['included_by_question_type']}")
    say(f"by passage kind: {stats['included_by_passage_kind']}")
    say("validation questions accepted by an AI quality review: not human-validated, not gold labels, never "
        "training or evaluation data")
    return 0


def _spread(plan: dict[str, Any]) -> str:
    def join(counts: dict[str, int]) -> str:
        return ", ".join(f"{k} {v}" for k, v in counts.items())

    return f"companies: {join(plan['by_ticker'])}; fiscal years: {join(plan['by_fiscal_year'])}"


def _plan(
    args: argparse.Namespace,
    config: Any,
    prepared: Any,
    passages: Any,
    *,
    corpus: Any,
    say: Callable[[str], None],
) -> int:
    approved = prepared.source_passages(args.split, passages, min_words=config.min_source_words)
    filtered = args.command == "sources" or not args.no_source_filter
    quality_settings = config.source_quality
    if getattr(args, "administrative_rule", False):
        quality_settings = quality_settings.model_copy(update={"administrative_rules": True})
    quality = (
        filter_sources(approved, quality_settings, filings_per_group(prepared.rows))
        if filtered
        else None
    )
    if quality is not None:
        stats = quality.stats()
        say(
            f"{args.split}: {stats['approved']} approved passages; source quality keeps {stats['kept']}, "
            f"excludes {stats['excluded']}"
        )
        say(f"excluded by rule (first match): {stats['excluded_by_rule']}")
        say(f"matched by rule (any match): {stats['matched_by_rule']}")
        say(f"excluded by section: {stats['excluded_by_item']}")
        say(f"kept by section: {stats['kept_by_item']}")
        say(f"kept by company: {stats['kept_by_ticker']}")
    if args.command == "sources":
        return 0
    balanced_only = " and ".join(BALANCED_METHODS)
    if (args.exclude_carryover or args.exclude_evidence_of or args.exclude_validation_set) and (
        args.method not in BALANCED_METHODS
    ):
        raise PlanError(f"exclusions apply to the {balanced_only} methods only")
    options = (args.min_isolated_share, args.max_per_company_year, args.max_per_unit, args.predictor)
    flags = (args.one_per != "unit", args.score_priority, args.after_generated is not None,
             bool(args.exclude_equivalents_of), bool(args.exclude_candidate_groups))
    if (any(flags) or any(o is not None for o in options)) and args.method not in BALANCED_METHODS:
        raise PlanError(
            "--one-per, --min-isolated-share, --max-per-company-year, --max-per-unit, --predictor, --score-priority, "
            f"--after-generated, --exclude-equivalents-of and --exclude-candidate-groups apply to the {balanced_only} "
            "methods only"
        )
    if args.predictor is not None and args.min_isolated_share is not None:
        raise PlanError("choose --predictor or --min-isolated-share, not both")
    if args.min_predicted_pass is not None and args.predictor is None:
        raise PlanError("--min-predicted-pass needs --predictor")
    if args.score_priority and args.predictor is None and args.min_isolated_share is None:
        raise PlanError("--score-priority needs scores: --predictor or --min-isolated-share")
    unit_of = prepared.split_units()
    excluded_units = set()
    if args.exclude_evidence_of:
        locator = evidence_locator(prepared, passages)
        for name in args.exclude_evidence_of:
            for candidate_split in CANDIDATE_FILES:
                for pair in load_candidates(prepared, passages, config, name, candidate_split):
                    for e in locator.locate(pair.evidence, pair.answer, pair.ticker, pair.positive_passage_id):
                        if prepared.rows[e.passage_id]["split"] == args.split:
                            excluded_units.add(unit_of[e.passage_id])
    for spec in args.exclude_validation_set:
        path, _, version = spec.rpartition(":")
        if not path or not version:
            raise PlanError(f"--exclude-validation-set takes CONFIG:VERSION, not {spec!r}")
        _, _, other = _other_preparation(path, config, corpus, passages)
        read_validation_manifest(other, version)  # its files and inputs are those recorded
        directory = validation_dir(other, version)
        held = {r["positive_passage_id"] for r in map(json.loads, (directory / VALIDATION_LOG_FILE).open())}
        for line in (directory / "validation_pairs.jsonl").read_text(encoding="utf-8").splitlines():
            pair = json.loads(line)
            held.update([*pair["equivalents"]["exact"], *pair["equivalents"]["uncertain"]])
        excluded_units.update(unit_of[pid] for pid in held)
    for name in args.exclude_carryover:
        path = prepared.directory / CARRYOVER_DIRNAME / f"{name}.jsonl"
        if not path.is_file():
            raise PlanError(f"no carry-over {name!r} at {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["eligible"]:
                excluded_units.add(unit_of[row["positive_passage_id"]])
    exclude_groups = {
        row["group_id"] for pid, row in prepared.rows.items() if unit_of[pid] in excluded_units
    }
    prior_units: Counter[str] = Counter()
    prior_company_years: Counter[tuple[str, str]] = Counter()
    generated_from = []
    if args.after_generated is not None:
        sources = [(args.config, prepared)] + [
            (path, _other_preparation(path, config, corpus, passages)[2]) for path in args.after_generated
        ]
        for path, source in sources:
            records = records_path(source.directory / GENERATION_DIRNAME, args.split)
            done = sorted({r["passage_id"] for r in read_records(records)}) if records.is_file() else []
            for pid in done:  # the same corpus, so every passage has a row and a unit here
                exclude_groups.add(prepared.rows[pid]["group_id"])
                prior_units[unit_of[pid]] += 1
                prior_company_years[(prepared.rows[pid]["ticker"], prepared.rows[pid]["fiscal_year"])] += 1
            generated_from.append(f"generated:{args.split}:{path}:{len(done)} passages")
    for spec in args.exclude_equivalents_of:
        path, _, version = spec.rpartition(":")
        if not path or not version:
            raise PlanError(f"--exclude-equivalents-of takes CONFIG:VERSION, not {spec!r}")
        _, _, other = _other_preparation(path, config, corpus, passages)
        read_validation_manifest(other, version)
        lines = (validation_dir(other, version) / "validation_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        for pair in map(json.loads, lines):
            for pid in (pair["positive_passage_id"], *pair["equivalents"]["exact"], *pair["equivalents"]["uncertain"]):
                exclude_groups.add(prepared.rows[pid]["group_id"])
        generated_from.append(f"equivalents:{spec}")
    for version in args.exclude_candidate_groups:
        used = 0
        for candidate_split in CANDIDATE_FILES:
            for pair in load_candidates(prepared, passages, config, version, candidate_split):  # re-verified
                for pid in (pair.positive_passage_id, *pair.equivalents.exact, *pair.equivalents.uncertain):
                    exclude_groups.add(prepared.rows[pid]["group_id"])
                used += 1
        generated_from.append(f"candidate_groups:{version}:{used} pairs")
    scores, score_method, calibration = None, None, None
    pool = quality.kept if quality is not None else approved
    if args.min_isolated_share is not None:
        scores = isolated_evidence_share(pool, evidence_locator(prepared, passages), prepared.rows)
        score_method = ISOLATION_METHOD
    if args.predictor is not None:
        calibration = load_calibration(args.predictor)
        locator = evidence_locator(prepared, passages)
        shares = isolated_evidence_share(pool, locator, prepared.rows)
        predicted = predict(pool, locator, prepared.rows, calibration, shares)
        scores = {pid: row["probability"] for pid, row in predicted.items()}
        score_method = calibration.method()
    plan = make_plan(
        name=args.name,
        split=args.split,
        method=args.method,
        seed=config.split.seed if args.seed is None else args.seed,
        size=args.size,
        approved=approved,
        quality=quality,
        quality_settings=quality_settings if quality is not None else None,
        prepared_manifest_sha256=prepared.manifest_sha256,
        unit_of=unit_of,
        max_company_share=args.max_company_share,
        exclude_groups=exclude_groups,
        excluded_from=[
            *(f"carryover:{n}" for n in args.exclude_carryover),
            *(f"candidates:{n}" for n in args.exclude_evidence_of),
            *(f"validation_set:{n}" for n in args.exclude_validation_set),
            *generated_from,
        ],
        one_per=args.one_per,
        scores=scores,
        min_score=args.min_predicted_pass if args.predictor is not None else args.min_isolated_share,
        score_method=score_method,
        max_per_company_year=args.max_per_company_year,
        max_per_unit=args.max_per_unit,
        prior_units=prior_units,
        prior_company_years=prior_company_years,
        score_priority=args.score_priority,
    )
    written = write_plan(prepared.directory, plan)
    path = plan_path(prepared.directory, args.split, args.name)
    say(
        f"{'wrote' if written else 'unchanged (identical plan present)'} plan {args.name} at {path}"
    )
    say(f"{plan['size']} passages ({plan['method']}, seed {plan['seed']}); {_spread(plan)}")
    say(f"sections: {plan['by_item']}")
    if "constraints" in plan:
        c = plan["constraints"]
        say(
            f"kinds: {plan['by_kind']}; section families: {plan['by_section_family']}; at most "
            f"{c['max_company_share']:.0%} per company, one passage per overlap group"
            f"{' and split unit' if c['one_per_split_unit'] else ''}, "
            + (f"at most {c['max_per_company_year']} per company and fiscal year, " if "max_per_company_year" in c else "")
            + (f"at most {c['max_per_split_unit']} per split unit, " if "max_per_split_unit" in c else "")
            + f"{c['excluded_groups']} overlap groups excluded {c['excluded_from'] or ''}".rstrip()
        )
        units = Counter(p["split_unit"] for p in plan["passages"])
        company_years = Counter((p["ticker"], p["fiscal_year"]) for p in plan["passages"])
        say(
            f"concentration: {len(units)} split units (at most {max(units.values())} passages from one), "
            f"{len(company_years)} company-years (at most {max(company_years.values())} from one)"
        )
    if "coverage" in plan:
        say(
            "fiscal years covered of those available: "
            + ", ".join(f"{t} {len(c['covered'])}/{len(c['available'])}" for t, c in plan["coverage"].items())
        )
    if calibration is not None:
        probabilities = [p["score"] for p in plan["passages"]]
        expected = sum(calibration.usable(p) for p in probabilities)
        below = plan["constraints"]["scores"]["below_min_score"]
        say(f"predicted evidence pass: mean {sum(probabilities) / len(probabilities):.3f} over the plan"
            + (f", {below} eligible passages below {args.min_predicted_pass}" if args.min_predicted_pass is not None else ""))
        say(f"estimated usable questions: about {expected:.0f} (predicted pass times generator acceptance "
            f"{calibration.generator_acceptance:.2f} and review acceptance {calibration.review_acceptance:.2f} of the "
            "calibration; an estimate, not a measurement)")
    elif scores is not None:
        shares = [p["score"] for p in plan["passages"]]
        expected = sum(shares) * PILOT_GENERATOR_ACCEPTANCE * PILOT_REVIEW_ACCEPTANCE
        say(f"isolated-evidence share: mean {sum(shares) / len(shares):.3f} over the plan, "
            f"{plan['constraints']['scores']['below_min_score']} eligible passages below {args.min_isolated_share}")
        say(f"estimated usable questions: about {expected:.0f} (the plan's shares times val_pilot_v1's generator and "
            "review acceptance; an estimate, not a measurement)")
    return 0


def _revalidate(
    args: argparse.Namespace,
    config: Any,
    prepared: Any,
    passages: Any,
    *,
    say: Callable[[str], None],
) -> int:
    sources = prepared.source_passages(args.split, passages, min_words=config.min_source_words)
    counts = revalidate_records(
        prepared.directory / GENERATION_DIRNAME,
        args.split,
        {p.passage_id: p for p in sources},
        config.generation,
        aliases_for(passages, config.generation.company_aliases),
        prepared_manifest_sha256=prepared.manifest_sha256,
        dry_run=args.dry_run,
    )
    unrecoverable = counts.pop("not_revalidatable")
    say(f"{args.split} records against validator {VALIDATOR_VERSION}: {counts}")
    if counts["revalidated"]:
        say(
            "dry run: nothing written"
            if args.dry_run
            else f"appended {counts['revalidated']} revision records; earlier records are unchanged"
        )
    for pid in unrecoverable:
        say(
            f"cannot revalidate {pid}: its record predates stored candidates (answers and evidence were "
            "not kept); only a new request could recover it"
        )
    return 0


def _stop_plan(
    args: argparse.Namespace, config: Any, prepared: Any, passages: Any, *, say: Callable[[str], None]
) -> int:
    sources = prepared.source_passages(args.split, passages, min_words=config.min_source_words)
    plan, planned = load_plan(
        prepared.directory,
        args.split,
        args.plan,
        {p.passage_id: p for p in sources},
        prepared_manifest_sha256=prepared.manifest_sha256,
    )
    records_file = records_path(prepared.directory / GENERATION_DIRNAME, args.split)
    lines, digest = records_prefix(records_file)
    record = make_stop_record(
        plan,
        planned,
        latest_records(read_records(records_file)),
        plan_sha256=sha256_file(plan_path(prepared.directory, args.split, args.plan)),
        records={"path": str(records_file), "lines": lines, "sha256": digest},
        reason=args.reason,
        stopped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    written = write_stop_record(prepared.directory, record)
    path = stop_path(prepared.directory, args.split, args.plan)
    say(f"{'wrote' if written else 'unchanged (already stopped there)'} {path}")
    say(
        f"plan {args.plan}: stopped after position {record['position']} of {record['passages']} "
        f"(records {record['statuses']}); {record['records']['lines']} record lines bound"
    )
    if record["later_positions_with_records"]:
        say(f"later positions with a record from another plan: {record['later_positions_with_records']}")
    say("generate --plan now refuses this plan; no request is sent for it")
    return 0


def _generate(
    args: argparse.Namespace,
    config: Any,
    prepared: Any,
    passages: Any,
    *,
    transport: Transport | None,
    sleep: Callable[[float], None],
    say: Callable[[str], None],
) -> int:
    settings = config.generation
    sources = prepared.source_passages(args.split, passages, min_words=config.min_source_words)
    if args.plan is not None and args.sample is not None:
        raise PlanError("use either --plan or --sample, not both")
    if args.plan is not None and (stop := read_stop_record(prepared.directory, args.split, args.plan)) is not None:
        raise PlanError(
            f"plan {args.plan} was stopped at position {stop['position']} of {stop['passages']} on "
            f"{stop['stopped_at']} ({stop['reason']}); generation under it is closed and no request is sent"
        )
    if args.plan is not None:
        plan, sources = load_plan(
            prepared.directory,
            args.split,
            args.plan,
            {p.passage_id: p for p in sources},
            prepared_manifest_sha256=prepared.manifest_sha256,
        )
        say(f"plan {args.plan}: {len(sources)} passages; {_spread(plan)}")
    elif args.sample is None:
        quality = filter_sources(sources, config.source_quality, filings_per_group(prepared.rows))
        if quality.excluded:
            say(
                f"source quality: {len(quality.excluded)} of {len(sources)} approved passages excluded "
                f"{quality.stats()['excluded_by_rule']}"
            )
        sources = quality.kept
    if args.sample is not None:
        seed = config.split.seed if args.sample_seed is None else args.sample_seed
        sources = sample_sources(sources, args.sample, seed)
        companies = ", ".join(
            f"{t} {n}" for t, n in sorted(Counter(p.ticker for p in sources).items())
        )
        say(f"sample of {len(sources)} passages (seed {seed}): {companies}")
    directory = prepared.directory / GENERATION_DIRNAME
    run = check_run(
        directory, args.split, settings, prepared.manifest_sha256
    )  # a different run is refused
    state = latest_records(read_records(records_path(directory, args.split)))
    done = sum(1 for p in sources if state.get(p.passage_id, {}).get("status") == "done")
    failed = sum(1 for p in sources if state.get(p.passage_id, {}).get("status") == "failed")
    todo = len(sources) - done - (0 if args.retry_failed else failed)
    if args.limit is not None:
        todo = min(todo, args.limit)
    per_minute, otpm = settings.requests_per_minute, settings.output_tokens_per_minute
    if otpm is not None:  # worst case: every reply uses its whole max_completion_tokens
        per_minute = min(per_minute, max(1, otpm // settings.max_completion_tokens))
    minutes = todo / per_minute
    worst = todo * (settings.max_retries + 1) * settings.max_response_attempts
    budget = "none" if args.max_requests is None else str(args.max_requests)
    say(
        f"{args.split}: {len(sources)} approved passages, {done} done, {failed} failed; {todo} to request from "
        f"{settings.provider} ({settings.model}), about {minutes:.0f} min at {per_minute:g} requests/min"
    )
    say(
        f"requests: {todo} if every reply is usable, at most {worst} with every retry; request budget {budget}"
    )
    say(
        f"request settings: temperature {settings.temperature:g}, max_completion_tokens "
        f"{settings.max_completion_tokens}, json_mode {settings.json_mode}, reasoning_format "
        f"{settings.reasoning_format or 'not sent'}, reasoning_effort {settings.reasoning_effort or 'not sent'}, "
        f"prompt {settings.prompt_version} ({prompt_sha256(settings.prompt_version)[:12]})"
    )
    if otpm is not None:
        say(
            f"pacing: {settings.requests_per_minute:g} requests/min and {otpm} output tokens/min, each request "
            f"reserving {settings.max_completion_tokens}; {per_minute:g}/min if every reply used it all, more with "
            f"shorter replies; rate-limit waits up to {settings.max_rate_limit_wait_seconds:g}s"
        )
    if run.state == "continuable":
        say(f"request settings differ from the recorded run: {describe_changes(run.changed)}")
        say(
            f"--continue-run records continuation {len(run.bindings)} in "
            f"{continuations_path(directory, args.split)} before the first request; every existing record is kept"
            if args.continue_run
            else "add --continue-run to record a continuation and resume this run; without it the run is refused"
        )
    elif len(run.bindings) > 1:
        say(
            f"run continued {len(run.bindings) - 1} time(s); the latest continuation matches these settings"
        )
    if not args.confirm_api_use:
        say("dry run: no request sent; add --confirm-api-use to call the API")
        return 0
    key = api_key_from_env(settings)
    logging.basicConfig(format=f"{PREFIX} %(message)s", stream=sys.stderr)
    client = GroqClient(
        settings, api_key=key, transport=transport, sleep=sleep, max_requests=args.max_requests
    )
    generator = QueryGenerator(
        client,
        settings,
        aliases_for(passages, settings.company_aliases),
        directory=directory,
        prepared_manifest_sha256=prepared.manifest_sha256,
        secret=key,
        continue_run=args.continue_run,
    )
    try:
        counts = generator.run(
            args.split, sources, limit=args.limit, retry_failed=args.retry_failed
        )
    finally:
        if generator.continuation is not None:
            c = generator.continuation
            say(f"recorded continuation {c['continuation']}: {describe_changes(c['changed'])}")
        tokens = dict(sorted(client.usage.items())) or "not reported"
        say(f"requests sent: {client.requests_sent}; tokens: {tokens}")
        if client.rate_limited or client.waited_seconds:
            say(
                f"HTTP 429 responses: {client.rate_limited}; waited {client.waited_seconds:.0f}s for pacing and "
                "rate limits"
            )
        if client.rate_limit:
            say(f"rate limits after the last response: {dict(sorted(client.rate_limit.items()))}")
    say(f"{args.split}: {counts}")
    if counts["stopped_by_budget"]:
        say("stopped at the request budget; rerun to resume (completed passages are skipped)")
    if counts["stopped_by_rate_limit"]:
        say(
            f"stopped by a rate limit ({client.stop_reason}); rerun later to resume (completed passages are skipped)"
        )
    return 0


__all__ = ["build_parser", "main"]
