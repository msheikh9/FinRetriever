"""Terminal interface for writing questions, pooling candidates and judging them.

``main(argv)`` backs ``scripts/03_annotate_gold_set.py``. Every subcommand
first checks the corpus against its manifest and against the frozen SHA-256
and version in the gold-set config, and opening a workspace checks it was
made for that corpus. A typical round::

    init --annotator XY
    add-question --text "..." --topic debt_liquidity --type semantic --authoring independent --ticker JPM
    pool q_0001 --reformulation "revolving credit facility borrowing capacity"
    annotate q_0001        # interactive; every command is saved immediately
    status
    export

Development labels (AI-generated, never gold) have their own commands:
``dev-labels`` and ``export-dev`` derive and export a label set,
``expand-pool`` deepens a label set's pool with the top results of stored runs,
and ``audit-init-pool`` draws a blind priority audit of an expanded pool's
proposals (graded with ``audit``, compared with ``audit-report``).

Exit codes: 0 done; 1 the action was refused (nothing changed); 2 config,
corpus or workspace binding error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic import ValidationError

from finretriever.data.schemas import Passage, Query
from finretriever.eval_set.ai_proposals import ProposalSet
from finretriever.eval_set.annotation_store import (
    QUESTIONS_FILE,
    AnnotationError,
    AnnotationStore,
    Judgment,
    load_corpus_passages,
    utc_now,
    verify_corpus,
)
from finretriever.eval_set.audit import (
    AUDIT_FILE,
    REPORT_FILE,
    AuditError,
    AuditWorkspace,
    audit_agreement,
    create_audit,
)
from finretriever.eval_set.candidate_generator import CandidateGenerator, Scope, build_pool
from finretriever.eval_set.dev_export import build_dev_export, write_dev_export
from finretriever.eval_set.dev_labels import (
    DevLabelError,
    DevLabelSet,
    derive_dev_labels,
    load_dev_label_spec,
)
from finretriever.eval_set.dev_pool import (
    POOL_INFO_FILE,
    DevPool,
    derive_expanded_labels,
    expand_pool,
    is_expanded_label_spec,
    load_dev_pool_spec,
    load_expanded_label_spec,
    open_pool_proposals,
    read_annotation_pool,
)
from finretriever.eval_set.pool_audit import create_pool_audit, load_pool_audit_spec
from finretriever.eval_set.qrels_builder import (
    GoldExportError,
    build_gold_export,
    find_gold_overlap,
    judgment_coverage,
    write_gold_export,
)
from finretriever.eval_set.question_stage import (
    SHEET_FILE,
    check_poolable,
    export_paths,
    export_questions,
    export_versions,
    import_questions,
    leakage_record,
    load_plan,
    revise_questions,
    slot_of,
    verify_export,
    write_leakage_record,
    write_question_sheet,
    write_revision_sheet,
)
from finretriever.eval_set.regrade import create_regrade, regrade_agreement, write_regrade_report
from finretriever.eval_set.remote_package import (
    build_package,
    import_returned,
    is_withdrawn,
    load_manifest,
    outstanding_questions,
    package_dirs,
    packaged_questions,
    remote_root,
    withdraw_package,
)
from finretriever.eval_set.rewording import (
    apply_rewordings,
    check_rewording_base,
    context_rows,
    pool_snapshot_versions,
    validate_rewording_sheet,
    verify_approved_pool,
    write_pool_snapshot,
    write_rewording_sheet,
)
from finretriever.eval_set.schemas import (
    AUTHORING,
    COMPARISONS,
    QUESTION_TYPES,
    RELEVANCE_GRADES,
    TOPICS,
    CandidateSource,
    GoldQuestion,
    GoldSetConfig,
    PoolEntry,
)
from finretriever.eval_set.text_assignment import (
    CROSSCHECK_RULE,
    SPLIT_DRAWS,
    SPLIT_SEED,
    assigned_questions,
    assignment_dirs,
    build_assignment,
    import_records,
    import_responses,
    load_assignment_manifest,
    load_split,
    outstanding_assigned,
    recorded_questions,
    split_questions,
    write_split,
)
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    CorpusIntegrityError,
    CorpusMismatchError,
)
from finretriever.evaluation.stored_runs import load_stored_run
from finretriever.utils.io import read_yaml

DEFAULT_CONFIG = "configs/eval/gold_set_v1.yaml"
ANNOTATOR_ENV = "FINRETRIEVER_ANNOTATOR"
PREFIX = "[03_annotate_gold_set]"


class GoldSetConfigError(ValueError):
    """The gold-set config is missing or invalid."""


def load_gold_set_config(path: str | Path) -> GoldSetConfig:
    file = Path(path)
    if not file.is_file():
        raise GoldSetConfigError(f"gold-set config not found: {file}")
    try:
        return GoldSetConfig.model_validate(read_yaml(file))
    except (ValidationError, yaml.YAMLError, ValueError) as exc:
        raise GoldSetConfigError(f"invalid gold-set config {file}: {exc}") from None


# --- rendering ----------------------------------------------------------------------


def section_label(passage: Passage) -> str:
    title = passage.section_title or "(untitled section)"
    item = passage.metadata.get("item_number")
    if item:
        return title if title.lower().startswith("item") else f"Item {item} · {title}"
    if passage.metadata.get("section_kind") == "appended_report":
        return f"{title} (appended report section; no Item number)"
    return title


def describe_source(source: CandidateSource, *, hide_scores: bool = False) -> str:
    """One pool source for display; ``hide_scores`` drops its rank and score (they stay stored)."""
    text = source.method
    if not hide_scores:
        if source.rank is not None:
            text += f" #{source.rank}"
        if source.score is not None:
            text += f" (score {source.score:g})"
    return f"{text} — {source.detail}" if source.detail else text


def describe_judgment(judgment: Judgment | None, revisions: int = 0) -> str:
    if judgment is None:
        return "unjudged (pending)"
    who = f"{judgment.annotator}, {judgment.at}" + (f", {revisions} revisions" if revisions > 1 else "")
    note = f" — {judgment.note}" if judgment.note else ""
    if judgment.grade is None:
        return f"skipped: unjudged, not a 0 ({who}){note}"
    return f"grade {judgment.grade} · {RELEVANCE_GRADES[judgment.grade]} ({who}){note}"


def filing_line(passage: Passage) -> str:
    company = passage.metadata.get("company_name") or passage.ticker
    accession = passage.metadata.get("accession_number") or "?"
    return f"{company} ({passage.ticker}) · {passage.form_type} filed {passage.filing_date} · accession {accession}"


def passage_lines(passage: Passage, *, full: bool, preview_chars: int, width: int) -> list[str]:
    """The passage text wrapped to ``width``; cut near ``preview_chars`` unless ``full``."""
    text = passage.text
    truncated = not full and len(text) > preview_chars
    if truncated:
        text = text[:preview_chars].rsplit(" ", 1)[0]
    lines = [textwrap.fill(line, width) if line else "" for line in text.split("\n")]
    if truncated:
        lines.append(f"... {len(passage.text) - len(text)} more characters (f shows the full passage)")
    return lines


def question_scope(question: GoldQuestion) -> str:
    parts = (
        f"{question.topic} / {question.question_type}",
        question.comparison if question.comparison != "none" else "",
        ",".join(question.tickers),
        ",".join(map(str, question.fiscal_years)),
    )
    return " · ".join(part for part in parts if part)


# Blind grading (``annotate --blind``): the display order of a question's candidates is shuffled with
# random.Random(f"{BLIND_ORDER_SEED}:{query_id}"), the same order in every session, and no pool source is shown.
BLIND_ORDER_SEED = "gold-blind-v1"

_SESSION_HELP = """\
  0 / 1 / 2   grade the shown passage: 0 not relevant, 1 partially, 2 directly relevant
  s [note]    skip: leave it unjudged (uncertain) - this is not a 0
  c           clear its grade or skip (back to pending)
  n / p       next / previous candidate       g N   go to candidate N
  u           next pending candidate          f     full passage text
  h           judgment history of this passage
  done        mark the question complete and stop
  q           stop (everything is already saved)
  A note can follow a grade too, e.g. "1 only covers 2023"."""


class AnnotationSession:
    """Interactive judging of one question's candidate pool.

    ``hide_scores`` hides each source's rank and score. ``blind`` goes further: no pool source is shown at all (not
    its method, rank, score or search), and the candidates appear in an order shuffled per question
    (``BLIND_ORDER_SEED``), so neither the source nor the position in a retriever's ranking can steer a grade. Both
    change the display only: the pool, its order on disk and every stored provenance field stay as they are.
    """

    def __init__(
        self,
        store: AnnotationStore,
        query_id: str,
        annotator: str,
        *,
        input_fn: Callable[[str], str] = input,
        out: TextIO | None = None,
        preview_chars: int = 1200,
        width: int = 100,
        hide_scores: bool = False,
        blind: bool = False,
    ) -> None:
        self.store = store
        self.query_id = query_id
        self.annotator = annotator
        self.input_fn = input_fn
        self.out = out or sys.stdout
        self.preview_chars = preview_chars
        self.width = width
        self.blind = blind
        self.hide_scores = hide_scores or blind
        self.entries: list[PoolEntry] = list(store.pool(query_id))
        if blind:
            random.Random(f"{BLIND_ORDER_SEED}:{query_id}").shuffle(self.entries)
        self.index = 0
        self.changes = 0

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def _next_pending(self, start: int) -> int | None:
        order = list(range(start, len(self.entries))) + list(range(start))
        return next((i for i in order if self.store.judgment(self.query_id, self.entries[i].passage_id) is None), None)

    def _progress(self) -> str:
        judgments = self.store.judgments(self.query_id)
        skipped = sum(j.grade is None for j in judgments)
        return (
            f"graded {len(judgments) - skipped} · skipped {skipped} · "
            f"pending {len(self.entries) - len(judgments)} of {len(self.entries)}"
        )

    def run(self) -> int:
        """Judge until ``q``, ``done`` or end of input; returns the number of changes."""
        question = self.store.question(self.query_id)
        if question.status != "open":
            raise AnnotationError(f"{self.query_id} is {question.status}; run `reopen {self.query_id}` to change it")
        if not self.entries:
            raise AnnotationError(f"{self.query_id} has no candidates yet; run `pool {self.query_id}` first")
        start = self._next_pending(0)
        self.index = 0 if start is None else start
        hidden = (" · blind: pool sources, ranks and scores hidden; order shuffled" if self.blind
                  else " · retrieval scores and ranks hidden" if self.hide_scores else "")
        self.say(f"{self.query_id}: {self._progress()}{hidden} — type ? for commands")
        self._show()
        while True:
            try:
                raw = self.input_fn("judge> ")
            except EOFError:
                self.say()
                break
            command, _, arg = raw.strip().partition(" ")
            if command.lower() == "q":
                break
            try:
                if self._handle(command.lower(), arg.strip()):
                    break
            except AnnotationError as exc:
                self.say(f"! {exc}")
        self.say(f"saved {self.query_id}: {self._progress()} ({self.changes} changes this session)")
        return self.changes

    def _state(self, entry: PoolEntry) -> str | None:
        judgment = self.store.judgment(self.query_id, entry.passage_id)
        return judgment.state if judgment else None

    def _handle(self, command: str, arg: str) -> bool:
        entry = self.entries[self.index]
        before = self._state(entry)
        if command in ("0", "1", "2", "s"):
            if command == "s":
                self.store.skip(self.query_id, entry.passage_id, annotator=self.annotator, note=arg)
            else:
                self.store.judge(self.query_id, entry.passage_id, int(command), annotator=self.annotator, note=arg)
            self.changes += self._state(entry) != before
            self._advance()
        elif command == "c":
            if before is None:
                self.say("nothing to clear: this passage is already pending")
            else:
                self.store.clear(self.query_id, entry.passage_id, annotator=self.annotator, note=arg)
                self.changes += 1
                self._show()
        elif command in ("n", "p"):
            self.index = max(0, min(len(self.entries) - 1, self.index + (1 if command == "n" else -1)))
            self._show()
        elif command == "g":
            if not arg.isdigit() or not 1 <= int(arg) <= len(self.entries):
                self.say(f"! usage: g N with N from 1 to {len(self.entries)}")
            else:
                self.index = int(arg) - 1
                self._show()
        elif command == "u":
            pending = self._next_pending(self.index + 1)
            if pending is None:
                self.say("no pending candidates")
            else:
                self.index = pending
                self._show()
        elif command == "f":
            self._show(full=True)
        elif command == "h":
            self._show_history(entry)
        elif command == "done":
            self.store.set_status(self.query_id, "complete", annotator=self.annotator, note=arg)
            self.say(f"{self.query_id} marked complete")
            return True
        elif command in ("?", "help", ""):
            self.say(_SESSION_HELP)
        else:
            self.say(f"! unknown command {command!r}; type ? for commands")
        return False

    def _advance(self) -> None:
        pending = self._next_pending(self.index + 1)
        if pending is None:
            self.say(
                f"all {len(self.entries)} candidates are graded or skipped — type done to mark "
                f"{self.query_id} complete, or q to stop"
            )
            return
        self.index = pending
        self._show()

    def _show_history(self, entry: PoolEntry) -> None:
        events = self.store.history(self.query_id, entry.passage_id)
        if not events:
            self.say("no judgments recorded for this passage")
        for event in events:
            new = {"grade": str(event.grade), "skip": "skipped", "clear": "pending"}[event.kind]
            note = f" — {event.note}" if event.note else ""
            self.say(f"  #{event.seq} {event.at} {event.annotator}: {event.previous or 'pending'} -> {new}{note}")

    def _show(self, *, full: bool = False) -> None:
        question = self.store.question(self.query_id)
        entry = self.entries[self.index]
        passage = self.store.passages[entry.passage_id]
        revisions = sum(e.kind != "clear" for e in self.store.history(self.query_id, entry.passage_id))
        rule = "-" * self.width
        self.say(rule)
        self.say(f"{question.query_id} · {question_scope(question)}")
        self.say(f"Q: {question.text}")
        self.say(f"candidate {self.index + 1}/{len(self.entries)} · {self._progress()}")
        self.say(f"passage   {passage.passage_id}")
        self.say(f"filing    {filing_line(passage)}")
        self.say(f"section   {section_label(passage)}")
        if self.blind:
            self.say("source    hidden (blind grading)")
        else:
            for source in entry.sources:
                self.say(f"source    {describe_source(source, hide_scores=self.hide_scores)}")
        self.say(f"current   {describe_judgment(self.store.judgment(self.query_id, entry.passage_id), revisions)}")
        self.say(rule)
        for line in passage_lines(passage, full=full, preview_chars=self.preview_chars, width=self.width):
            self.say(line)


_AUDIT_HELP = """\
  0 / 1 / 2   grade the shown passage: 0 not relevant, 1 partially, 2 directly relevant
  s [note]    skip: leave it unjudged (uncertain) - this is not a 0
  c           clear its grade or skip (back to pending)
  n / p       next / previous item            g N   go to item N
  u           next pending item               f     full passage text
  q           stop (everything is already saved)
  A note can follow a grade, e.g. "1 only the 2022 figure". AI labels stay hidden until every
  item is graded or skipped; then run audit-report."""


class AuditSession:
    """Blind grading of an audit sample, across questions, in the audit's fixed order.

    Shows the question, the filing and the passage only: no retrieval source,
    rank, score or AI label is ever loaded.
    """

    def __init__(
        self,
        audit: AuditWorkspace,
        annotator: str,
        *,
        input_fn: Callable[[str], str] = input,
        out: TextIO | None = None,
        preview_chars: int = 1200,
        width: int = 100,
    ) -> None:
        self.audit = audit
        self.store = audit.store
        self.annotator = annotator
        self.input_fn = input_fn
        self.out = out or sys.stdout
        self.preview_chars = preview_chars
        self.width = width
        self.items = audit.items()
        self.index = 0
        self.changes = 0

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def _state(self, index: int) -> str | None:
        judgment = self.audit.judgment(self.items[index])
        return judgment.state if judgment else None

    def _next_pending(self, start: int) -> int | None:
        order = list(range(start, len(self.items))) + list(range(start))
        return next((i for i in order if self._state(i) is None), None)

    def _progress(self) -> str:
        p = self.audit.progress()
        return f"graded {p['graded']} · skipped {p['skipped']} · pending {p['pending']} of {p['items']}"

    def run(self) -> int:
        start = self._next_pending(0)
        self.index = 0 if start is None else start
        self.say(f"blind audit: {self._progress()} · AI labels hidden until the audit is complete — type ? for commands")
        if start is None:
            self.say("every item is graded or skipped; corrections are still possible, or run audit-report")
        self._show()
        while True:
            try:
                raw = self.input_fn("audit> ")
            except EOFError:
                self.say()
                break
            command, _, arg = raw.strip().partition(" ")
            if command.lower() == "q":
                break
            try:
                self._handle(command.lower(), arg.strip())
            except AnnotationError as exc:
                self.say(f"! {exc}")
        self.say(f"saved audit: {self._progress()} ({self.changes} changes this session)")
        return self.changes

    def _handle(self, command: str, arg: str) -> None:
        item = self.items[self.index]
        before = self._state(self.index)
        if command in ("0", "1", "2", "s"):
            if command == "s":
                self.store.skip(item.query_id, item.passage_id, annotator=self.annotator, note=arg)
            else:
                self.store.judge(item.query_id, item.passage_id, int(command), annotator=self.annotator, note=arg)
            self.changes += self._state(self.index) != before
            pending = self._next_pending(self.index + 1)
            if pending is None:
                self.say(
                    f"all {len(self.items)} audit items are graded or skipped — run audit-report to compare "
                    "with the AI proposals (q to stop)"
                )
            else:
                self.index = pending
                self._show()
        elif command == "c":
            if before is None:
                self.say("nothing to clear: this item is already pending")
            else:
                self.store.clear(item.query_id, item.passage_id, annotator=self.annotator, note=arg)
                self.changes += 1
                self._show()
        elif command in ("n", "p"):
            self.index = max(0, min(len(self.items) - 1, self.index + (1 if command == "n" else -1)))
            self._show()
        elif command == "g":
            if not arg.isdigit() or not 1 <= int(arg) <= len(self.items):
                self.say(f"! usage: g N with N from 1 to {len(self.items)}")
            else:
                self.index = int(arg) - 1
                self._show()
        elif command == "u":
            pending = self._next_pending(self.index + 1)
            if pending is None:
                self.say("no pending items")
            else:
                self.index = pending
                self._show()
        elif command == "f":
            self._show(full=True)
        elif command in ("?", "help", ""):
            self.say(_AUDIT_HELP)
        else:
            self.say(f"! unknown command {command!r}; type ? for commands")

    def _show(self, *, full: bool = False) -> None:
        item = self.items[self.index]
        question = self.store.question(item.query_id)
        passage = self.store.passages[item.passage_id]
        rule = "-" * self.width
        self.say(rule)
        self.say(f"audit item {self.index + 1}/{len(self.items)} · {self._progress()}")
        self.say(f"{question.query_id} · {question_scope(question)}")
        self.say(f"Q: {question.text}")
        self.say(f"passage   {passage.passage_id}")
        self.say(f"filing    {filing_line(passage)}")
        self.say(f"section   {section_label(passage)}")
        self.say(f"current   {describe_judgment(self.audit.judgment(item))}")
        self.say(rule)
        for line in passage_lines(passage, full=full, preview_chars=self.preview_chars, width=self.width):
            self.say(line)


# --- commands -----------------------------------------------------------------------


@dataclass
class _Context:
    args: argparse.Namespace
    config: GoldSetConfig
    corpus: CorpusIdentity
    workspace: Path
    input_fn: Callable[[str], str]
    out: TextIO
    _store: AnnotationStore | None = field(default=None, repr=False)

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def annotator(self) -> str:
        name = (self.args.annotator or os.environ.get(ANNOTATOR_ENV) or "").strip()
        if not name:
            raise AnnotationError(f"pass --annotator (your initials) or set {ANNOTATOR_ENV}")
        return name

    def human_annotator(self) -> str:
        """The annotator, refused if it is one of the configured AI identities."""
        name = self.annotator()
        if name.casefold() in {n.casefold() for n in self.config.ai_annotators}:
            raise AnnotationError(f"{name!r} is an AI identity; relevance judgments must come from a person")
        return name

    def store(self) -> AnnotationStore:
        if self._store is None:
            self._store = AnnotationStore.open(self.workspace, self.corpus)
        return self._store


def _cmd_init(ctx: _Context) -> int:
    store = AnnotationStore.create(
        ctx.workspace, ctx.corpus, annotation_version=ctx.config.annotation_version, created_by=ctx.annotator()
    )
    ctx.say(
        f"{PREFIX} created {ctx.workspace} for {ctx.corpus.corpus_version} "
        f"(sha256 {ctx.corpus.corpus_sha256}, {len(store.passages)} passages); "
        f"annotation version {store.info.evaluation.annotation_version}"
    )
    return 0


def _cmd_add_question(ctx: _Context) -> int:
    args = ctx.args
    question = ctx.store().add_question(
        args.text,
        topic=args.topic,
        question_type=args.question_type,
        comparison=args.comparison,
        tickers=args.ticker,
        fiscal_years=args.year,
        authoring=args.authoring,
        author=ctx.annotator(),
        notes=args.notes,
        ai_assisted=args.ai_assisted,
        reviewed_by=args.reviewed_by,
    )
    ctx.say(f"{PREFIX} added {question.query_id}: {question.text}")
    if question.ai_assisted:
        ctx.say(f"{PREFIX} provenance: AI-assisted, author {question.author}, reviewed by {question.reviewed_by}")
    ctx.say(f"{PREFIX} next: pool {question.query_id}")
    return 0


def _cmd_pool(ctx: _Context) -> int:
    args, config, store = ctx.args, ctx.config, ctx.store()
    question = store.question(args.query_id)
    if config.question_stage_dir:  # a gold v1 slot question is pooled only once the questions are frozen
        check_poolable(question, config.question_stage_dir)
        _refuse_if_pools_approved(ctx, question.query_id)
    if (slot := slot_of(question)) in config.provisional_slots:
        ctx.say(f"{PREFIX} NOTE {question.query_id} ({slot}) is provisional: {config.provisional_slots[slot]}")
    candidates = build_pool(
        CandidateGenerator(store.passages),
        question,
        bm25_depth=args.bm25_depth or config.bm25_depth,
        reformulations=args.reformulation,
        manual_ids=args.passage_id,
        manual_note=args.manual_note,
        stratified_per_filing=config.stratified_per_filing if args.stratified is None else args.stratified,
        random_count=config.random_candidates if args.random is None else args.random,
        seed=config.seed,
        use_years=args.use_years,
    )
    new = store.add_candidates(question.query_id, candidates, added_by=ctx.annotator())
    methods = ", ".join(f"{m} {n}" for m, n in sorted(Counter(c.source.method for c in candidates).items()))
    ctx.say(
        f"{PREFIX} {question.query_id}: {len(candidates)} proposed ({methods or 'none'}), {len(new)} new; "
        f"pool now {len(store.pool(question.query_id))}"
    )
    return 0


def _cmd_search(ctx: _Context) -> int:
    passages = load_corpus_passages(ctx.corpus)
    generator = CandidateGenerator(passages)
    scope = Scope(tickers=frozenset(t.upper() for t in ctx.args.ticker))
    for candidate in generator.bm25(ctx.args.text, depth=ctx.args.top_k, scope=scope):
        passage = passages[candidate.passage_id]
        preview = " ".join(passage.text.split())[:120]
        ctx.say(f"{candidate.source.rank:>3}. {passage.passage_id}")
        ctx.say(f"     {passage.ticker} {passage.filing_date} · {section_label(passage)} · {preview}")
    return 0


def _cmd_show(ctx: _Context) -> int:
    passages = load_corpus_passages(ctx.corpus)
    passage = passages.get(ctx.args.passage_id)
    if passage is None:
        raise AnnotationError(f"passage {ctx.args.passage_id} is not in the corpus")
    ctx.say(f"passage   {passage.passage_id}")
    ctx.say(f"filing    {filing_line(passage)}")
    ctx.say(f"section   {section_label(passage)}")
    ctx.say(passage.text)
    return 0


def _cmd_annotate(ctx: _Context) -> int:
    store = ctx.store()
    annotator = ctx.human_annotator()
    approved = _approved_pool_gate(ctx, store)  # once a pool is approved, grading serves only its questions
    remote = ({**outstanding_questions(_stage_dir(ctx)), **outstanding_assigned(_stage_dir(ctx))}
              if ctx.config.question_stage_dir else {})
    query_id = ctx.args.query_id
    if query_id is None:
        query_id = next((q.query_id for q in store.questions() if q.status == "open" and store.pending(q.query_id)
                         and (approved is None or q.query_id in approved[0]) and q.query_id not in remote), None)
        if query_id is None:
            ctx.say(f"{PREFIX} no open question has pending candidates")
            return 0
    if approved is not None:
        if query_id not in approved[0]:
            raise AnnotationError(f"{query_id} is not one of the {len(approved[0])} questions of the approved pool "
                                  "snapshot; this grading stage covers only those")
        if query_id in remote:
            kind, how = (("package", "remote-import") if remote[query_id].startswith("p")
                         else ("assignment", "assign-import"))
            raise AnnotationError(f"{query_id} is out for remote grading in {kind} {remote[query_id]}; its grades "
                                  f"come back through {how}")
        registered = ctx.config.registered_annotators
        if registered and annotator not in registered:
            raise AnnotationError(f"{annotator!r} is not a registered primary annotator of the approved questions "
                                  f"({', '.join(registered)}); a re-grade is graded in its own workspace")
        ctx.say(f"{PREFIX} grading pool snapshot version {approved[1]['version']} "
                f"({approved[1]['snapshot']['sha256']}): verified, unchanged")
    session = AnnotationSession(
        store,
        query_id,
        annotator,
        input_fn=ctx.input_fn,
        out=ctx.out,
        preview_chars=ctx.args.preview_chars,
        hide_scores=ctx.args.hide_scores,
        blind=ctx.args.blind or ctx.config.require_blind,
    )
    session.run()
    return 0


def _cmd_status_change(status: str) -> Callable[[_Context], int]:
    def command(ctx: _Context) -> int:
        question = ctx.store().set_status(ctx.args.query_id, status, annotator=ctx.annotator(), note=ctx.args.note)
        ctx.say(f"{PREFIX} {question.query_id} is now {question.status}")
        return 0

    return command


def _cmd_status(ctx: _Context) -> int:
    store, config = ctx.store(), ctx.config
    # Once a pool is approved, only its questions are shown (other questions' grades stay out of view); while grading
    # is blind, no question's candidate counts by pooling source are shown either.
    ids = {r["query_id"] for r in _approved_export(ctx)[2]} if config.approved_pool_snapshot else None
    report = judgment_coverage(store, small_pool_below=config.small_pool_below, target_queries=config.target_queries,
                               query_ids=ids)
    if ctx.args.json:
        data = report.to_dict()
        if config.require_blind:
            for row in data["queries"]:
                row.pop("sources", None)
        ctx.say(json.dumps(data, indent=2))
        return 0
    if ids is not None:
        ctx.say(f"{PREFIX} the {len(ids)} questions of approved pool snapshot version "
                f"{config.approved_pool_snapshot.version} only")
    ctx.say(
        f"{PREFIX} {store.corpus.corpus_version} sha256 {store.corpus.corpus_sha256[:12]}... · "
        f"annotation {store.info.evaluation.annotation_version}"
    )
    statuses = ", ".join(f"{s} {n}" for s, n in sorted(report.status_counts.items())) or "none"
    ctx.say(f"questions: {len(report.queries)} written of {config.target_queries} targeted ({statuses})")
    ctx.say(f"{'query':<8} {'status':<9} {'pool':>4} {'graded':>6} {'0':>3} {'1':>3} {'2':>3} {'skip':>4} {'pend':>4}  flags")
    for row in report.queries:
        ctx.say(
            f"{row.query_id:<8} {row.status:<9} {row.pool_size:>4} {row.graded:>6} {row.grades[0]:>3} "
            f"{row.grades[1]:>3} {row.grades[2]:>3} {row.skipped:>4} {row.pending:>4}  {','.join(row.flags) or '-'}"
        )
    t = report.totals
    ctx.say(
        f"totals: pooled {t['pooled']}, graded {t['graded']} (0: {t['grade_0']}, 1: {t['grade_1']}, "
        f"2: {t['grade_2']}), skipped {t['skipped']}, pending {t['pending']}"
    )
    for flag, ids in report.flagged.items():
        if ids:
            ctx.say(f"{flag}: {', '.join(ids)}")
    for name, counts in report.distribution.items():
        ctx.say(f"{name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    ctx.say("unjudged passages (skipped, pending or never pooled) are unknown, not negatives")
    return 0


LEAKAGE_SOURCES = (  # the questions A1 trained and was selected on (docs/gold_set_v1_protocol.md §2.1)
    "data/processed/train/bge_base_ft_v3/candidates/train_candidates_v3/train_candidates.jsonl",
    "data/processed/train/bge_base_ft_v3/validation_sets/validation_pairs_v4/validation_pairs.jsonl",
)


def _cmd_leakage_check(ctx: _Context) -> int:
    """Each written question's closest training or validation question (word-set Jaccard). Reads question texts only:
    the workspace's ``questions.jsonl`` (never its pool or judgment history) and the ``query`` field of the given
    files."""
    rows = [json.loads(line) for line in (Path(ctx.workspace) / QUESTIONS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    questions = [Query(query_id=r["query_id"], text=r["text"], source="gold") for r in rows if r.get("status") != "withdrawn"]
    texts: list[str] = []
    origins: list[str] = []
    for path in ctx.args.against or LEAKAGE_SOURCES:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                texts.append(str(row[ctx.args.field]))
                origins.append(f"{Path(path).name}:{row.get('pair_id', len(origins))}")
    best: dict[str, tuple[float, int]] = {}
    for overlap in find_gold_overlap(questions, texts, threshold=0.0):
        if overlap.similarity > best.get(overlap.query_id, (-1.0, -1))[0]:
            best[overlap.query_id] = (overlap.similarity, overlap.index)
    if ctx.config.question_stage_dir:  # the record export-questions requires (question texts and similarities only)
        sources = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in ctx.args.against or LEAKAGE_SOURCES}
        results = {q.query_id: (best.get(q.query_id, (0.0, -1))[0],
                                origins[best[q.query_id][1]] if q.query_id in best else None) for q in questions}
        record = leakage_record([(q.query_id, q.text) for q in questions], results, threshold=ctx.args.threshold,
                                sources=sources, against=len(texts))
        path = write_leakage_record(ctx.config.question_stage_dir, record)
        ctx.say(f"{PREFIX} recorded in {path}")
    flagged = 0
    ctx.say(f"{PREFIX} {len(questions)} questions against {len(texts)} training and validation questions "
            f"(word-set Jaccard; rewrite at >= {ctx.args.threshold})")
    for q in questions:
        similarity, index = best.get(q.query_id, (0.0, -1))
        mark = "REWRITE" if similarity >= ctx.args.threshold else "ok"
        flagged += mark == "REWRITE"
        closest = f" closest {origins[index]}" if index >= 0 else ""
        ctx.say(f"{q.query_id}  {similarity:.3f}  {mark}{closest}")
    ctx.say(f"{PREFIX} {flagged} question(s) at or above {ctx.args.threshold}")
    return 1 if flagged else 0


def _stage_dir(ctx: _Context) -> Path:
    if not ctx.config.question_stage_dir or not ctx.config.query_plan:
        raise AnnotationError("the gold-set config sets no question_stage_dir or query_plan")
    return Path(ctx.config.question_stage_dir)


def _cmd_question_sheet(ctx: _Context) -> int:
    """The writers' worksheet: one row per slot of the query plan, with empty question columns."""
    path = write_question_sheet(load_plan(ctx.config.query_plan or ""), ctx.args.out or _stage_dir(ctx) / SHEET_FILE)
    ctx.say(f"{PREFIX} wrote {path}: fill in question, writer and ai_assistance (none) for each slot, writing each "
            "question from its slot alone, before any search; then run import-questions")
    return 0


def _cmd_import_questions(ctx: _Context) -> int:
    """Validate the filled worksheet and add its questions (independent, by the named person, tagged with the slot)."""
    ctx.human_annotator()  # whoever imports is a person too
    sheet = ctx.args.sheet or _stage_dir(ctx) / SHEET_FILE
    result = import_questions(ctx.store(), sheet, load_plan(ctx.config.query_plan or ""),
                              ai_identities=ctx.config.ai_annotators)
    ctx.say(f"{PREFIX} imported {len(result['added'])} question(s) from {sheet}; {len(result['unchanged'])} already "
            f"present unchanged")
    if result["added"]:
        ctx.say(f"{PREFIX} added: {', '.join(result['added'])}; next: leakage-check")
    return 0


def _cmd_export_questions(ctx: _Context) -> int:
    """Freeze the next question export version: every slot written, the leakage check current and passing, every
    earlier version intact. Each version is written once and kept."""
    manifest = export_questions(ctx.store(), _stage_dir(ctx), load_plan(ctx.config.query_plan or ""),
                                created_by=ctx.human_annotator())
    ctx.say(f"{PREFIX} froze version {manifest['version']}: {manifest['questions']} questions in {manifest['file']} "
            f"(SHA-256 {manifest['sha256']})")
    if manifest["supersedes"]:
        ctx.say(f"{PREFIX} supersedes version {manifest['supersedes']['version']} ({manifest['supersedes']['sha256']}), "
                f"which is kept unchanged; revised slots: {', '.join(r['slot'] for r in manifest['revisions'])}")
    ctx.say(f"{PREFIX} record it in docs/gold_set_v1_protocol.md; pooling waits for approval")
    return 0


def _cmd_revision_sheet(ctx: _Context) -> int:
    """The revision worksheet for the listed slots (their current questions and writers); never overwritten."""
    slots = [s.strip() for s in ctx.args.slots.split(",") if s.strip()]
    version = (export_versions(_stage_dir(ctx)) or [0])[-1] + 1
    path = write_revision_sheet(ctx.store(), load_plan(ctx.config.query_plan or ""), slots,
                                ctx.args.out or _stage_dir(ctx) / f"revision_sheet_v{version}.csv", reason=ctx.args.reason)
    ctx.say(f"{PREFIX} wrote {path}: {len(slots)} slot(s) for their original writers to revise (revised_question, "
            "writer, ai_assistance = none); then run revise-questions")
    return 0


def _cmd_revise_questions(ctx: _Context) -> int:
    """Validate the filled revision sheet and replace each listed slot's question (all or nothing)."""
    done = revise_questions(ctx.store(), ctx.args.sheet, load_plan(ctx.config.query_plan or ""),
                            ai_identities=ctx.config.ai_annotators, annotator=ctx.human_annotator())
    for change in done:
        ctx.say(f"{PREFIX} {change['slot']}: {change['withdrawn']} withdrawn, revision added as {change['added']}")
    ctx.say(f"{PREFIX} next: leakage-check, then export-questions (a new version; earlier versions are kept)")
    return 0


def _approved_export(ctx: _Context) -> tuple[int, str, list[dict]]:
    """The latest question export: its version, SHA-256 and rows, verified (with every earlier version) first."""
    directory = _stage_dir(ctx)
    versions = export_versions(directory)
    if not versions:
        raise AnnotationError("no frozen question export")
    manifests = {v: verify_export(directory, v) for v in versions}
    out = export_paths(directory, versions[-1])[0]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return versions[-1], manifests[versions[-1]]["sha256"], rows


def _expected_rewording_rows(ctx: _Context, rows: list[dict]) -> list[dict[str, str]]:
    plan = load_plan(ctx.config.query_plan or "")
    return context_rows(rows, {slot: p.period for slot, p in plan.items()})


def _cmd_rewording_sheet(ctx: _Context) -> int:
    """The human writer's rewording worksheet: every approved question, its context and its required number of
    reworded searches, with the rewording, writer and AI-assistance fields left empty."""
    version, sha, rows = _approved_export(ctx)
    versions = pool_snapshot_versions(_stage_dir(ctx))
    if not versions:
        raise AnnotationError("no pool snapshot: freeze the base pool before issuing the rewording sheet")
    next_pool = versions[-1] + 1
    path = write_rewording_sheet(_expected_rewording_rows(ctx, rows),
                                 ctx.args.out or _stage_dir(ctx) / f"rewording_sheet_pool_v{next_pool}.csv")
    required = sum(int(r["rewordings_required"]) for r in _expected_rewording_rows(ctx, rows))
    ctx.say(f"{PREFIX} wrote {path}: {len(rows)} questions of question export version {version} ({sha}), "
            f"{required} rewordings required; for a person to fill in (no AI), then apply-rewordings")
    return 0


def _cmd_apply_rewordings(ctx: _Context) -> int:
    """Validate the filled rewording sheet, run its BM25 searches into the existing pools and freeze the next pool
    snapshot. With --validate-only: the checks alone (no search, no write). With --dry-run: the checks and the
    searches in memory, counting what would be added (no write)."""
    _refuse_if_pools_approved(ctx)
    directory = _stage_dir(ctx)
    version, sha, rows = _approved_export(ctx)
    texts = []
    for path in LEAKAGE_SOURCES:
        texts += [json.loads(line)["query"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = validate_rewording_sheet(ctx.args.sheet, _expected_rewording_rows(ctx, rows),
                                      ai_identities=ctx.config.ai_annotators, training_texts=texts)
    store = ctx.store()
    for qid in result["questions"]:
        check_poolable(store.question(qid), directory)
    parent = check_rewording_base(store, directory, result["questions"])
    if ctx.args.validate_only:
        writers = sorted({v["writer"] for v in result["questions"].values()})
        ctx.say(f"{PREFIX} valid: {len(result['questions'])} questions of question export version {version} ({sha}), "
                f"{sum(len(v['rewordings']) for v in result['questions'].values())} rewordings by {writers}; "
                f"leakage max {result['leakage']['max']} against {result['leakage']['against']} questions (threshold "
                f"{result['leakage']['threshold']}); extends pool snapshot version {parent}; no search run, nothing "
                "written")
        for line in result["observations"]:
            ctx.say(f"{PREFIX} observation: {line}")
        return 0
    sheet_sha = hashlib.sha256(Path(ctx.args.sheet).read_bytes()).hexdigest()
    counts = apply_rewordings(store, result["questions"], depth=ctx.config.bm25_depth, added_by=ctx.annotator(),
                              sheet_label=f"rewording sheet {Path(ctx.args.sheet).name} {sheet_sha[:12]}",
                              dry_run=ctx.args.dry_run)
    searches = sum(c["searches"] for c in counts.values())
    new = sum(c["new"] for c in counts.values())
    ctx.say(f"{PREFIX} {'dry run: ' if ctx.args.dry_run else ''}{len(counts)} questions, {searches} reworded searches, "
            f"{new} new candidates, {sum(c['already_pooled'] for c in counts.values())} already pooled (gain a source); "
            f"leakage max {result['leakage']['max']} (threshold {result['leakage']['threshold']}); extends pool "
            f"snapshot version {parent}")
    for line in result["observations"]:
        ctx.say(f"{PREFIX} observation: {line}")
    if ctx.args.dry_run:
        return 0
    manifest = write_pool_snapshot(
        store, directory, result["questions"], created_by=ctx.annotator(),
        inputs={"question_export": {"version": version, "sha256": sha},
                "issued_sheet": ({"file": issued.name, "sha256": hashlib.sha256(issued.read_bytes()).hexdigest()}
                                 if (issued := directory / f"rewording_sheet_pool_v{parent + 1}.csv").is_file() else None),
                "rewording_sheet": {"file": Path(ctx.args.sheet).name, "sha256": sheet_sha},
                "rewordings": {qid: {"writer": v["writer"], "count": len(v["rewordings"])}
                               for qid, v in result["questions"].items()},
                "leakage": {k: result["leakage"][k] for k in ("threshold", "against", "max")},
                "bm25": f"depth {ctx.config.bm25_depth}, each question's companies, no year filter",
                "added": counts},
        notice="Internal pool record with sources, ranks and scores; annotators never see it (grading is blind). "
               "Reworded searches were written by people, not AI.")
    ctx.say(f"{PREFIX} froze pool snapshot version {manifest['version']} ({manifest['snapshot']['sha256']}), parent "
            f"version {manifest['parent']['version'] if manifest['parent'] else None}")
    return 0


def _approved_pool_gate(ctx: _Context, store: AnnotationStore) -> tuple[set[str], dict] | None:
    """With an approved pool snapshot configured: verify it (``verify_approved_pool``) and return the approved
    questions and its manifest. ``None`` when no snapshot is approved yet."""
    approved = ctx.config.approved_pool_snapshot
    if approved is None:
        return None
    _, _, rows = _approved_export(ctx)
    ids = {r["query_id"] for r in rows}
    manifest = verify_approved_pool(store, _stage_dir(ctx), ids, version=approved.version, sha256=approved.sha256)
    return ids, manifest


def _refuse_if_pools_approved(ctx: _Context, query_id: str | None = None) -> None:
    """Pooling is closed for the approved questions once their pool snapshot is approved."""
    approved = ctx.config.approved_pool_snapshot
    if approved is None:
        return
    if query_id is None or query_id in {r["query_id"] for r in _approved_export(ctx)[2]}:
        raise AnnotationError(f"pooling is closed: pool snapshot version {approved.version} ({approved.sha256[:12]}…) "
                              "is approved, and grading uses it unchanged")


def _cmd_pool_snapshot(ctx: _Context) -> int:
    """Freeze the current pools of the approved questions as the next read-only pool snapshot version."""
    _refuse_if_pools_approved(ctx)
    version, sha, rows = _approved_export(ctx)
    manifest = write_pool_snapshot(ctx.store(), _stage_dir(ctx), [r["query_id"] for r in rows], created_by=ctx.annotator(),
                                   inputs={"question_export": {"version": version, "sha256": sha}},
                                   notice="Internal pool record with sources, ranks and scores; grading is blind.")
    ctx.say(f"{PREFIX} froze pool snapshot version {manifest['version']} ({manifest['snapshot']['sha256']})")
    return 0


def _remote_ready(ctx: _Context) -> tuple[list[str], Any]:
    """Remote grading needs an approved pool snapshot and at least one registered primary annotator."""
    config = ctx.config
    if config.approved_pool_snapshot is None or not config.registered_annotators:
        raise AnnotationError("remote packages need approved_pool_snapshot and primary_annotator (or "
                              "primary_annotators) in the gold-set config")
    return config.registered_annotators, config.approved_pool_snapshot


def _issue_to(ctx: _Context, registered: list[str]) -> str:
    """The annotator a package or assignment is issued to: ``--for``, or the only registered primary annotator."""
    wanted = getattr(ctx.args, "issue_to", None)
    if wanted is None:
        if len(registered) != 1:
            raise AnnotationError(f"pass --for: the registered primary annotators are {', '.join(registered)}")
        return registered[0]
    if wanted not in registered:
        raise AnnotationError(f"{wanted!r} is not a registered primary annotator ({', '.join(registered)})")
    return wanted


def _forbidden_texts(ctx: _Context, store: AnnotationStore, query_ids: Iterable[str]) -> list[str]:
    """Texts that must not reach the annotator: every pooling source detail and every rewording."""
    texts = {s.detail for qid in query_ids for e in store.pool(qid) for s in e.sources}
    for sheet in _stage_dir(ctx).glob("rewording_sheet_pool_v*_completed.csv"):
        with sheet.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                texts.update(row.get(f, "").strip() for f in ("rewording_1", "rewording_2", "rewording_3"))
    return sorted(t for t in texts if t)


def _cmd_remote_package(ctx: _Context) -> int:
    """Build blinded remote grading packages (eval_set.remote_package) for approved, open, ungraded questions."""
    registered, approved_cfg = _remote_ready(ctx)
    primary = _issue_to(ctx, registered)
    store = ctx.store()
    approved = _approved_pool_gate(ctx, store)
    assert approved is not None
    ids, pool_manifest = approved
    version, sha, _ = _approved_export(ctx)
    directory = _stage_dir(ctx)
    assigned = assigned_questions(directory)
    if ctx.args.questions:
        wanted = sorted({q.strip() for arg in ctx.args.questions for q in arg.split(",") if q.strip()})
        if outside := [q for q in wanted if q not in ids]:
            raise AnnotationError(f"not approved questions: {outside}")
        if clash := [f"{q} ({assigned[q][0]})" for q in wanted if q in assigned]:
            raise AnnotationError(f"already in a plain-text assignment: {', '.join(clash)}")
        batches = [wanted]
    else:
        taken = packaged_questions(directory)
        rest = sorted(q for q in ids if q not in taken and q not in assigned and store.question(q).status == "open"
                      and not store.judgments(q))
        size = ctx.args.batch_size
        batches = [rest[i:i + size] for i in range(0, len(rest), size)]
        if not batches:
            ctx.say(f"{PREFIX} no approved question is left to package")
            return 0
    for batch in batches:
        manifest = build_package(
            store, directory, batch, annotator=primary,
            approved_pool_snapshot={"version": approved_cfg.version, "sha256": pool_manifest["snapshot"]["sha256"]},
            question_export={"version": version, "sha256": sha}, created_by=ctx.annotator(),
            filing=filing_line, section=section_label, order_seed=BLIND_ORDER_SEED,
            forbidden_texts=_forbidden_texts(ctx, store, batch))
        zip_rel = next(k for k in manifest["files"] if k.endswith(".zip"))
        ctx.say(f"{PREFIX} package {manifest['package_id']}: {len(manifest['questions'])} question(s), "
                f"{manifest['candidates']} candidates, audit passed; send "
                f"{remote_root(directory) / manifest['package_id'] / zip_rel} "
                f"(sha256 {manifest['files'][zip_rel]})")
    return 0



def _cmd_remote_import(ctx: _Context) -> int:
    """Validate a returned grades file and append its grades to the history (``--dry-run``: validate only)."""
    registered, _ = _remote_ready(ctx)
    store = ctx.store()
    issued = load_manifest(remote_root(_stage_dir(ctx)) / ctx.args.package)["annotator"] \
        if (remote_root(_stage_dir(ctx)) / ctx.args.package / "manifest.json").is_file() else None
    primary = issued if issued in registered else registered[0]
    result = import_returned(
        store, _stage_dir(ctx), ctx.args.package, ctx.args.file, primary=primary,
        ai_identities=ctx.config.ai_annotators, operator=ctx.annotator(),
        verify_pools=lambda: _approved_pool_gate(ctx, store)[1], dry_run=ctx.args.dry_run)
    g = result["grades"]
    ctx.say(f"{PREFIX} {'dry run: ' if ctx.args.dry_run else ''}{result['package_id']} from {result['annotator']}: "
            f"{result['candidates']} candidates (2: {g['2']}, 1: {g['1']}, 0: {g['0']}, skipped: {g['s']}), "
            f"{result['notes']} notes, P3 {result['p3']}, P4 {result['p4']}, assisted {result['assisted']}; "
            f"file sha256 {result['sha256']}")
    for line in result["observations"]:
        ctx.say(f"{PREFIX} observation: {line}")
    if not ctx.args.dry_run:
        h = result["history"]
        ctx.say(f"{PREFIX} imported: history events {h['first_seq']}-{h['last_seq']}; questions complete: "
                f"{', '.join(result['questions'])}; original kept at {result['kept_as']}")
    return 0


def _cmd_remote_status(ctx: _Context) -> int:
    for d in package_dirs(_stage_dir(ctx)):
        m = load_manifest(d)
        state = ("withdrawn" if is_withdrawn(d) else "imported" if (d / "import_record.json").is_file()
                 else "out for grading")
        ctx.say(f"{m['package_id']}  {state:<16} {len(m['questions']):>3} question(s) {m['candidates']:>5} candidates  "
                f"issued to {m['annotator']} {m['created_at'][:10]}")
    if not package_dirs(_stage_dir(ctx)):
        ctx.say(f"{PREFIX} no remote package yet")
    return 0


def _cmd_remote_withdraw(ctx: _Context) -> int:
    """Withdraw remote packages that were never imported (their files stay; their questions can be issued again)."""
    operator = ctx.annotator()
    directory = _stage_dir(ctx)
    wanted = list(dict.fromkeys(ctx.args.package))
    for package_id in wanted:  # check every package before withdrawing any
        d = remote_root(directory) / package_id
        if not (d / "manifest.json").is_file():
            raise AnnotationError(f"no package {package_id!r}")
        if is_withdrawn(d) or (d / "import_record.json").exists() or (d / "returned").exists():
            raise AnnotationError(f"{package_id} is already withdrawn or imported; nothing withdrawn")
    for package_id in wanted:
        record = withdraw_package(directory, package_id, reason=ctx.args.reason, operator=operator)
        ctx.say(f"{PREFIX} {package_id} withdrawn ({len(record['questions'])} question(s)); its files are kept "
                "unchanged, and it can never be imported")
    return 0


def _split_inputs(ctx: _Context, store: AnnotationStore, ids: set[str], registered: list[str]
                  ) -> tuple[list[str], dict[str, list[str]]]:
    """The approved questions still to grade, and those each registered annotator already completed."""
    directory = _stage_dir(ctx)
    taken, assigned = packaged_questions(directory), assigned_questions(directory)
    remaining = sorted(q for q in ids if q not in taken and q not in assigned and store.question(q).status == "open"
                       and not store.judgments(q))
    credited: dict[str, list[str]] = {n: [] for n in registered}
    for q in sorted(ids):
        who = {j.annotator for j in store.judgments(q)}
        if store.question(q).status == "complete" and len(who) == 1 and (w := next(iter(who))) in credited:
            credited[w].append(q)
    done = {q for qs in credited.values() for q in qs}
    if other := sorted(q for q in ids if q not in done and q not in remaining):
        raise AnnotationError(f"{len(other)} approved question(s) are neither open and ungraded nor completed by a "
                              f"registered annotator (out in a package or assignment, or partly graded), e.g. "
                              f"{other[:5]}; withdraw or finish them first")
    return remaining, credited


def _cmd_assign_split(ctx: _Context) -> int:
    """Split the remaining approved questions between the two registered annotators (written once)."""
    registered, approved_cfg = _remote_ready(ctx)
    store = ctx.store()
    approved = _approved_pool_gate(ctx, store)
    assert approved is not None
    ids, pool_manifest = approved
    version, sha, rows = _approved_export(ctx)
    counts: dict[str, int] = {}
    for item in ctx.args.count:
        name, sep, n = item.partition("=")
        if not sep or not n.strip().isdigit():
            raise AnnotationError(f"--count takes INITIALS=N, not {item!r}")
        counts[name.strip()] = int(n)
    if sorted(counts) != sorted(registered):
        raise AnnotationError(f"give a --count for each registered primary annotator ({', '.join(registered)})")
    remaining, credited = _split_inputs(ctx, store, ids, registered)
    types = {r["query_id"]: r["question_type"] for r in rows}
    sizes = {q: len(store.pool(q)) for q in ids}
    result = split_questions({q: sizes[q] for q in remaining}, types, counts, credited=credited,
                             credited_sizes=sizes, credited_types=types, seed=SPLIT_SEED, draws=SPLIT_DRAWS)
    record = {
        "kind": "finretriever_gold_split", "created_at": utc_now(), "created_by": ctx.annotator(),
        "approved_pool_snapshot": {"version": approved_cfg.version, "sha256": pool_manifest["snapshot"]["sha256"]},
        "question_export": {"version": version, "sha256": sha}, "counts": counts,
        "remaining_questions": len(remaining),
        "method": (f"within each question type, each annotator's share (counting questions they already completed) is "
                   f"as even as the counts allow; draws from random.Random('{SPLIT_SEED}'), the first draw with the "
                   "smallest difference in total passages (completed and new) is kept"),
        **result, "crosscheck_rule": CROSSCHECK_RULE,
        "notice": "Internal: which annotator grades which question. Neither annotator sees the other's grades."}
    path = write_split(_stage_dir(ctx), record)
    for name, t in result["totals"].items():
        ctx.say(f"{PREFIX} {name}: {t['new_questions']} new question(s), {t['new_passages']} passages; with "
                f"{t['credited_questions']} completed ({t['credited_passages']} passages): {t['total_questions']} "
                f"questions, {t['total_passages']} passages; types {t['types']}")
    ctx.say(f"{PREFIX} passage difference {result['passage_difference']} (draw {result['draw_kept']}); written once "
            f"to {path}")
    return 0


def _cmd_assign_build(ctx: _Context) -> int:
    """Build one annotator's plain-text assignment from the split."""
    registered, approved_cfg = _remote_ready(ctx)
    who = _issue_to(ctx, registered)
    store = ctx.store()
    approved = _approved_pool_gate(ctx, store)
    assert approved is not None
    _, pool_manifest = approved
    version, sha, _ = _approved_export(ctx)
    directory = _stage_dir(ctx)
    split = load_split(directory)
    if split["approved_pool_snapshot"]["sha256"] != pool_manifest["snapshot"]["sha256"]:
        raise AnnotationError("the split was made from another pool snapshot than the approved one")
    split_path = directory / "assignments" / "split_v1.json"
    qids = split["assignments"][who]
    manifest = build_assignment(
        store, directory, qids, annotator=who,
        approved_pool_snapshot={"version": approved_cfg.version, "sha256": pool_manifest["snapshot"]["sha256"]},
        question_export={"version": version, "sha256": sha}, created_by=ctx.annotator(),
        filing=filing_line, section=section_label, order_seed=BLIND_ORDER_SEED,
        forbidden_texts=_forbidden_texts(ctx, store, qids),
        split={"file": split_path.name, "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()})
    zip_rel = next(k for k in manifest["files"] if k.endswith(".zip"))
    ctx.say(f"{PREFIX} assignment {manifest['assignment_id']} for {who}: {len(manifest['questions'])} question(s), "
            f"{manifest['candidates']} passages, audit passed; send "
            f"{directory / 'assignments' / manifest['assignment_id'] / zip_rel} (sha256 {manifest['files'][zip_rel]})")
    return 0


def _cmd_assign_import(ctx: _Context) -> int:
    """Validate a returned response file and record its fully graded questions (``--dry-run``: validate only)."""
    registered, _ = _remote_ready(ctx)
    store = ctx.store()
    r = import_responses(store, _stage_dir(ctx), ctx.args.assignment, ctx.args.file, registered=registered,
                         ai_identities=ctx.config.ai_annotators, operator=ctx.annotator(),
                         verify_pools=lambda: _approved_pool_gate(ctx, store)[1], dry_run=ctx.args.dry_run)
    g = {k: r["new_grades"].get(k, 0) for k in ("2", "1", "0", "s")}
    ctx.say(f"{PREFIX} {'dry run: ' if ctx.args.dry_run else ''}{r['assignment_id']} from {r['annotator']}, file "
            f"sha256 {r['sha256']}")
    ctx.say(f"{PREFIX} to record: {len(r['recorded'])} question(s), {r['new_candidates']} passages (2: {g['2']}, "
            f"1: {g['1']}, 0: {g['0']}, skipped: {g['s']}): {', '.join(r['recorded']) or 'none'}")
    ctx.say(f"{PREFIX} already recorded, unchanged: {len(r['already_recorded_unchanged'])}; partly graded (waiting): "
            f"{', '.join(r['partial']) or 'none'}; not started: {len(r['blank'])}")
    for line in r["observations"]:
        ctx.say(f"{PREFIX} observation: {line}")
    if not ctx.args.dry_run:
        h = r["history"]
        ctx.say(f"{PREFIX} recorded: history events {h['first_seq']}-{h['last_seq']}; file kept at {r['kept_as']}")
    return 0


def _cmd_assign_status(ctx: _Context) -> int:
    dirs = assignment_dirs(_stage_dir(ctx))
    for d in dirs:
        m = load_assignment_manifest(d)
        done = recorded_questions(d)
        passages = sum(q["candidates"] for q in m["questions"] if q["query_id"] in done)
        ctx.say(f"{m['assignment_id']}  {m['annotator']:<4} {len(done):>3}/{len(m['questions'])} questions recorded  "
                f"{passages:>5}/{m['candidates']} passages  {len(import_records(d))} import(s)  "
                f"issued {m['created_at'][:10]}")
    if not dirs:
        ctx.say(f"{PREFIX} no plain-text assignment yet")
    return 0


REGRADE_DIR = "data/eval/regrade/gold_v1"  # docs/gold_set_v1_protocol.md §2.4
REGRADE_QUESTIONS = 20
REGRADE_SEED = 20261112


def _cmd_regrade_init(ctx: _Context) -> int:
    """Set up the second annotator's blind re-grade: every pooled passage of a seeded sample of complete questions,
    without any grade, note, source or rank."""
    store = ctx.store()
    if (approved := _approved_pool_gate(ctx, store)) is not None:
        ctx.say(f"{PREFIX} pool snapshot version {approved[1]['version']} ({approved[1]['snapshot']['sha256']}): "
                "verified, unchanged")
    regrade = create_regrade(ctx.args.regrade_dir, store, count=ctx.args.questions, seed=ctx.args.seed,
                             created_by=ctx.annotator())
    questions = len({item.query_id for item in regrade.items()})
    ctx.say(f"{PREFIX} blind re-grade of {len(regrade.items())} passages across {questions} questions created at "
            f"{regrade.directory} (seed {ctx.args.seed}); it holds no grade, source or rank")
    ctx.say(f"{PREFIX} the second person grades with: python3 scripts/03_annotate_gold_set.py audit --audit-dir "
            f"{regrade.directory} --annotator <their initials>; then run regrade-report")
    return 0


def _cmd_regrade_report(ctx: _Context) -> int:
    regrade = AuditWorkspace.open(ctx.args.regrade_dir, ctx.corpus)
    report = regrade_agreement(regrade, ctx.store())
    path = write_regrade_report(regrade, report)

    def fmt(value: float | None, pct: bool = False) -> str:
        return "n/a" if value is None else (f"{value:.1%}" if pct else f"{value:.3f}")

    ctx.say(f"{PREFIX} re-grade {regrade.directory}: {report['compared']} passages compared across "
            f"{report['questions']} questions (skipped: first {report['skipped']['first']}, second "
            f"{report['skipped']['second']})")
    ctx.say(f"raw agreement {fmt(report['raw_agreement'], True)} · relevant-vs-not {fmt(report['relevant_vs_not_agreement'], True)}"
            f" · Cohen's kappa {fmt(report['cohen_kappa'])} · linear-weighted kappa {fmt(report['linear_weighted_kappa'])}")
    ctx.say(f"{PREFIX} {len(report['disagreements'])} disagreement(s), listed with both grades and notes in the report")
    ctx.say(f"{PREFIX} written once to {path}; reported, never deciding")
    return 0


def _cmd_history(ctx: _Context) -> int:
    events = ctx.store().history(ctx.args.query_id, ctx.args.passage_id)
    for e in events:
        change = {
            "grade": f"{e.previous or 'pending'} -> {e.grade}",
            "skip": f"{e.previous or 'pending'} -> skipped",
            "clear": f"{e.previous} -> pending",
            "question_status": f"{e.previous} -> {e.status}",
        }.get(e.kind, "")
        note = e.note
        if e.kind == "candidates_added" and ctx.config.require_blind:
            note = "details hidden (blind grading)"  # the note counts candidates by pooling method
        parts = [f"#{e.seq}", e.at, e.annotator, e.kind, e.passage_id or "", change, f"— {note}" if note else ""]
        ctx.say(" ".join(part for part in parts if part))
    if not events:
        ctx.say(f"{PREFIX} no events")
    return 0


def _cmd_export(ctx: _Context) -> int:
    config = ctx.config
    if (ctx.workspace / AUDIT_FILE).exists():
        raise AnnotationError(f"{ctx.workspace} is an audit workspace; audit judgments are never exported as gold")
    export = build_gold_export(
        ctx.store(),
        include_open=ctx.args.include_open,
        small_pool_below=config.small_pool_below,
        target_queries=config.target_queries,
        ai_identities=config.ai_annotators,
    )
    paths = write_gold_export(export, ctx.args.out_dir or config.export_dir)
    grades = Counter(r.relevance for r in export.qrels)
    ctx.say(
        f"{PREFIX} exported {len(export.queries)} queries and {len(export.qrels)} qrels "
        f"(0: {grades[0]}, 1: {grades[1]}, 2: {grades[2]}) for {config.target_queries} targeted"
    )
    for query_id, reason in export.excluded.items():
        ctx.say(f"{PREFIX} not exported {query_id}: {reason}")
    for name, path in paths.items():
        ctx.say(f"{PREFIX} {name}: {path}")
    return 0


def _refuse_protected(target: str | Path, protected: dict[str, str | Path], *, what: str = "development labels") -> None:
    resolved = Path(target).resolve()
    for name, path in protected.items():
        guard = Path(path).resolve()
        if resolved == guard or guard in resolved.parents:
            raise DevLabelError(f"refusing to write {what} into {name} ({path})")


def _refuse_gold_export_dir(ctx: _Context, target: str | Path) -> None:
    """The gold files' own directory is refused; development subdirectories such as dev_ai/ are not."""
    if Path(target).resolve() == Path(ctx.config.export_dir).resolve():
        raise DevLabelError(f"refusing to write development data into the gold export directory {ctx.config.export_dir}")


def _cmd_expand_pool(ctx: _Context) -> int:
    spec, spec_sha256 = load_dev_pool_spec(ctx.args.spec)
    passages = load_corpus_passages(ctx.corpus)
    base = DevLabelSet.open(spec.base_labels_dir, ctx.corpus, passages)
    base.verify_inputs()
    _refuse_protected(
        spec.output_dir,
        {
            "the annotation workspace": ctx.workspace,
            "the base labels": spec.base_labels_dir,
            "the stored runs": spec.runs_dir,
            **{f"an input of {base.info.label_version}": path for path in base.input_dirs()},
        },
        what="an expanded pool",
    )
    _refuse_gold_export_dir(ctx, spec.output_dir)
    runs = [
        load_stored_run(spec.runs_dir, ref.mode, ref.run, corpus=ctx.corpus, passages=passages) for ref in spec.runs
    ]
    pool = expand_pool(
        spec, spec_path=ctx.args.spec, spec_sha256=spec_sha256, base=base,
        base_pool=read_annotation_pool(spec.base_pool_path), runs=runs, passages=passages,
    )
    written = pool.write()
    c = pool.info.coverage
    ctx.say(
        f"{PREFIX} {'wrote' if written else 'unchanged (identical copy present)'} expanded pool "
        f"{spec.pool_version} at {spec.output_dir}: {c['entries']} entries = {c['base_pool_entries']} from the "
        f"{base.info.label_version} pool + {c['new_entries']} new from the top {c['depth']} of {len(runs)} runs "
        f"({c['new_in_scope']} from the companies asked about, {c['new_out_of_scope']} from others)"
    )
    ctx.say(f"{PREFIX} new entries by system: {c['new_by_system']}; by system set: {c['new_by_system_set']}")
    depth = c["depth"]
    for run_id, row in c["per_run"].items():
        ctx.say(
            f"{PREFIX} {run_id}: judged@{depth} before expansion {row[f'judged@{depth}_before']:.3f}, "
            f"{row['unjudged_top_k']} unjudged in its top {depth}"
        )
    ctx.say(
        f"{PREFIX} {c['unjudged_entries']} entries are unjudged: adding a passage to a pool is not a judgment; "
        "they stay out of the qrels until a label set grades them"
    )
    return 0


def _cmd_dev_labels(ctx: _Context) -> int:
    if is_expanded_label_spec(ctx.args.spec):
        return _dev_labels_expanded(ctx)
    spec, spec_sha256 = load_dev_label_spec(ctx.args.spec)
    store = ctx.store()
    proposals = ProposalSet.open(spec.base_proposals_dir, store)
    audit = AuditWorkspace.open(spec.audit_dir, ctx.corpus)
    _refuse_protected(
        spec.output_dir,
        {"the annotation workspace": ctx.workspace, "the base proposals": proposals.directory, "the audit": audit.directory},
    )
    _refuse_gold_export_dir(ctx, spec.output_dir)
    label_set = derive_dev_labels(
        spec, spec_path=ctx.args.spec, spec_sha256=spec_sha256, store=store, proposals=proposals, audit=audit
    )
    written = label_set.write(spec.output_dir)
    counts = label_set.info.counts
    ctx.say(
        f"{PREFIX} {'wrote' if written else 'unchanged (identical copy present)'} development labels "
        f"{spec.label_version} at {spec.output_dir}: {counts['labels']} labels {counts['by_source']}, "
        f"grades {counts['grades']}"
    )
    ctx.say(f"{PREFIX} grade changes by rule: {counts['changes'] or 'none'}; open flags by rule: {counts['flags'] or 'none'}")
    for d in label_set.info.audit_dispositions:
        ctx.say(
            f"{PREFIX} audit item {d.item} ({d.query_id}): human {d.human_grade}, base {d.base_grade}, "
            f"label {d.grade} -> {d.disposition} under {d.rule}"
        )
    if counts["questions_without_positive"]:
        ctx.say(f"{PREFIX} questions without a positive label: {', '.join(counts['questions_without_positive'])}")
    ctx.say(f"{PREFIX} these are AI-generated development labels; export them with export-dev, never as gold")
    return 0


def _dev_labels_expanded(ctx: _Context) -> int:
    spec, spec_sha256 = load_expanded_label_spec(ctx.args.spec)
    passages = load_corpus_passages(ctx.corpus)
    base = DevLabelSet.open(spec.base_labels_dir, ctx.corpus, passages)
    base.verify_inputs()
    pool = DevPool.open(spec.pool_dir, ctx.corpus, passages)
    proposals = open_pool_proposals(spec.proposals_dir, pool)
    _refuse_protected(
        spec.output_dir,
        {
            "the annotation workspace": ctx.workspace,
            "the base labels": spec.base_labels_dir,
            "the pool": spec.pool_dir,
            "the proposals": spec.proposals_dir,
            **{f"an input of {base.info.label_version}": path for path in base.input_dirs()},
        },
    )
    _refuse_gold_export_dir(ctx, spec.output_dir)
    label_set = derive_expanded_labels(
        spec, spec_path=ctx.args.spec, spec_sha256=spec_sha256, base=base, pool=pool, proposals=proposals,
        passages=passages,
    )
    written = label_set.write(spec.output_dir)
    counts = label_set.info.counts
    ctx.say(
        f"{PREFIX} {'wrote' if written else 'unchanged (identical copy present)'} development labels "
        f"{spec.label_version} at {spec.output_dir}: {counts['labels']} labels = {counts['carried']} carried "
        f"unchanged from {base.info.label_version} + {counts['new']} new AI labels {counts['new_grades']}; "
        f"{counts['unlabelled']} pool entries still unjudged"
    )
    ctx.say(
        f"{PREFIX} new labels: confidence {counts['new_confidence']}, changes by rule "
        f"{counts['new_changes'] or 'none'}, flags by rule {counts['new_flags'] or 'none'}"
    )
    if counts["questions_with_new_positives"]:
        ctx.say(f"{PREFIX} questions with new positive labels: {', '.join(counts['questions_with_new_positives'])}")
    ctx.say(f"{PREFIX} these are AI-generated development labels; export them with export-dev, never as gold")
    return 0


def _cmd_export_dev(ctx: _Context) -> int:
    labels_dir = Path(ctx.args.labels)
    passages = load_corpus_passages(ctx.corpus)
    label_set = DevLabelSet.open(labels_dir, ctx.corpus, passages)
    label_set.verify_inputs()
    export = build_dev_export(label_set, passages)
    paths = write_dev_export(
        export,
        ctx.args.out_dir or labels_dir,
        gold_export_dir=ctx.config.export_dir,
        protected_dirs=[ctx.workspace, *label_set.input_dirs()],
    )
    labels = export.report["labels"]
    ctx.say(
        f"{PREFIX} exported {label_set.info.label_version} (provisional development labels, not gold): "
        f"{len(export.queries)} queries, {labels['qrels']} qrels {labels['grades']}, sources {labels['by_source']}"
    )
    exceptions = export.report["exceptions"]
    ctx.say(
        f"{PREFIX} questions without a positive label: {', '.join(exceptions['no_positive']) or 'none'}; "
        f"without a grade-2 label: {', '.join(exceptions['no_grade_2']) or 'none'}"
    )
    for name, path in paths.items():
        ctx.say(f"{PREFIX} {name}: {path}")
    return 0


def _audit_dir(ctx: _Context) -> Path:
    return Path(ctx.args.audit_dir or ctx.config.audit_dir)


def _cmd_audit_init(ctx: _Context) -> int:
    store = ctx.store()
    proposals = ProposalSet.open(ctx.args.proposals_dir or ctx.config.ai_proposals_dir, store)
    audit = create_audit(
        _audit_dir(ctx),
        store,
        proposals,
        size=ctx.args.size or ctx.config.audit_sample_size,
        seed=ctx.config.audit_seed if ctx.args.seed is None else ctx.args.seed,
        created_by=ctx.annotator(),
        annotation_version=f"{store.info.evaluation.annotation_version}-audit",
    )
    questions = len({item.query_id for item in audit.items()})
    ctx.say(
        f"{PREFIX} blind audit of {len(audit.items())} passages across {questions} questions created at "
        f"{audit.directory} (seed {audit.info.seed}); it contains no AI labels"
    )
    ctx.say(f"{PREFIX} grade with: python3 scripts/03_annotate_gold_set.py audit --annotator <initials>")
    return 0


def _cmd_audit_init_pool(ctx: _Context) -> int:
    spec, _ = load_pool_audit_spec(ctx.args.spec)
    passages = load_corpus_passages(ctx.corpus)
    labels = DevLabelSet.open(spec.labels_dir, ctx.corpus, passages)
    labels.verify_inputs()
    pool = DevPool.open(spec.pool_dir, ctx.corpus, passages)
    proposals = open_pool_proposals(spec.proposals_dir, pool)
    if labels.info.derived_from.get("proposals_sha256") != proposals.sha256():
        raise AuditError(f"{spec.labels_dir} was not derived from the proposals in {spec.proposals_dir}")
    ref = (spec.lexical_run.run, spec.lexical_run.mode)
    recorded = next((r for r in pool.info.runs if (r["run"], r["mode"]) == ref), None)
    if recorded is None:
        raise AuditError(f"the pool was not expanded from a stored run {ref[1]}/{ref[0]}")
    lexical = load_stored_run(
        Path(recorded["run_file"]).parents[2], ref[1], ref[0], corpus=ctx.corpus, passages=passages
    )
    if lexical.run_sha256 != recorded["run_sha256"]:
        raise AuditError(f"{recorded['run_file']} changed after the pool was expanded")
    _refuse_protected(
        spec.output_dir,
        {
            "the annotation workspace": ctx.workspace,
            "the labels": spec.labels_dir,
            "the pool": spec.pool_dir,
            "the proposals": spec.proposals_dir,
        },
        what="an audit",
    )
    audit = create_pool_audit(
        spec, corpus=ctx.corpus, questions=labels.questions, pool=pool, proposals=proposals,
        lexical=lexical, passages=passages, created_by=ctx.annotator(),
    )
    counts = Counter(item.query_id for item in audit.items())
    ctx.say(
        f"{PREFIX} blind priority audit of {len(audit.items())} passages across {len(counts)} questions created "
        f"at {audit.directory} (seed {spec.seed}); it contains no AI labels"
    )
    ctx.say(f"{PREFIX} items per question: {', '.join(f'{q} {n}' for q, n in sorted(counts.items()))}")
    ctx.say(f"{PREFIX} grade with: python3 scripts/03_annotate_gold_set.py audit --audit-dir {audit.directory} --annotator <initials>")
    return 0


def _audit_proposals(ctx: _Context, audit: AuditWorkspace) -> ProposalSet:
    """The proposals an audit was drawn from: an annotation workspace's, or an expanded pool's."""
    source = Path(audit.info.source_workspace)
    if (source / POOL_INFO_FILE).is_file():
        pool = DevPool.open(source, ctx.corpus, load_corpus_passages(ctx.corpus))
        return open_pool_proposals(audit.info.proposals_dir, pool)
    return ProposalSet.open(audit.info.proposals_dir, AnnotationStore.open(source, ctx.corpus))


def _cmd_audit(ctx: _Context) -> int:
    audit = AuditWorkspace.open(_audit_dir(ctx), ctx.corpus)
    AuditSession(
        audit, ctx.human_annotator(), input_fn=ctx.input_fn, out=ctx.out, preview_chars=ctx.args.preview_chars
    ).run()
    return 0


def _cmd_audit_status(ctx: _Context) -> int:
    audit = AuditWorkspace.open(_audit_dir(ctx), ctx.corpus)
    p = audit.progress()
    ctx.say(
        f"{PREFIX} blind audit {audit.directory}: {p['items']} items · graded {p['graded']} · "
        f"skipped {p['skipped']} · pending {p['pending']}"
    )
    ctx.say(f"{PREFIX} " + ("complete: run audit-report" if not p["pending"] else "AI labels stay hidden until complete"))
    return 0


def _cmd_audit_report(ctx: _Context) -> int:
    audit = AuditWorkspace.open(_audit_dir(ctx), ctx.corpus)
    report = audit_agreement(audit, _audit_proposals(ctx, audit))
    payload = report.to_dict()
    (audit.directory / REPORT_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if ctx.args.json:
        ctx.say(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    ctx.say(f"{PREFIX} audit {audit.directory}: {report.graded} graded, {report.skipped} skipped of {report.items}")
    kappa = "n/a" if report.cohen_kappa is None else f"{report.cohen_kappa:.3f}"
    ctx.say(
        f"raw agreement {pct(report.raw_agreement)} · relevant-vs-not agreement {pct(report.binary_agreement)} · "
        f"Cohen's kappa {kappa} · stratum-weighted agreement {pct(report.weighted_raw_agreement)}"
    )
    ctx.say("confusion (rows: human grade, columns: AI grade)")
    ctx.say("        AI 0  AI 1  AI 2")
    for human, row in report.confusion.items():
        ctx.say(f"  H {human}  " + "".join(f"{row[g]:>5} " for g in ("0", "1", "2")))
    for title, table in (
        ("by question type", report.by_question_type),
        ("by human grade", report.by_human_grade),
        ("by AI grade", report.by_ai_grade),
        ("by AI confidence", report.by_ai_confidence),
    ):
        cells = ", ".join(f"{k} {v['disagree']}/{v['items']}" for k, v in table.items() if v["items"])
        ctx.say(f"disagreements {title}: {cells}")
    ctx.say(f"{PREFIX} report: {audit.directory / REPORT_FILE}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help=f"Gold-set YAML config (default {DEFAULT_CONFIG}).")
    common.add_argument("--workspace", help="Annotation workspace (default: workspace_dir in the config).")
    common.add_argument("--corpus", help="Corpus JSONL (default: corpus_path); it must match the frozen SHA-256.")
    common.add_argument("--manifest", help="Corpus manifest (default: next to the corpus).")
    common.add_argument("--annotator", help=f"Your initials, recorded with every change (or set {ANNOTATOR_ENV}).")

    parser = argparse.ArgumentParser(
        prog="03_annotate_gold_set.py",
        description="Write, pool and judge gold evaluation questions over the frozen corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], help=help_text, description=help_text)

    add("init", "Create a workspace bound to the frozen corpus.")
    p = add("add-question", "Register a manually written question.")
    p.add_argument("--text", required=True)
    p.add_argument("--topic", required=True, choices=TOPICS)
    p.add_argument("--type", dest="question_type", required=True, choices=QUESTION_TYPES)
    p.add_argument("--comparison", default="none", choices=COMPARISONS)
    p.add_argument("--ticker", action="append", default=[], help="Company the question is about (repeatable).")
    p.add_argument("--year", action="append", type=int, default=[], help="Fiscal year asked about (repeatable).")
    p.add_argument(
        "--authoring", required=True, choices=AUTHORING,
        help="independent: written before reading passages; passage_inspired: written after reading one.",
    )
    p.add_argument(
        "--ai-assisted", action="store_true",
        help="The question was drafted by an AI model (--annotator names it); needs --reviewed-by.",
    )
    p.add_argument("--reviewed-by", default="", help="Person who approved an AI-assisted question.")
    p.add_argument("--notes", default="")

    p = add("pool", "Propose candidates for a question and add them to its pool.")
    p.add_argument("query_id")
    p.add_argument("--bm25-depth", type=int, help="BM25 candidates per search (default from config).")
    p.add_argument("--reformulation", action="append", default=[], help="Extra BM25 search text (repeatable).")
    p.add_argument("--passage-id", action="append", default=[], help="Add a passage found manually (repeatable).")
    p.add_argument("--manual-note", default="", help="How the --passage-id passages were found (kept in the pool).")
    p.add_argument("--stratified", type=int, help="Sampled passages per in-scope filing (default from config).")
    p.add_argument("--random", type=int, help="Uniformly sampled in-scope passages (default from config).")
    p.add_argument("--use-years", action="store_true", help="Also restrict candidates to the question's fiscal years.")

    p = add("search", "BM25 lookup to find passages for manual candidates (changes nothing).")
    p.add_argument("text")
    p.add_argument("--ticker", action="append", default=[])
    p.add_argument("--top-k", type=int, default=10)

    p = add("show", "Print one passage with its filing details.")
    p.add_argument("passage_id")

    p = add("annotate", "Judge a question's candidates interactively (resumes where you stopped).")
    p.add_argument("query_id", nargs="?", help="Default: the first open question with pending candidates.")
    p.add_argument("--preview-chars", type=int, default=1200)
    p.add_argument(
        "--hide-scores", action="store_true",
        help="Hide retrieval ranks and scores while grading (display only; nothing stored changes).",
    )
    p.add_argument(
        "--blind", action="store_true",
        help="Blind grading: hide every pool source (method, rank, score, search) and show the candidates in an order "
        "shuffled per question, the same in every session. Display only; nothing stored changes. The gold set's "
        "protocol requires it (docs/gold_set_v1_protocol.md).",
    )

    for name, help_text in (
        ("complete", "Mark a fully judged question complete (freezes it)."),
        ("reopen", "Reopen a complete or withdrawn question for editing."),
        ("withdraw", "Withdraw a question; it keeps its ID and is never exported."),
    ):
        p = add(name, help_text)
        p.add_argument("query_id")
        p.add_argument("--note", default="")

    p = add("status", "Judgment coverage diagnostics.")
    p.add_argument("--json", action="store_true")

    p = add("leakage-check", "Compare the written questions with the training and validation questions (texts only).")
    p.add_argument("--against", action="append", metavar="JSONL",
                   help="A JSONL file of questions to compare with (repeat). Default: the A1 training and validation "
                   "pairs.")
    p.add_argument("--field", default="query", help="The question field in those files (default: query).")
    p.add_argument("--threshold", type=float, default=0.8)

    p = add("question-sheet", "Write the gold v1 writers' worksheet from the query plan (never overwritten).")
    p.add_argument("--out", help="Default: question_sheet.csv in the config's question_stage_dir.")

    p = add("import-questions", "Validate the filled worksheet and add its questions (human-written, independent).")
    p.add_argument("--sheet", help="Default: question_sheet.csv in the config's question_stage_dir.")

    p = add("export-questions", "Freeze the next question export version (every slot, leakage check passing).")

    p = add("revision-sheet", "Write a revision worksheet for listed slots (their current question and writer).")
    p.add_argument("--slots", required=True, help="Comma-separated slots, e.g. S024,S035.")
    p.add_argument("--reason", required=True, help="Why the slots are revised (recorded with every revision).")
    p.add_argument("--out", help="Default: revision_sheet_v<next version>.csv in the question_stage_dir.")

    p = add("revise-questions", "Validate a filled revision sheet and replace the listed slots' questions.")
    p.add_argument("--sheet", required=True)

    add("pool-snapshot", "Freeze the approved questions' current pools as the next read-only snapshot version.")

    p = add("rewording-sheet", "Write the human writer's reworded-search worksheet for the approved questions.")
    p.add_argument("--out", help="Default: rewording_sheet_pool_v<next>.csv in the question_stage_dir.")

    p = add("apply-rewordings", "Validate the filled rewording sheet, run its BM25 searches and freeze a pool snapshot.")
    p.add_argument("--sheet", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true",
                      help="Run every check (sheet, leakage, export, base pool) and stop: no BM25 search, no write.")
    mode.add_argument("--dry-run", action="store_true", help="Validate, run the BM25 searches in memory and count; "
                                                             "write nothing.")

    p = add("remote-package", "Build blinded, offline grading packages for the primary annotator to grade remotely.")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--questions", action="append", help="Comma-separated approved question IDs (repeatable).")
    which.add_argument("--remaining", action="store_true", help="Every approved question not yet packaged or graded.")
    p.add_argument("--batch-size", type=int, default=10, help="Questions per package with --remaining (default 10).")
    p.add_argument("--for", dest="issue_to", help="The primary annotator (needed when several are registered).")

    p = add("remote-import", "Validate a returned remote grades file and append its grades to the history.")
    p.add_argument("--package", required=True, help="The package ID, e.g. p001.")
    p.add_argument("--file", required=True, help="The returned grades file (kept read-only in the package).")
    p.add_argument("--dry-run", action="store_true", help="Validate only; write nothing.")

    add("remote-status", "List the remote grading packages and whether their grades came back.")

    p = add("remote-withdraw", "Withdraw remote packages that were never imported; their questions can be issued again.")
    p.add_argument("--package", nargs="+", required=True, help="Package IDs, e.g. p002 p003.")
    p.add_argument("--reason", required=True)

    p = add("assign-split", "Split the remaining approved questions between the two registered primary annotators.")
    p.add_argument("--count", action="append", required=True, help="INITIALS=N, once per annotator.")

    p = add("assign-build", "Build one annotator's plain-text grading assignment from the split.")
    p.add_argument("--for", dest="issue_to", required=True, help="The primary annotator.")

    p = add("assign-import", "Validate a returned response file and record its fully graded questions.")
    p.add_argument("--assignment", required=True, help="The assignment ID, e.g. a001.")
    p.add_argument("--file", required=True, help="The returned response file (kept read-only in the assignment).")
    p.add_argument("--dry-run", action="store_true", help="Validate only; write nothing.")

    add("assign-status", "List the plain-text assignments and how many of their questions are recorded.")

    p = add("regrade-init", "Set up the second annotator's blind re-grade of a seeded sample of complete questions.")
    p.add_argument("--regrade-dir", default=REGRADE_DIR)
    p.add_argument("--questions", type=int, default=REGRADE_QUESTIONS)
    p.add_argument("--seed", type=int, default=REGRADE_SEED)

    p = add("regrade-report", "Agreement between the first grades and the completed blind re-grade (written once).")
    p.add_argument("--regrade-dir", default=REGRADE_DIR)

    p = add("history", "Revision log of a question or one of its passages.")
    p.add_argument("query_id")
    p.add_argument("passage_id", nargs="?")

    p = add("audit-init", "Draw the blind audit sample from the AI proposals and set up its workspace.")
    p.add_argument("--proposals-dir", help="Default: ai_proposals_dir in the config.")
    p.add_argument("--audit-dir", help="Default: audit_dir in the config.")
    p.add_argument("--size", type=int, help="Sample size (default: audit_sample_size).")
    p.add_argument("--seed", type=int, help="Sampling seed (default: audit_seed).")

    p = add(
        "audit-init-pool",
        "Draw a blind priority audit of an expanded pool's AI proposals and set up its workspace.",
    )
    p.add_argument("--spec", required=True, help="Audit spec YAML, e.g. configs/eval/audit_pilot_v3.yaml.")

    p = add("audit", "Grade the blind audit sample: AI labels stay hidden; resumes where you stopped.")
    p.add_argument("--audit-dir", help="Default: audit_dir in the config.")
    p.add_argument("--preview-chars", type=int, default=1200)

    p = add("audit-status", "Blind audit progress (shows no AI labels).")
    p.add_argument("--audit-dir", help="Default: audit_dir in the config.")

    p = add("audit-report", "Agreement between the finished audit and the AI proposals.")
    p.add_argument("--audit-dir", help="Default: audit_dir in the config.")
    p.add_argument("--json", action="store_true")

    p = add("dev-labels", "Derive a versioned development label set from AI proposals under a rubric spec.")
    p.add_argument(
        "--spec", required=True,
        help="Spec YAML, e.g. configs/eval/dev_labels_pilot_v2.yaml; a spec naming base_labels_dir derives a "
        "new version from that label set and an expanded pool (e.g. dev_labels_pilot_v3.yaml).",
    )

    p = add("expand-pool", "Add the top results of stored runs to a development label set's pool (no judgments).")
    p.add_argument("--spec", required=True, help="Pool spec YAML, e.g. configs/eval/dev_pool_pilot_v3.yaml.")

    p = add("export-dev", "Export a development label set as dev queries, qrels and provenance (never gold).")
    p.add_argument("--labels", required=True, help="Label set directory, e.g. data/eval/dev_ai/pilot_v2.")
    p.add_argument("--out-dir", help="Default: the label set directory.")

    p = add("export", "Write gold_queries.jsonl, gold_qrels.jsonl, metadata and a coverage report.")
    p.add_argument("--out-dir", help="Default: export_dir in the config.")
    p.add_argument("--include-open", action="store_true", help="Also export questions not marked complete.")
    return parser


_COMMANDS: dict[str, Callable[[_Context], int]] = {
    "init": _cmd_init,
    "add-question": _cmd_add_question,
    "pool": _cmd_pool,
    "search": _cmd_search,
    "show": _cmd_show,
    "annotate": _cmd_annotate,
    "complete": _cmd_status_change("complete"),
    "reopen": _cmd_status_change("open"),
    "withdraw": _cmd_status_change("withdrawn"),
    "status": _cmd_status,
    "leakage-check": _cmd_leakage_check,
    "question-sheet": _cmd_question_sheet,
    "import-questions": _cmd_import_questions,
    "export-questions": _cmd_export_questions,
    "revision-sheet": _cmd_revision_sheet,
    "revise-questions": _cmd_revise_questions,
    "pool-snapshot": _cmd_pool_snapshot,
    "rewording-sheet": _cmd_rewording_sheet,
    "apply-rewordings": _cmd_apply_rewordings,
    "remote-package": _cmd_remote_package,
    "remote-import": _cmd_remote_import,
    "remote-status": _cmd_remote_status,
    "remote-withdraw": _cmd_remote_withdraw,
    "assign-split": _cmd_assign_split,
    "assign-build": _cmd_assign_build,
    "assign-import": _cmd_assign_import,
    "assign-status": _cmd_assign_status,
    "regrade-init": _cmd_regrade_init,
    "regrade-report": _cmd_regrade_report,
    "history": _cmd_history,
    "export": _cmd_export,
    "audit-init": _cmd_audit_init,
    "audit-init-pool": _cmd_audit_init_pool,
    "audit": _cmd_audit,
    "audit-status": _cmd_audit_status,
    "audit-report": _cmd_audit_report,
    "dev-labels": _cmd_dev_labels,
    "expand-pool": _cmd_expand_pool,
    "export-dev": _cmd_export_dev,
}


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        config = load_gold_set_config(args.config)
        corpus = verify_corpus(
            args.corpus or config.corpus_path,
            args.manifest,
            expected_sha256=config.corpus_sha256,
            expected_version=config.corpus_version,
        )
        ctx = _Context(args, config, corpus, Path(args.workspace or config.workspace_dir), input_fn, out)
        return _COMMANDS[args.command](ctx)
    except (GoldSetConfigError, CorpusIntegrityError, CorpusMismatchError) as exc:
        print(f"{PREFIX} error: {exc}", file=err)
        return 2
    except (AnnotationError, GoldExportError, ValueError) as exc:
        print(f"{PREFIX} refused: {exc}", file=err)
        return 1


__all__ = [
    "ANNOTATOR_ENV",
    "DEFAULT_CONFIG",
    "AnnotationSession",
    "AuditSession",
    "GoldSetConfigError",
    "build_parser",
    "load_gold_set_config",
    "main",
]
