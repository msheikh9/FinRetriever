"""Offline ranking of validation source passages by how likely a question generated from them passes the
validation set's evidence rule: a prediction for sampling plans, never a relevance label.

``validation_set`` excludes a question when any passage outside the validation split holds its evidence
(``EvidenceLocator.locate``): the quote or one of its sentences, the answer's figure under the evidence's
label, a labelled evidence figure, or the answer's amount anywhere in the company's passages. A plan cannot
know the question, so it measures two things on the passage's visible text:

* ``figure_isolation``: how many of its significant figures (``evidence_reuse.figures``) no passage of the
  company outside the passage's split holds, by the locator's own amount test (``FigureSet.holds_amount``)
  on full text: unique financial figures, the answers a question is most likely to use;
* the sentence share (``validation_set.isolated_evidence_share``): the share of its prose sentences that no
  such passage holds by any locator match. It never sees table cells or answer figures, which is why it
  over-predicted ``val_pilot_v2e`` (docs/validation_pilot_v2e_batch1_audit.md §8).

The two are mixed by how often the generator quotes a significant figure in each passage kind
(``figure_rate``, measured on the calibration questions) and calibrated by a logistic fit on the mixture's
logit. The fit has two parameters, ridge-regularised toward the uncalibrated mixture, so a few dozen questions
cannot over-fit it, and its leave-one-out error is recorded. Expected usable questions are the calibrated
probability times the observed generator acceptance and review acceptance.

A calibration is a write-once JSON file (``calibrate-predictor``). It holds every observation (plan, position,
passage, the question's SHA-256, the measurements and the outcome), the files they came from by SHA-256, the
fit and its metrics. Calibrating reads generation records and AI-review decisions of validation pilots, never
evaluation labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finretriever.training.carryover import question_sha256
from finretriever.training.evidence_reuse import EvidenceLocator, figures
from finretriever.training.query_generation import SourcePassage
from finretriever.training.sampling_plan import passage_kind
from finretriever.training.validation_set import outside_evidence_reasons, pair_id_for

PREDICTOR_VERSION = "evidence-pass-v1"
CALIBRATION_KIND = "evidence_pass_calibration"
CALIBRATION_FORMAT = 1
KINDS = ("narrative", "table")
LOGIT_BOUND = 0.02  # scores are clipped to [0.02, 0.98] before the logit
RIDGE = 1.0  # pulls the fit toward intercept 0 and slope 1, the uncalibrated mixture
NOTICE = (
    "A ranking aid for validation sampling plans, fitted on the outcomes of AI-reviewed synthetic validation "
    "questions. It predicts how likely a question passes the evidence rule; it is never a relevance label, a "
    "gold label or evaluation data."
)


class PredictionError(ValueError):
    """A calibration cannot be built, written or read; nothing is written."""


def figure_isolation(
    source: SourcePassage, locator: EvidenceLocator, rows: Mapping[str, Mapping[str, Any]]
) -> tuple[int, int]:
    """The significant figures of ``source``'s visible text, and how many of them no passage of its company
    outside its split holds (full text, by amount)."""
    found = figures(source.visible_text, source.ticker, locator.index.settings, declared=locator.declared(source.passage_id))
    split = rows[source.passage_id]["split"]
    unique = 0
    for figure in found:
        holders = locator.amount_holders(figure, source.ticker) - {source.passage_id}
        if not any(rows[q]["split"] != split and locator.figures_of(q, False).holds_amount(figure) for q in holders):
            unique += 1
    return len(found), unique


def mixture(kind: str, n_figures: int, unique: int, sentence_share: float, figure_rate: Mapping[str, float]) -> float:
    """The uncalibrated pass score: the unique-figure share where the generator quotes a figure, the
    sentence share otherwise (the sentence share alone for a passage with no significant figure)."""
    if not n_figures:
        return sentence_share
    rate = figure_rate[kind]
    return rate * unique / n_figures + (1 - rate) * sentence_share


def _logit(p: float) -> float:
    p = min(max(p, LOGIT_BOUND), 1 - LOGIT_BOUND)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fit(scores: Sequence[float], outcomes: Sequence[bool], ridge: float = RIDGE) -> tuple[float, float]:
    """Intercept and slope of a logistic fit of ``outcomes`` on the logit of ``scores``, with a ridge penalty
    toward (0, 1); Newton steps, deterministic."""
    if len(scores) != len(outcomes) or not scores:
        raise PredictionError("a fit needs one outcome per score, and at least one")
    xs = [_logit(s) for s in scores]
    a, b = 0.0, 1.0
    for _ in range(100):
        ga, gb, haa, hab, hbb = ridge * a, ridge * (b - 1), ridge, 0.0, ridge
        for x, y in zip(xs, outcomes, strict=True):
            p = _sigmoid(a + b * x)
            w = p * (1 - p)
            ga, gb = ga + p - y, gb + (p - y) * x
            haa, hab, hbb = haa + w, hab + w * x, hbb + w * x * x
        det = haa * hbb - hab * hab
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-12:
            break
    return a, b


def auc(scores: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    """The probability that a passing question scores above a failing one (ties count half)."""
    pos = [s for s, y in zip(scores, outcomes, strict=True) if y]
    neg = [s for s, y in zip(scores, outcomes, strict=True) if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def brier(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=True)) / len(outcomes)


@dataclass(frozen=True)
class Calibration:
    """A written calibration, as a plan uses it."""

    figure_rate: dict[str, float]
    intercept: float
    slope: float
    generator_acceptance: float
    review_acceptance: float
    sha256: str
    path: str

    def probability(self, raw: float) -> float:
        """The calibrated probability that a question passes the evidence rule, from the mixture score."""
        return _sigmoid(self.intercept + self.slope * _logit(raw))

    def usable(self, probability: float) -> float:
        """Expected usable questions from one request: generated, passing the evidence rule, accepted."""
        return self.generator_acceptance * probability * self.review_acceptance

    def method(self) -> str:
        return (
            f"{PREDICTOR_VERSION}: calibrated probability that a question on the passage passes the validation "
            f"evidence rule, from unique significant figures and isolated sentences (calibration {self.path}, "
            f"SHA-256 {self.sha256[:12]})"
        )


def predict(
    sources: Sequence[SourcePassage],
    locator: EvidenceLocator,
    rows: Mapping[str, Mapping[str, Any]],
    calibration: Calibration,
    sentence_shares: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """For each source: its figures, unique figures, sentence share, mixture score and calibrated probability."""
    out = {}
    for source in sources:
        n, unique = figure_isolation(source, locator, rows)
        share = sentence_shares[source.passage_id]
        raw = mixture(passage_kind(source), n, unique, share, calibration.figure_rate)
        out[source.passage_id] = {
            "figures": n,
            "unique_figures": unique,
            "sentence_share": share,
            "raw": round(raw, 4),
            "probability": round(calibration.probability(raw), 4),
        }
    return out


def observe(
    *,
    pilot: str,
    planned: Sequence[SourcePassage],
    records: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    locator: EvidenceLocator,
    rows: Mapping[str, Mapping[str, Any]],
    sentence_shares: Mapping[str, float],
) -> list[dict[str, Any]]:
    """One observation for each planned passage with a done generation record, in plan order: its
    measurements and, when the generator accepted a question, whether the question passed the evidence rule
    (``validation_set.outside_evidence_reasons``, the validation set's own test) and the AI review's decision
    (``decisions`` by pair ID, as ``load_validation_decisions`` reads them)."""
    out = []
    for position, source in enumerate(planned, start=1):
        record = records.get(source.passage_id)
        if record is None or record.get("status") != "done":
            continue
        n, unique = figure_isolation(source, locator, rows)
        row: dict[str, Any] = {
            "pilot": pilot,
            "plan_position": position,
            "passage_id": source.passage_id,
            "ticker": source.ticker,
            "kind": passage_kind(source),
            "figures": n,
            "unique_figures": unique,
            "sentence_share": sentence_shares[source.passage_id],
            "generated": bool(record["accepted"]),
        }
        if record["accepted"]:
            question = record["accepted"][0]
            decision = decisions.get(pair_id_for(position))
            if decision is not None and decision["question_sha256"] != question_sha256(question["question"]):
                raise PredictionError(f"{pilot} position {position}: the decision names another question")
            equivalents = locator.locate(question["evidence"], question["answer"], source.ticker, source.passage_id)
            row.update({
                "question_sha256": question_sha256(question["question"]),
                "quote_has_figure": bool(figures(question["evidence"], source.ticker, locator.index.settings)),
                "evidence_pass": not outside_evidence_reasons(equivalents, rows),
                "review": None if decision is None else decision["decision"],
            })
        out.append(row)
    return out


def build_calibration(observations: Sequence[Mapping[str, Any]], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The calibration document: generator figure rates by kind, the fit, acceptance rates and metrics."""
    generated = [o for o in observations if o["generated"]]
    if not generated or not any(o["evidence_pass"] for o in generated) or all(o["evidence_pass"] for o in generated):
        raise PredictionError("a calibration needs generated questions that both pass and fail the evidence rule")
    rate = {}
    for kind in KINDS:  # a kind with no generated question takes the rate over every generated question
        of_kind = [o for o in generated if o["kind"] == kind] or generated
        rate[kind] = round(sum(o["quote_has_figure"] for o in of_kind) / len(of_kind), 4)

    def raw(o: Mapping[str, Any]) -> float:
        return mixture(o["kind"], o["figures"], o["unique_figures"], o["sentence_share"], rate)

    scores = [raw(o) for o in generated]
    outcomes = [bool(o["evidence_pass"]) for o in generated]
    a, b = fit(scores, outcomes)
    loo = []
    for i in range(len(scores)):
        ai, bi = fit(scores[:i] + scores[i + 1 :], outcomes[:i] + outcomes[i + 1 :])
        loo.append(_sigmoid(ai + bi * _logit(scores[i])))
    base = sum(outcomes) / len(outcomes)
    passed = [o for o in generated if o["evidence_pass"] and o["review"] is not None]
    if not passed:
        raise PredictionError("no reviewed question passed the evidence rule; the review acceptance is unknown")
    by_pilot = {}
    for pilot in sorted({o["pilot"] for o in observations}):
        idx = [i for i, o in enumerate(generated) if o["pilot"] == pilot]
        of_pilot = [o for o in observations if o["pilot"] == pilot]
        by_pilot[pilot] = {
            "passages": len(of_pilot),
            "generated": len(idx),
            "evidence_pass": sum(outcomes[i] for i in idx),
            "predicted_pass_leave_one_out": round(sum(loo[i] for i in idx), 2),
            "usable": sum(1 for o in of_pilot if o["generated"] and o["evidence_pass"] and o["review"] == "accept"),
        }
    return {
        "kind": CALIBRATION_KIND,
        "format_version": CALIBRATION_FORMAT,
        "predictor": PREDICTOR_VERSION,
        "notice": NOTICE,
        "sources": list(sources),
        "figure_rate": rate,
        "fit": {"intercept": round(a, 6), "slope": round(b, 6), "ridge": RIDGE, "logit_bound": LOGIT_BOUND},
        "rates": {
            "generator_acceptance": round(len(generated) / len(observations), 4),
            "review_acceptance": round(sum(o["review"] == "accept" for o in passed) / len(passed), 4),
        },
        "metrics": {
            "questions": len(generated),
            "auc_mixture": round(auc(scores, outcomes) or 0.0, 4),
            "auc_sentence_share": round(auc([o["sentence_share"] for o in generated], outcomes) or 0.0, 4),
            "brier_leave_one_out": round(brier(loo, outcomes), 4),
            "brier_constant": round(brier([base] * len(outcomes), outcomes), 4),
            "by_pilot": by_pilot,
        },
        "observations": [dict(o) for o in observations],
    }


def write_calibration(path: str | Path, doc: Mapping[str, Any]) -> bool:
    """Write the calibration; False if an identical copy exists. A different existing file is refused."""
    path = Path(path)
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise PredictionError(f"{path} already holds a different calibration; calibrations are never changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def load_calibration(path: str | Path) -> Calibration:
    path = Path(path)
    if not path.is_file():
        raise PredictionError(f"no calibration at {path}; write one with calibrate-predictor")
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("kind") != CALIBRATION_KIND or doc.get("format_version") != CALIBRATION_FORMAT:
        raise PredictionError(f"{path} is not a format-{CALIBRATION_FORMAT} evidence-pass calibration")
    if doc.get("predictor") != PREDICTOR_VERSION:
        raise PredictionError(f"{path} calibrates {doc.get('predictor')!r}, not {PREDICTOR_VERSION}")
    if sorted(doc["figure_rate"]) != sorted(KINDS):
        raise PredictionError(f"{path} needs a figure rate for each of {', '.join(KINDS)}")
    return Calibration(
        figure_rate=dict(doc["figure_rate"]),
        intercept=doc["fit"]["intercept"],
        slope=doc["fit"]["slope"],
        generator_acceptance=doc["rates"]["generator_acceptance"],
        review_acceptance=doc["rates"]["review_acceptance"],
        sha256=hashlib.sha256(raw).hexdigest(),
        path=str(path),
    )


def describe(doc: Mapping[str, Any]) -> list[str]:
    """Summary lines for a calibration document."""
    m, f, r = doc["metrics"], doc["fit"], doc["rates"]
    kinds = Counter(o["kind"] for o in doc["observations"] if o["generated"])
    return [
        f"{m['questions']} generated questions ({dict(sorted(kinds.items()))}); quotes holding a significant figure: "
        f"{doc['figure_rate']}",
        f"fit: intercept {f['intercept']:.3f}, slope {f['slope']:.3f} (ridge {f['ridge']} toward 0 and 1)",
        f"AUC {m['auc_mixture']:.3f} (sentence share alone {m['auc_sentence_share']:.3f}); leave-one-out Brier "
        f"{m['brier_leave_one_out']:.3f} against {m['brier_constant']:.3f} for a constant",
        f"generator acceptance {r['generator_acceptance']:.2f}; review acceptance of evidence-passing questions "
        f"{r['review_acceptance']:.2f}",
        *(f"{p}: {v}" for p, v in m["by_pilot"].items()),
    ]


__all__ = [
    "CALIBRATION_FORMAT",
    "CALIBRATION_KIND",
    "KINDS",
    "NOTICE",
    "PREDICTOR_VERSION",
    "Calibration",
    "PredictionError",
    "auc",
    "brier",
    "build_calibration",
    "describe",
    "figure_isolation",
    "fit",
    "load_calibration",
    "mixture",
    "observe",
    "predict",
    "write_calibration",
]
