# Research publication readiness

**Date:** 2026-11-16. **Question:** could FinRetriever's results support an academic paper, and what would one
still need? This is separate from the public GitHub release, which
does not depend on any of the evidence gaps below. No overall readiness score is given: the gaps are of different
kinds and do not add up.

## Verdict

**Not ready for a retrieval-quality paper.** The missing piece is evidence, not engineering. Nothing in the
repository shows on independent, human-labelled questions that A1 or the v2 pipeline retrieves better evidence.
The engineering and the methodology could support a workshop or engineering-track paper about *how* the work was
done, if the evaluation gap is stated as plainly as it is here.

## What the current evidence supports

| Claim | Supported? | Evidence and its limits |
|---|---|---|
| A1 ranks the synthetic validation positives higher than its base model (A0) | Yes, for that set | Pre-registered selection-aware sign-flip test over three seeds, p = 0.0001, adjusted effect +0.365 MRR@10. The 93 questions are synthetic, come from the training distribution and chose the checkpoints, so this is an upper bound |
| Company scoping, and fusion with a fiscal-year boost, raise MRR@10 on the validation set | Yes, for that set | Consistent in all three seeds (p ≤ 0.0001). The same caveats apply, and the year window was chosen on these questions |
| Metadata enrichment and company-year cells improve company and year *alignment* on real questions | Yes, as a metadata diagnostic | 100 human-written questions, label-free. When the parser is right, scoped systems align by construction. This is not relevance |
| Any v2 change improves results on the development set | No | 10 questions with AI-written labels. No primary difference is significant (smallest p 0.50). The pools predate v2, so judged@10 is below 1 |
| A1 or v2 retrieves more relevant evidence for real analyst questions | **No evidence** | The human relevance annotation closed as an incomplete pilot: 45 judgments, all for one question (q_0011). There was no second annotator, so there is no agreement measure |
| Year-role parsing (qu-v2) is more accurate than qu-v1 | Partly | Checked by regression tests and by reviewing every changed parse on the 771 training and validation questions. qu-v1's errors had been seen first, so this is not a blind test ([v2.1 results](retrieval_v2_metadata_hybrid.md#v21-fiscal-year-roles-parser-qu-v2)) |

## What a paper would need

| Requirement | Status | Gap, and the smallest step that would close it |
|---|---|---|
| Independent relevance labels | **Missing** | A complete human-labelled set, written and judged without reference to the systems' outputs. 30–50 questions with pooled judgments from several systems would be enough for paired comparisons |
| Inter-annotator agreement | **Missing** | A second annotator on at least a subset, reporting Cohen's or Krippendorff's agreement on graded relevance |
| An external benchmark | Missing | Results on a public financial retrieval or QA dataset (for example the retrieval setting of FinanceBench or FinQA-style corpora), so that the numbers can be compared with other work |
| Stronger baselines | Missing | A cross-encoder reranker, a larger or newer embedding model, and a domain-adapted embedder. The current baselines are BM25, zero-shot BGE-base and E5-base |
| Scale and diversity | Insufficient | 7 companies, 49 annual reports, fiscal years 2019–2026, 10-K only. A paper would need more companies and sectors, 10-Q and 8-K filings, or a stated reason for the narrow scope |
| Statistical design | Partly in place | Paired bootstraps, sign-flip tests and three training seeds are in place, and primary settings were pre-registered in the configs. Missing: control for multiple comparisons across the 26 v2 experiments, and pre-registration before v1 had been examined |
| Ablations | Partly in place | v2 changes one factor at a time. Enrichment and windows were applied at inference only, and A1 was not retrained with them |
| Training-data provenance | In place | Leakage-safe splits, blocklists, semantic screening against evaluation questions, evidence-reuse equivalents, hashed manifests. The synthetic questions come from a single LLM, which limits their diversity |
| Related work and positioning | Missing | Domain-adapted dense retrieval, temporal and metadata-aware retrieval, hybrid fusion, and financial document QA |
| Reproducibility package | Mostly in place | Configs, hashes, deterministic outputs and pinned environments are in place. The corpus, the training pairs and the checkpoints would have to be released, which the licences permit ([licensing_and_permissions.md](licensing_and_permissions.md)) |

## Suggested order

1. Build the small, complete human-labelled evaluation set with two annotators. This one step converts most
   of the existing results from in-distribution evidence into testable claims.
2. Evaluate the frozen systems on it as they are: A0, A1 (three seeds), BM25, E5 and the release default. Do this
   before any further tuning.
3. Add one external benchmark and one stronger baseline (a cross-encoder reranker).
4. Scope the claim to what those results show. An engineering or methodology paper is the realistic first venue.
