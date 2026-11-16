# Licensing and permissions

**Date:** 2026-11-16. This report covers what FinRetriever uses and what it might redistribute. Each item says
what the primary source states, quoted where possible, and what remains open. It is a summary of published
terms, not legal advice. Publicly accessible material is not assumed to be redistributable: each permission
below comes from the owner's own terms.

## The initial public release

The terms below permit more than the initial public release contains. The project owner chose a narrower first
release:

| Included | Not included (kept in the project's private archive) |
|---|---|
| Source code, tests, configurations, documentation, the pinned corpus configuration, experiment results (metrics and passage IDs) | The SEC corpus (rebuild it from EDGAR with `configs/data/corpus_v1_pinned.yaml`), the fine-tuned checkpoints, the synthetic training and validation data, the development labels and audits, and all gold-set v1 human-annotation records |

The documentation quotes short excerpts of 10-K text, with their accession numbers, as the SEC's policy permits.

## Summary

| Material | Licence or terms | Redistribution | Status |
|---|---|---|---|
| FinRetriever code and documentation | MIT (`LICENSE`) | Yes | ✅ Confirmed |
| Base model `BAAI/bge-base-en-v1.5` (revision `a5beb1e3…`) | MIT | Yes, with its licence notice | ✅ Confirmed |
| Fine-tuned A1 checkpoints (3 seeds) | A derivative of an MIT model; trained on the synthetic questions and SEC text below | Yes, with BGE's MIT notice, subject to the two rows below | ✅ Permitted by the terms below; publishing is your decision |
| SEC 10-K text (corpus, excerpts in evaluation files) | SEC: "public information" that "may be copied or further distributed … without the SEC's permission" | Yes, with attribution, without SEC logos or trademarks | ✅ Confirmed (see the caveat below) |
| Synthetic training and validation questions (generated with `qwen/qwen3.8-27b` through Groq) | Groq Services Agreement: the customer retains the IP in Outputs. Model licence: Apache-2.0 | Yes; keep them labelled as AI-generated | ✅ Confirmed (see the caveats below) |
| AI-written development questions and labels (written with Claude) | Anthropic Consumer Terms §4: rights in Outputs are assigned to the user | Yes | ✅ Confirmed for the consumer terms; check them if another Anthropic plan was used |
| `intfloat/e5-base-v2` (v1 baseline only, never redistributed) | MIT | Not needed | ✅ Confirmed |
| Third-party Python packages (63 locked) | MIT, BSD, Apache-2.0, ISC, PSF; MPL-2.0 for three | Not redistributed (only listed in the requirement files) | ✅ Compatible with MIT |
| Human-written evaluation questions and human relevance grades | Authored by the project's contributors; no licence or release agreement on record | **Only with the contributors' permission** | ❌ **Unresolved: consent needed** |

## 1. The base model: `BAAI/bge-base-en-v1.5`

- **Licence:** MIT. The model card at the pinned revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` declares
  `license: mit`, and the Hugging Face model record carries the tag `license:mit`
  ([model page](https://huggingface.co/BAAI/bge-base-en-v1.5)).
- **The model card says:** "FlagEmbedding is licensed under the MIT License. The released models can be used for
  commercial purposes free of charge."
- **What MIT requires of a redistribution:** the copyright notice and the permission notice must be included.
  Ship BGE's MIT licence text, from the [FlagEmbedding repository](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE),
  with any checkpoint release.

## 2. The fine-tuned checkpoints (A1, seeds 42, 1337 and 2026)

- **What they are:** BGE-base weights after 40–55 contrastive steps. A derivative of an MIT-licensed model, which
  MIT permits to "use, copy, modify, merge, publish, distribute, sublicense, and/or sell".
- **What they were trained on:** the 678 synthetic training pairs (§4), whose passages are SEC 10-K text (§3).
  Both permit redistribution, so nothing found restricts publishing the checkpoints.
- **To publish them** (for example on the Hugging Face Hub):
  - include BGE's MIT notice;
  - add a model card stating the base model and revision, the training data, the fixed selection rule, and the
    evaluation limits (synthetic validation questions, no human relevance evaluation);
  - choose a licence for the weights (MIT is consistent with the base model).
- **Approval:** publishing is the project owner's decision.

## 3. SEC EDGAR filings

**What the SEC states.**
- **Website Dissemination policy** ([sec.gov/privacy.htm](https://www.sec.gov/privacy.htm)):
  - "Information presented on sec.gov is considered public information and may be copied or further distributed
    by users of the web site without the SEC's permission."
  - "Please consider appropriate citation to the SEC as the source."
  - The SEC seal, logos and artwork may not be used. "SEC", the EDGAR logo and the EDGAR names are registered
    trademarks: they need a licence for use in a business name, but may be mentioned in text without implying
    affiliation or approval.
- **Webmaster FAQ** ([webmaster FAQ](https://www.sec.gov/files/about/webmaster-faq.htm)):
  - "All Government-created content on sec.gov and EDGAR public filing content are free to access and reuse."
  - The exceptions it names are things like the stock art used on sec.gov pages.
- **Accessing EDGAR Data** ([fair-access page](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)):
  governs downloading. The maximum is 10 requests per second, with a declared User-Agent. The downloader enforces
  both: 0.2 s between requests, and `SEC_USER_AGENT` is required.

**What follows.**
- The corpus (10,137 passages from 49 10-K filings) and the filing excerpts in the evaluation files may be
  redistributed.
- Cite SEC EDGAR as the source. Every passage carries its accession number and CIK.
- Do not use SEC or EDGAR logos, and do not imply endorsement.

**Caveat.** The SEC's statement is the publisher's position on content it distributes. Filings are written by
the companies. The corpus holds only the text of 10-K Items, with no images, exhibits or logos. The SEC's "free
to access and reuse" covers EDGAR filing content, so no open question remains for the text. If extra caution is
wanted, publish the scripts and the pinned configuration (`configs/data/corpus_v1_pinned.yaml`) instead of the
corpus. Anyone can then rebuild the identical corpus (SHA-256 `ff9dadf3…`) from EDGAR.

## 4. Synthetic training and validation questions (Groq, `qwen/qwen3.8-27b`)

- **Generation:** November 3–8, 2026, through GroqCloud's API (`configs/train/data_prep_v3.yaml`;
  `provider: groq`, `model: qwen/qwen3.8-27b`).
- **Groq Services Agreement** ([agreement](https://console.groq.com/docs/legal/services-agreement), "Last
  Modified: June 22, 2026", so the version in force during generation). Groq states it is "the primary contract
  governing your use of GroqCloud" ([contractual framework](https://console.groq.com/docs/legal/contractual-framework-overview)).
  - §1: "Outputs are Customer Data."
  - §8.1: "Customer retains all Intellectual Property Rights in Customer Data (including in Inputs and Outputs)."
  - §4.1: the customer complies with the "AI Model Terms" of each model, which are the model provider's terms.
  - §6.3(e): the customer may not use the service to "develop or improve products, services, or other offerings
    that are similar or compete with the Cloud Services". A financial retriever is not a similar or competing
    offering.
  - §6.3(f): no removal or alteration of "transparency or provenance information" that identifies output as
    AI-generated.
  - The agreement has no clause that restricts redistributing Outputs.
- **The model's terms:** the official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) record on
  Hugging Face has `license: apache-2.0` (created 2026-08-05). Apache-2.0 places no restriction on generated
  output.

**What follows.**
- The 678 training pairs and the 93 validation questions may be redistributed.
- Keep them marked as AI-generated (§6.3(f)). Their records already name the provider and model; publish that
  provenance with them.

**Caveats.**
- `qwen/qwen3.8-27b` is Groq's identifier. The Apache-2.0 licence was confirmed for Qwen's official release of
  Qwen3.8-27B, which Groq serves under that name.
- Whether purely AI-generated text is protected by copyright at all is unsettled. That does not limit
  redistribution.

## 5. AI-written development data (Claude)

- **What it covers:** the 10 development questions and their provisional labels (`data/eval/dev_ai/`) and the
  AI-proposed audit grades, which were written with Claude (`author`/`annotator` "claude").
- **Anthropic's [Consumer Terms](https://www.anthropic.com/legal/consumer-terms)** (effective Oct 8, 2025), §4:
  "we assign to you all of our right, title, and interest—if any—in Outputs". §3 bars only using the services "to
  develop any products or services that compete with" Anthropic's.
- **If another plan applies:** check its terms instead, for example the Commercial Terms of an organisation plan.
- **Human review:** a human reviewer's grades and approvals are recorded with these files, under initials. They
  fall under §6.

## 6. Human-written evaluation questions and grades: unresolved

- **The material:**
  - the 100 human-written questions, with their rewordings and revisions;
  - the human relevance grades;
  - the review decisions.
- **Where it is:** `data/eval/gold_v1/`, `data/eval/annotation/`, part of `data/eval/dev_ai/` and
  `data/eval/audit/`, three questions quoted in `docs/retrieval_analysis_v1.md`, and one question in each
  `parser_accuracy.json`.
- **What the records show:** the contributors wrote and graded them, and are recorded by initials. No licence,
  release form or publication consent is recorded. Their authorship gives them rights in the questions, and the
  initials identify them to anyone who knows them.
- **Consequence:** this material is not published until each contributor agrees.

## 7. Third-party Python dependencies

The 63 packages in `requirements/lock-*.txt` use these licences:

- **MIT, BSD (2- and 3-clause), Apache-2.0, ISC and PSF-2.0:** all but three, including torch (BSD-3-Clause),
  transformers, sentence-transformers, tokenizers, safetensors, huggingface-hub, requests and rank-bm25
  (Apache-2.0), and numpy, scipy, scikit-learn and lxml (BSD).
- **MPL-2.0:** `certifi`, `pathspec` (via black) and `tqdm` (MPL-2.0 AND MIT). MPL-2.0 applies to modified copies
  of those packages' own files.

FinRetriever lists these packages in its requirement files but does not copy their code, so none of these licences
places a condition on FinRetriever's MIT licence.
