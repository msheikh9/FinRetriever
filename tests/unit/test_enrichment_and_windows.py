"""Passage views (finretriever.retrieval.enrichment) and long-passage windows (finretriever.retrieval.windows)."""

import re
from pathlib import Path

import numpy as np
import pytest

from finretriever.data.schemas import Passage
from finretriever.retrieval.biencoder import BiEncoderRetriever
from finretriever.retrieval.bm25 import BM25Retriever
from finretriever.retrieval.enrichment import encoded_field, header_v1, render_passage
from finretriever.retrieval.windows import parent_id, sliding_windows, tail_windows, windows_sha256
from tests.unit.dense_fixtures import CORPUS_SHA256, REVISION, FakeEncoder

BGE = "BAAI/bge-base-en-v1.5"


def apple(text: str = "Net sales rose because of iPhone demand.", pid: str = "p1") -> Passage:
    return Passage(passage_id=pid, filing_id="f1", ticker="AAPL", form_type="10-K", filing_date="2022-10-28",
                   section_title="Management's Discussion and Analysis", text=text,
                   metadata={"company_name": "Apple Inc.", "report_date": "2022-09-24", "item_number": "7"})


def test_header_is_built_from_the_passages_own_metadata_and_the_text_is_unchanged():
    p = apple()
    assert header_v1(p) == ("Company: Apple Inc. (AAPL). Filing: 10-K for fiscal year 2022. "
                            "Section: Item 7. Management's Discussion and Analysis.")
    assert render_passage(p, "header_v1").endswith("\n" + p.text)
    assert render_passage(p) == p.text and p.text == "Net sales rose because of iPhone demand."
    with pytest.raises(ValueError):
        render_passage(p, "summary")


def test_the_default_view_keeps_the_existing_cache_key_and_enriched_gets_its_own(tmp_path: Path):
    assert encoded_field("text") == "passage.text"  # unchanged: existing embedding caches stay valid
    passages = [apple(), apple("Services revenue grew.", "p2")]
    plain = BiEncoderRetriever(BGE, model_revision=REVISION, encoder=FakeEncoder(), cache_dir=tmp_path,
                               corpus_sha256=CORPUS_SHA256)
    plain.index(passages)
    encoder = FakeEncoder()
    enriched = BiEncoderRetriever(BGE, model_revision=REVISION, encoder=encoder, cache_dir=tmp_path,
                                  corpus_sha256=CORPUS_SHA256, passage_view="header_v1")
    enriched.index(passages)
    assert encoder.encoded == [render_passage(p, "header_v1") for p in passages]  # not served from plain's cache
    assert plain.describe()["cache"]["key"]["encoding"]["encoded_field"] == "passage.text"
    assert enriched.describe()["cache"]["key"]["encoding"]["encoded_field"] == "header_v1+passage.text"
    assert plain.describe()["cache"]["dir"] != enriched.describe()["cache"]["dir"]
    hits = enriched.search("iPhone", top_k=2)
    assert {h.passage_id for h in hits} == {"p1", "p2"}  # hits keep the original IDs
    assert np.allclose(enriched.scores("iPhone")[[0, 1]], [h.score for h in sorted(hits, key=lambda h: h.passage_id)])


def test_bm25_view_and_scores_match_search():
    passages = [apple(), apple("Walmart grocery sales.", "p2")]
    bm25 = BM25Retriever(tokenizer="word", backend="python", passage_view="header_v1")
    bm25.index(passages)
    scores = bm25.scores("apple fiscal year 2022")  # only the header names Apple and the year
    assert scores[0] > 0 and scores[1] > 0
    assert [h.passage_id for h in bm25.search("iphone", top_k=1)] == ["p1"]
    assert bm25.describe()["passage_view"] == "header_v1"


class FakeWordPiece:
    """Whitespace words; a word longer than 6 characters becomes a stem and ``##`` continuation pieces."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offsets, tokens = [], []
        for m in re.finditer(r"\S+", text):
            start, word = m.start(), m.group(0)
            pieces = [word[i:i + 6] for i in range(0, len(word), 6)]
            pos = start
            for j, piece in enumerate(pieces):
                tokens.append(piece if j == 0 else "##" + piece)
                offsets.append((pos, pos + len(piece)))
                pos += len(piece)
        return {"input_ids": tokens, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, ids):
        return list(ids)


def test_tail_windows_cover_everything_past_the_head_on_word_boundaries():
    words = [f"w{i}" for i in range(30)] + ["extraordinarily"] + [f"x{i}" for i in range(20)]
    p = apple(" ".join(words))
    tok = FakeWordPiece()
    windows = tail_windows([p], tok, head_tokens=20, size=12, overlap=4)
    n_tokens = len(tok(p.text)["input_ids"])
    assert windows and all(parent_id(w.passage_id) == "p1" for w in windows)
    assert windows[0].metadata["window_start_token"] == 16  # head end 20 minus overlap 4
    assert windows[-1].metadata["window_end_token"] == n_tokens  # nothing past the head is dropped
    for w in windows:
        assert w.text in p.text and not w.text.startswith("##")
        assert not re.match(r"\w", p.text[p.text.index(w.text) - 1]) if p.text.index(w.text) else True
    assert "extraordinarily" in " ".join(w.text for w in windows)  # never split inside the word
    assert tail_windows([apple("short text")], tok, head_tokens=20, size=12, overlap=4) == []


def test_sliding_windows_cover_the_whole_passage_and_hash_is_stable():
    p = apple(" ".join(f"t{i}" for i in range(25)))
    windows = sliding_windows([p], FakeWordPiece(), size=10, overlap=3)
    covered = set()
    for w in windows:
        covered.update(range(w.metadata["window_start_token"], w.metadata["window_end_token"]))
    assert covered == set(range(25)) and [w.metadata["window_index"] for w in windows] == list(range(len(windows)))
    assert windows_sha256(windows) == windows_sha256(sliding_windows([p], FakeWordPiece(), size=10, overlap=3))
    with pytest.raises(ValueError):
        sliding_windows([p], FakeWordPiece(), size=4, overlap=4)
