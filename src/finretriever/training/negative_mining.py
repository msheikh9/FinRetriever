"""Hard-negative mining for bi-encoder training.

Given the synthetic (query, positive_passage) pairs, retrieves top-K nearest
passages with a zero-shot encoder and selects difficult-but-not-gold passages
as hard negatives.
"""

# TODO(v1): mine top-K candidates with the zero-shot BGE encoder.
# TODO(v1): filter out the gold positive and near-duplicates above a cosine
#           similarity threshold to avoid false negatives.
# TODO(v1): write data/processed/hard_negatives.jsonl in the schema referenced
#           by configs/train/biencoder_bge_base.yaml.
