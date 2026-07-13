# Multilingual Embedding Experiment

This is a simple, mechanistic interpretability inspired exploration of multilingual sentence embeddings. This project uses a multilingual transformer model and analyzes English, Chinese, and Japanese sentence embeddings that are related through translation, negation, synonyms, or word order.

## Main Question

How does a multilingual embedding model represent semantic similarity across languages, and does it distinguish meaning-changing edits such as negation or argument reversal?

## Experiment Plan

1. Generate a small dataset of sentence pairs, grouped by category (listed below)
2. Extract embeddings using a multilingual transformer model
3. Compare sentence pairs using cosine similarity

## Sentence Categories

- Cross-lingual same meaning
- Cross-lingual negation mismatch
- English negation
- Chinese negation
- Japanese negation
- Synonym substitution
- Word-order or role reversal
- Unrelated sentence pairs