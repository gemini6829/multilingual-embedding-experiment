# Multilingual Embedding Experiment

This is a simple, mechanistic interpretability inspired exploration of multilingual sentence embeddings. This project uses a multilingual transformer model and analyzes English, Chinese, and Japanese sentence embeddings that are related through translation, negation, synonyms, or word order.

## Main Question

How does a multilingual embedding model represent semantic similarity across languages, and does it distinguish meaning-changing edits such as negation or argument reversal?

## Experiment Plan

1. Generate a small dataset of sentence pairs, grouped by category (listed below)
2. Extract embeddings using a multilingual transformer model
3. Compare sentence pairs using cosine similarity

## Dataset

The dataset contains 36 sentence pairs across English, Chinese, and Japanese.

The categories are:

| Category | Description |
|---|---|
| `cross_lingual_same_meaning` | Translated sentence pairs with the same meaning |
| `cross_lingual_negation_mismatch` | Cross-lingual pairs where one sentence is negated |
| `english_negation` | English affirmative/negative sentence pairs |
| `chinese_negation` | Chinese affirmative/negative sentence pairs |
| `japanese_negation` | Japanese affirmative/negative sentence pairs |
| `synonym_substitution` | English pairs with similar meaning but different word choice |
| `word_order_argument_swap` | English pairs where the subject/object roles are reversed |
| `unrelated` | Sentence pairs with unrelated meanings |

## Initial Predictions

| Category | Expected Similarity | Reason |
|---|:---:|---|
| `cross_lingual_same_meaning` | High | The two sentences are translations with the same meaning. |
| `synonym_substitution` | High | The wording changes, but the meaning is mostly preserved. |
| `cross_lingual_negation_mismatch` | Low | The meaning and language both change. |
| `english_negation` | Medium | The meaning changes but the words remain similar. |
| `chinese_negation` | Medium | Same as above. |
| `japanese_negation` | Medium | Same as above. |
| `word_order_argument_swap` | Low | Reversing words changes sentence meaning. |
| `unrelated` | Low | Completely different meanings and words. |

## 1st Experiment

I used `bert-base-multilingual-cased` from Hugging Face Transformers and for each sentence:

1. Tokenized the sentence.
2. Passed it through multilingual BERT.
3. Used mean pooling to create one sentence embedding.
4. Compared sentence embeddings using cosine similarity.

### Results

| Category | Expected Similarity | Mean Similarity |
|---|:---:|:---:|
| `cross_lingual_same_meaning` | High | 0.5572 |
| `synonym_substitution` | High | 0.8128 |
| `cross_lingual_negation_mismatch` | Low | 0.5279 |
| `english_negation` | Medium | 0.8859 |
| `chinese_negation` | Medium | 0.8674 |
| `japanese_negation` | Medium | 0.9142 |
| `word_order_argument_swap` | Low | 0.9868 |
| `unrelated` | Low | 0.4758 |

The categories above are listed in the order of expected similarity from high to low. Unexpectedly, the highest average similarity came from the word-order/argument-swap category. For example:

    The boy chased the girl.
    The girl chased the boy.

These two sentences have different meanings because the roles are reversed. However, they contain nearly the same words despite different ordering, and the mean-pooled mBERT embeddings gave them very high similarity.

Negation pairs were also much higher than expected. For example:

    The cat is sleeping.
    The cat is not sleeping.

These two sentences differ logically, but they still share most of their words. The model representation remained highly similar.

### Visualization

![Mean cosine similarity by category](figures/category_mean_similarity.png)

The bar chart shows that several categories predicted to have low similarity, especially word-order/argument swaps and negation pairs, received the highest cosine similarity scores. This suggests that mean-pooled mBERT embeddings may be more sensitive to word overlap and topic similarity than to precise logical meaning.

![Pair-level cosine similarities by category](figures/pair_similarity_color.png)

The pair-level plot shows that there is some variation within each type of sentence pair. However, the high similarity scores for negation and role-reversal pairs were not caused by a single outlier and most examples in those categories clustered at high similarity.

### Interpretation

These results suggest that mean-pooled multilingual BERT embeddings capture surface-level or topical similarity more strongly than precise logical meaning.

In this setup, high cosine similarity did not always mean that two sentences had the same meaning. Sentences with nearly identical words often received very high similarity, even when their meanings changed through negation or the words switched places.

For the cross-lingual comparisons, same-meaning translated pairs scored only slightly higher than cross-lingual negation mismatches (0.5572 vs. 0.5279). This suggests that basic mean-pooled mBERT embeddings may not be enough to clearly distinguish between the two categories.

### Conclusion

A pair of sentences can receive high similarity because they share words, topics, or structure, even if their precise meaning is different.

### Experiment Limitations

- The dataset is small and uses simple language.
- The languages only include English, Chinese, and Japanese.
- The method uses mean pooling over token embeddings.
- Cosine similarity gives behavioral evidence, not a causal explanation of the model.

## 2nd Experiment: CLS-Token Embeddings

After Experiment 1, I tested whether the results were mainly effected by the mean-pooling method. I reran the same sentence-pair dataset using the CLS-token embedding from `bert-base-multilingual-cased` instead of mean pooling. The prediction was that CLS-token embeddings would be more sensitive to meaning-changing edits such as negation and role reversal.

### Results

| Category | Expected Similarity | Mean Pooling | CLS Token |
|---|---:|---:|---:|
| `word_order_argument_swap` | Low | 0.9868 | 0.9945 |
| `japanese_negation` | Low | 0.9142 | 0.9262 |
| `english_negation` | Low | 0.8859 | 0.9604 |
| `chinese_negation` | Low | 0.8674 | 0.9777 |
| `synonym_substitution` | High | 0.8128 | 0.9588 |
| `cross_lingual_same_meaning` | High | 0.5572 | 0.8244 |
| `cross_lingual_negation_mismatch` | Low | 0.5279 | 0.7747 |
| `unrelated` | Low | 0.4758 | 0.7898 |

CLS-token embeddings did not fix the unexpected behavior from Experiment 1, instead it increased the cosine similarity for every category, including categories that were expected to have low similarity.

The role-reversal category remained the highest-scoring category, with a mean similarity of 0.9945. Negation categories also remained very high. The cross-lingual same-meaning category improved from 0.5572 to 0.8244, but the cross-lingual negation mismatch category also increased from 0.5279 to 0.7747. One surprising result was that unrelated sentence pairs also became much more similar, increasing from 0.4758 with mean pooling to 0.7898 with CLS. 

### Interpretation

The consistent high scores of the role-reversal and negation categories suggests that the previous results in Experiment 1 were not only caused by mean pooling. All categories showed an increase, which means CLS embeddings created higher cross-lingual similarity overall, but still did not clearly separate same-meaning translations from negation mismatches. This suggests that CLS-token embeddings may compress many sentence pairs into a high-similarity range, making cosine similarity less useful for distinguishing precise semantic relationships in this setup.

### Next Steps

1. Expanding the dataset with more sentence pairs.
2. Testing a sentence-transformer model trained specifically for semantic similarity.
3. Adding more controlled examples for negation and role reversal.