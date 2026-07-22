"""
2ND EXPERIMENT

1. Loads sentence pairs from data/sentence_pairs.csv
2. Embeds each unique sentence using multilingual BERT's CLS token
3. Computes cosine similarity for each sentence pair
4. Saves pair-level and category-level CLS results

This is designed to compare against Experiment 1, which used mean pooling.
"""

from pathlib import Path

import pandas as pd

from src.embedding_utils import (
    DEFAULT_MODEL_NAME,
    cosine_similarity,
    create_embedding_lookup,
    load_embedding_model,
)


DATA_PATH = Path("data/sentence_pairs.csv")
RESULTS_DIR = Path("results")

PAIR_RESULTS_PATH = RESULTS_DIR / "pair_similarities_mbert_cls.csv"
CATEGORY_RESULTS_PATH = RESULTS_DIR / "category_summary_mbert_cls.csv"


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Make sure the dataset has the columns this experiment expects.
    """
    required_columns = {
        "id",
        "sentence_1",
        "sentence_2",
        "language_1",
        "language_2",
        "category",
        "expected_similarity",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading sentence pair dataset...")
    df = pd.read_csv(DATA_PATH)
    validate_input_data(df)

    print(f"Loaded {len(df)} sentence pairs.")
    print(f"Categories: {df['category'].nunique()}")

    all_sentences = list(df["sentence_1"]) + list(df["sentence_2"])
    unique_sentences = list(dict.fromkeys(all_sentences))

    print(f"Unique sentences to embed: {len(unique_sentences)}")

    print(f"\nLoading model: {DEFAULT_MODEL_NAME}")
    tokenizer, model, device = load_embedding_model()
    print(f"Using device: {device}")

    print("\nCreating CLS-token sentence embeddings...")
    embedding_lookup = create_embedding_lookup(
        unique_sentences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=8,
        pooling_method="cls",
    )

    print("Computing cosine similarities...")

    similarities = []

    for _, row in df.iterrows():
        sentence_1 = row["sentence_1"]
        sentence_2 = row["sentence_2"]

        embedding_1 = embedding_lookup[sentence_1]
        embedding_2 = embedding_lookup[sentence_2]

        similarity = cosine_similarity(embedding_1, embedding_2)
        similarities.append(similarity)

    df["similarity"] = similarities
    df["similarity"] = df["similarity"].round(4)

    print("\nCreating category summary...")

    category_summary = (
        df.groupby(["category", "expected_similarity"])["similarity"]
        .agg(
            mean_similarity="mean",
            std_similarity="std",
            min_similarity="min",
            max_similarity="max",
            count="count",
        )
        .reset_index()
        .sort_values("mean_similarity", ascending=False)
    )

    numeric_columns = [
        "mean_similarity",
        "std_similarity",
        "min_similarity",
        "max_similarity",
    ]

    category_summary[numeric_columns] = category_summary[numeric_columns].round(4)

    df.to_csv(PAIR_RESULTS_PATH, index=False)
    category_summary.to_csv(CATEGORY_RESULTS_PATH, index=False)

    print(f"\nSaved pair-level CLS results to: {PAIR_RESULTS_PATH}")
    print(f"Saved category CLS summary to: {CATEGORY_RESULTS_PATH}")

    print("\nCLS category summary:")
    print(category_summary.to_string(index=False))

    print("\nHighest-similarity CLS pairs:")
    highest_pairs = df.sort_values("similarity", ascending=False).head(5)
    print(
        highest_pairs[
            [
                "id",
                "sentence_1",
                "sentence_2",
                "category",
                "expected_similarity",
                "similarity",
            ]
        ].to_string(index=False)
    )

    print("\nLowest-similarity CLS pairs:")
    lowest_pairs = df.sort_values("similarity", ascending=True).head(5)
    print(
        lowest_pairs[
            [
                "id",
                "sentence_1",
                "sentence_2",
                "category",
                "expected_similarity",
                "similarity",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()