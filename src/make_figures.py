from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""
1st experiment visualizations
"""

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

CATEGORY_RESULTS_PATH = RESULTS_DIR / "category_summary_mbert.csv"
PAIR_RESULTS_PATH = RESULTS_DIR / "pair_similarities_mbert.csv"

CATEGORY_FIGURE_PATH = FIGURES_DIR / "category_mean_similarity.png"
PAIR_FIGURE_PATH = FIGURES_DIR / "pair_similarity_by_category.png"
COLOR_FIGURE_PATH = FIGURES_DIR / "pair_similarity_color.png"


def clean_category_name(category: str) -> str:
    """
    Convert category names from snake_case into readable labels.
    """
    replacements = {
        "cross_lingual_same_meaning": "Cross-lingual\nsame meaning",
        "cross_lingual_negation_mismatch": "Cross-lingual\nnegation mismatch",
        "english_negation": "English\nnegation",
        "chinese_negation": "Chinese\nnegation",
        "japanese_negation": "Japanese\nnegation",
        "synonym_substitution": "Synonym\nsubstitution",
        "word_order_argument_swap": "Word-order /\nargument swap",
        "unrelated": "Unrelated",
    }

    return replacements.get(category, category.replace("_", " ").title())


def make_category_bar_chart() -> None:
    """
    Make a horizontal bar chart of mean similarity by category.
    """
    summary = pd.read_csv(CATEGORY_RESULTS_PATH)

    summary = summary.sort_values("mean_similarity", ascending=True)
    summary["label"] = summary["category"].apply(clean_category_name)

    plt.figure(figsize=(9, 6))

    plt.barh(summary["label"], summary["mean_similarity"])

    max_value = summary["mean_similarity"].max()
    label_offset = 0.01
    right_padding = 0.08

    for index, value in enumerate(summary["mean_similarity"]):
        plt.text(
            value + label_offset,
            index,
            f"{value:.3f}",
            va="center",
        )

    plt.xlabel("Mean Cosine Similarity")
    plt.ylabel("Sentence Pair Category")
    plt.title("Mean Cosine Similarity by Category")
    plt.xlim(0, max_value + right_padding)
    plt.tight_layout()

    plt.savefig(CATEGORY_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    

def make_pair_level_scatter_plot() -> None:
    """
    Make a pair-level scatter plot showing each sentence pair's similarity.
    """
    df = pd.read_csv(PAIR_RESULTS_PATH)
    summary = pd.read_csv(CATEGORY_RESULTS_PATH)

    category_order = (
        summary.sort_values("mean_similarity", ascending=True)["category"]
        .tolist()
    )

    category_to_y = {
        category: index
        for index, category in enumerate(category_order)
    }

    df["y_position"] = df["category"].map(category_to_y)
    df["label"] = df["category"].apply(clean_category_name)

    plt.figure(figsize=(9, 6))

    plt.scatter(df["similarity"], df["y_position"])

    plt.yticks(
        ticks=list(category_to_y.values()),
        labels=[clean_category_name(category) for category in category_order],
    )

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Sentence Pair Category")
    plt.title("Pair-Level Cosine Similarities by Category")
    plt.xlim(0, 1.05)
    plt.tight_layout()

    plt.savefig(PAIR_FIGURE_PATH, dpi=200)
    plt.close()


def make_pair_level_scatter_plot_color() -> None:
    """
    Each category is plotted separately so it gets its own color and legend entry.
    """
    df = pd.read_csv(PAIR_RESULTS_PATH)
    summary = pd.read_csv(CATEGORY_RESULTS_PATH)

    category_order = (
        summary.sort_values("mean_similarity", ascending=True)["category"]
        .tolist()
    )

    category_to_y = {
        category: index
        for index, category in enumerate(category_order)
    }

    df["y_position"] = df["category"].map(category_to_y)

    plt.figure(figsize=(10, 6))

    for category in category_order:
        category_df = df[df["category"] == category]

        plt.scatter(
            category_df["similarity"],
            category_df["y_position"],
            label=clean_category_name(category),
        )

    plt.yticks(
        ticks=list(category_to_y.values()),
        labels=[clean_category_name(category) for category in category_order],
    )

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Sentence Pair Category")
    plt.title("Pair-Level Cosine Similarities by Category")
    plt.xlim(0, 1.05)
    plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(COLOR_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close()

def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    make_category_bar_chart()
    make_pair_level_scatter_plot()
    make_pair_level_scatter_plot_color()

    print(f"Saved category bar chart to: {CATEGORY_FIGURE_PATH}")
    print(f"Saved pair-level scatter plot to: {PAIR_FIGURE_PATH}")
    print(f"Saved color-coded scatter plot to: {COLOR_FIGURE_PATH}")


if __name__ == "__main__":
    main()