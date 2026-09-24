"""Titanic profiling, cleaning and exploratory analysis.

The raw Titanic dataset is loaded from seaborn exactly once in this module,
then immediately saved to titanic.csv. All later work uses that DataFrame.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

RAW_FALLBACK = HERE / "titanic.csv"
CLEANED = HERE / "cleaned_titanic.csv"
REPORT = HERE / "EDA_REPORT.md"


def iqr_outliers(series: pd.Series) -> int:
    """Return the number of IQR-based outliers in a numeric series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(((series < lower) | (series > upper)).sum())


def add_plot_interpretations(
    report: list[str],
    name: str,
    text: str,
) -> None:
    """Add a chart interpretation to the report."""
    report.append(f"### {name}\n{text}\n")


def main() -> None:
    # Load Titanic exactly once and save the raw dataset.
    df = sns.load_dataset("titanic")
    df.to_csv(RAW_FALLBACK, index=False)

    report: list[str] = [
        "# Titanic EDA Report",
        "",
        "## Initial profile",
        "",
        f"Shape: `{df.shape}`",
        "",
    ]

    # Required df.info().
    report.append("### df.info()")
    info_buf = StringIO()
    df.info(buf=info_buf)
    report.append("```text\n" + info_buf.getvalue() + "\n```")

    # Required df.describe().
    report.append(
        "### df.describe()\n```text\n"
        + df.describe().to_string()
        + "\n```"
    )

    # Missing-value percentages for all affected columns.
    missing = (
        (df.isna().mean() * 100)
        .loc[lambda s: s > 0]
        .sort_values(ascending=False)
    )

    report.append("## Missing-value percentages\n")
    report.append(
        missing.to_frame("missing_percent").to_markdown(
            floatfmt=".2f"
        )
    )

    # ---------------------------------------------------------
    # Cleaning based on the required missing-value thresholds.
    # <5%   -> drop affected rows
    # 5-30% -> impute
    # >30%  -> drop column
    # ---------------------------------------------------------
    cleaned = df.copy()
    decisions = []

    for col, pct in missing.items():

        if pct < 5:
            cleaned = cleaned.dropna(subset=[col])

            decisions.append(
                (
                    col,
                    pct,
                    "drop affected rows (<5%)",
                )
            )

        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].fillna(
                    cleaned[col].median()
                )

                decisions.append(
                    (
                        col,
                        pct,
                        "median imputation (5-30%)",
                    )
                )

            else:
                cleaned[col] = cleaned[col].fillna(
                    cleaned[col].mode().iloc[0]
                )

                decisions.append(
                    (
                        col,
                        pct,
                        "mode imputation (5-30%)",
                    )
                )

        else:
            cleaned = cleaned.drop(columns=[col])

            decisions.append(
                (
                    col,
                    pct,
                    "drop column (>30%; imputation unreliable)",
                )
            )

    report.append("## Cleaning decisions\n")
    report.append(
        pd.DataFrame(
            decisions,
            columns=[
                "column",
                "missing_percent",
                "strategy",
            ],
        ).to_markdown(
            index=False,
            floatfmt=".2f",
        )
    )

    report.append(f"\nCleaned shape: `{cleaned.shape}`\n")

    # Save the cleaned dataset.
    cleaned.to_csv(CLEANED, index=False)

    # ---------------------------------------------------------
    # Univariate analysis.
    # ---------------------------------------------------------
    for col in ["age", "fare"]:

        # Histogram.
        plt.figure(figsize=(7, 4))
        sns.histplot(cleaned[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(
            PLOTS / f"hist_{col}.png",
            dpi=150,
        )
        plt.close()

        # Boxplot.
        plt.figure(figsize=(7, 3))
        sns.boxplot(x=cleaned[col])
        plt.title(f"Box plot of {col}")
        plt.tight_layout()
        plt.savefig(
            PLOTS / f"box_{col}.png",
            dpi=150,
        )
        plt.close()

    # Fare statistics.
    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()
    fare_mode = cleaned["fare"].mode().iloc[0]

    # Interpret skewness from mean/median/mode ordering.
    if fare_mean > fare_median > fare_mode:
        skew_text = "right-skewed"

    elif fare_mean < fare_median < fare_mode:
        skew_text = "left-skewed"

    else:
        skew_text = (
            "not strictly classified by the "
            "mean/median/mode ordering alone"
        )

    report += [
        "## Univariate analysis",
        "",
        f"- Age IQR outliers: **{iqr_outliers(cleaned['age'])}**",
        f"- Fare IQR outliers: **{iqr_outliers(cleaned['fare'])}**",
        f"- Fare mean: **{fare_mean:.4f}**",
        f"- Fare median: **{fare_median:.4f}**",
        f"- Fare mode: **{fare_mode:.4f}**",
        (
            f"- Fare distribution: **{skew_text}** because "
            f"mean/median/mode are ordered "
            f"{fare_mean:.2f} / "
            f"{fare_median:.2f} / "
            f"{fare_mode:.2f}."
        ),
        "",
    ]

    # ---------------------------------------------------------
    # Survival-rate breakdowns using explicit boolean masks.
    # ---------------------------------------------------------
    female_mask = cleaned["sex"] == "female"
    male_mask = cleaned["sex"] == "male"

    first_class_mask = cleaned["pclass"] == 1
    second_class_mask = cleaned["pclass"] == 2
    third_class_mask = cleaned["pclass"] == 3

    female_rate = cleaned.loc[
        female_mask,
        "survived",
    ].mean()

    male_rate = cleaned.loc[
        male_mask,
        "survived",
    ].mean()

    pclass_rates = (
        cleaned.groupby(
            "pclass",
            observed=True,
        )["survived"]
        .mean()
    )

    # Explicit & boolean masking for sex + passenger class.
    sex_pclass_rates = pd.DataFrame(
        {
            "female": [
                cleaned.loc[
                    female_mask & first_class_mask,
                    "survived",
                ].mean(),
                cleaned.loc[
                    female_mask & second_class_mask,
                    "survived",
                ].mean(),
                cleaned.loc[
                    female_mask & third_class_mask,
                    "survived",
                ].mean(),
            ],
            "male": [
                cleaned.loc[
                    male_mask & first_class_mask,
                    "survived",
                ].mean(),
                cleaned.loc[
                    male_mask & second_class_mask,
                    "survived",
                ].mean(),
                cleaned.loc[
                    male_mask & third_class_mask,
                    "survived",
                ].mean(),
            ],
        },
        index=[1, 2, 3],
    )

    report.append("## Survival-rate breakdowns\n")

    report.append(
        f"Sex - female: **{female_rate:.3f}**, "
        f"male: **{male_rate:.3f}**."
    )

    report.append(
        "\nBy pclass:\n\n"
        + pclass_rates.to_frame(
            "survival_rate"
        ).to_markdown(
            floatfmt=".3f"
        )
    )

    report.append(
        "\nBy sex and pclass:\n\n"
        + sex_pclass_rates.to_markdown(
            floatfmt=".3f"
        )
    )

    # ---------------------------------------------------------
    # Exact six-column correlation matrix.
    # ---------------------------------------------------------
    corr_cols = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    corr = cleaned[corr_cols].corr()

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
    )

    plt.title("Titanic correlation matrix")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "correlation_heatmap.png",
        dpi=150,
    )

    plt.close()

    # Find top two strongest absolute off-diagonal correlations.
    pairs = []

    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            value = corr.loc[a, b]

            pairs.append(
                (
                    a,
                    b,
                    value,
                    abs(value),
                )
            )

    top2 = sorted(
        pairs,
        key=lambda x: x[3],
        reverse=True,
    )[:2]

    report.append(
        "\n### Two strongest absolute off-diagonal correlations\n"
    )

    for a, b, value, _ in top2:
        report.append(
            f"- `{a}` vs `{b}`: "
            f"correlation = **{value:.4f}**."
        )

    report.append(
        "\nThe correlation matrix contains exactly the six "
        "required columns; `adult_male` and `alone` are excluded.\n"
    )

    # ---------------------------------------------------------
    # Multivariate charts.
    # ---------------------------------------------------------

    # Chart 1 - Survival by sex.
    survival_by_sex = (
        cleaned.groupby(
            "sex",
            observed=True,
        )["survived"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))

    sns.barplot(
        data=survival_by_sex,
        x="sex",
        y="survived",
    )

    plt.ylabel("Survival rate")
    plt.title("Survival rate by sex")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "survival_by_sex.png",
        dpi=150,
    )

    plt.close()

    add_plot_interpretations(
        report,
        "Chart 1 - Survival by sex",
        (
            "The survival rates differ substantially between "
            "the two sex categories. This indicates sex is "
            "strongly associated with the observed survival "
            "outcome in this dataset, although the chart is "
            "descriptive rather than causal."
        ),
    )

    # Chart 2 - Survival by class and sex.
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=cleaned,
        x="pclass",
        y="survived",
        hue="sex",
    )

    plt.ylabel("Survival rate")
    plt.title("Survival by passenger class and sex")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "survival_by_pclass_sex.png",
        dpi=150,
    )

    plt.close()

    add_plot_interpretations(
        report,
        "Chart 2 - Survival by class and sex",
        (
            "Survival varies across passenger classes within "
            "the sex groups. The combined view shows why "
            "looking at only one grouping can hide important "
            "differences between subgroups."
        ),
    )

    # Chart 3 - Age and survival.
    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=cleaned,
        x="survived",
        y="age",
    )

    plt.title("Age distribution by survival")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "age_by_survival.png",
        dpi=150,
    )

    plt.close()

    add_plot_interpretations(
        report,
        "Chart 3 - Age and survival",
        (
            "The age distributions overlap considerably, but "
            "their medians and spreads can still differ. This "
            "suggests age may contribute information while "
            "not being sufficient by itself to separate "
            "survivors from non-survivors."
        ),
    )

    # Chart 4 - Fare and survival.
    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=cleaned,
        x="survived",
        y="fare",
    )

    plt.ylim(
        0,
        cleaned["fare"].quantile(0.99),
    )

    plt.title("Fare distribution by survival")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "fare_by_survival.png",
        dpi=150,
    )

    plt.close()

    add_plot_interpretations(
        report,
        "Chart 4 - Fare and survival",
        (
            "Fare distributions differ between the observed "
            "survival groups, with the non-survivor group and "
            "survivor group showing different central "
            "tendencies. Because fare is related to passenger "
            "class, this relationship should be interpreted "
            "together with pclass rather than in isolation."
        ),
    )

    # Chart 5 - Age, fare and survival.
    plt.figure(figsize=(7, 4))

    sns.scatterplot(
        data=cleaned,
        x="age",
        y="fare",
        hue="survived",
        alpha=0.65,
    )

    plt.title("Age, fare and survival")
    plt.tight_layout()

    plt.savefig(
        PLOTS / "age_fare_survival.png",
        dpi=150,
    )

    plt.close()

    add_plot_interpretations(
        report,
        "Chart 5 - Age, fare and survival",
        (
            "The scatter plot combines two numeric variables "
            "with the survival label. It illustrates that the "
            "groups overlap, reinforcing the need for a "
            "multivariate model rather than a single-threshold "
            "rule."
        ),
    )

    # ---------------------------------------------------------
    # EDA-only standardization sanity check.
    # ---------------------------------------------------------
    for col in ["age", "fare"]:

        z = (
            cleaned[col] - cleaned[col].mean()
        ) / cleaned[col].std()

        report.append(
            f"\n## Standardization sanity check - {col}\n"
        )

        report.append(
            f"Before: mean={cleaned[col].mean():.6f}, "
            f"std={cleaned[col].std():.6f}. "
            f"After: mean={z.mean():.6f}, "
            f"std={z.std():.6f}."
        )

    report.append(
        "\nThis standardization is EDA-only and is not passed "
        "into the modeling pipeline. The modeling pipeline "
        "performs its own training-only scaling.\n"
    )

    # Save the final report.
    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Wrote {RAW_FALLBACK}")
    print(f"Wrote {CLEANED}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
