"""
eda.py
Exploratory Data Analysis (EDA) of the Titanic passenger dataset.

Dataset: Titanic passengers (891 rows) from the public "seaborn-data" repository
https://github.com/mwaskom/seaborn-data  (a classic teaching dataset)

What this script does:
  1. Loads the data with pandas (local copy first, download if missing)
  2. Checks shape, data types, missing values and duplicates
  3. Cleans the data (and explains each decision)
  4. Calculates descriptive statistics and group comparisons
  5. Studies distributions, relationships, correlations and outliers
  6. Saves 8 charts, a cleaned CSV and two text reports

Run with:  python eda.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")                      # save charts to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------- Settings -------------------------------
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
RAW_PATH = os.path.join("data", "raw", "titanic.csv")
CLEAN_PATH = os.path.join("data", "processed", "titanic_clean.csv")
CHART_DIR = "charts"
REPORT_DIR = "reports"

REPORT_LINES = []                          # everything we print is also saved here


def say(text=""):
    """Print a line and also remember it for the saved report."""
    print(text)
    REPORT_LINES.append(str(text))


# ------------------------------ 1. Load data ------------------------------
def load_data():
    """Load the dataset from data/raw, or download it once and save a copy."""
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"Loading local file: {RAW_PATH}")
        return pd.read_csv(RAW_PATH)
    print(f"Local file not found. Downloading from {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as error:             # no internet, blocked URL, etc.
        print(f"Download failed: {error}")
        print(f"Open the URL above in your browser, save the file as {RAW_PATH}, then run again.")
        sys.exit(1)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved a copy to {RAW_PATH}")
    return df


# ---------------------------- 2. Inspect the data ----------------------------
def inspect_data(df):
    """Dimensions, data types, statistics, missing values and duplicates."""
    say("=" * 70)
    say("1. DATASET STRUCTURE")
    say("=" * 70)
    say(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    say("\nColumn names and data types:")
    say(df.dtypes.to_string())
    say("\nFirst 5 rows:")
    say(df.head().to_string())

    say("\n" + "=" * 70)
    say("2. DESCRIPTIVE STATISTICS (numeric columns)")
    say("=" * 70)
    say(df.describe().round(2).to_string())

    say("\nDescriptive statistics (text/category columns):")
    say(df.describe(include=["object", "string", "bool"]).to_string())

    say("\n" + "=" * 70)
    say("3. DATA QUALITY CHECKS")
    say("=" * 70)
    missing = df.isnull().sum()
    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / len(df) * 100).round(1),
    })
    missing_table = missing_table[missing_table["missing_count"] > 0]
    say("Missing values (only columns that have some):")
    say(missing_table.to_string())

    duplicate_rows = int(df.duplicated().sum())
    say(f"\nExact duplicate rows: {duplicate_rows}")
    say("Decision: duplicates are NOT removed. This dataset has no name or ticket")
    say("column, so two different passengers can look identical (same class, sex,")
    say("age, fare, port). We cannot prove these rows are copies, so we keep them.")

    # Check whether some columns just repeat the information of other columns
    alive_matches = bool((df["alive"].eq("yes") == df["survived"].eq(1)).all())
    class_matches = bool((df["class"].map({"First": 1, "Second": 2, "Third": 3}) == df["pclass"]).all())
    say(f"\n'alive' repeats 'survived': {alive_matches}")
    say(f"'class' repeats 'pclass':  {class_matches}")
    say("These columns are kept, but we only use the numeric versions in the analysis.")
    return missing_table, duplicate_rows


# ------------------------------- 3. Clean data -------------------------------
def clean_data(df):
    """Handle missing values and create helper columns. Every step is explained."""
    say("\n" + "=" * 70)
    say("4. DATA CLEANING")
    say("=" * 70)
    clean = df.copy()

    # Age: 'missing' is itself information, so we keep the original column and
    # add a flag plus a separate filled column.
    clean["age_missing"] = clean["age"].isna()
    typical_age = clean.groupby(["pclass", "sex"])["age"].transform("median")
    clean["age_filled"] = clean["age"].fillna(typical_age)
    say(f"age: {int(clean['age_missing'].sum())} missing values. Original 'age' is kept as it is.")
    say("     New column 'age_filled' uses the median age of the same class and sex.")
    say("     'age_missing' marks which rows were filled. Age charts use real ages only.")

    # Deck: most values are missing, so filling with a guess would be wrong.
    deck_missing = int(clean["deck"].isna().sum())
    clean["deck"] = clean["deck"].fillna("Unknown")
    say(f"deck: {deck_missing} missing values (too many to estimate). Labelled 'Unknown'.")

    # Embarked: only a couple of rows are missing, so the most common value is used.
    port_mode = clean["embarked"].mode()[0]
    town_mode = clean["embark_town"].mode()[0]
    port_missing = int(clean["embarked"].isna().sum())
    clean["embarked"] = clean["embarked"].fillna(port_mode)
    clean["embark_town"] = clean["embark_town"].fillna(town_mode)
    say(f"embarked / embark_town: {port_missing} missing values filled with the most common value ({port_mode}).")

    # New helper columns
    clean["family_size"] = clean["sibsp"] + clean["parch"] + 1
    age_bins = [0, 12, 18, 35, 60, 120]
    age_labels = ["Child (0-12)", "Teen (13-18)", "Young adult (19-35)",
                  "Adult (36-60)", "Senior (61+)"]
    age_group = pd.cut(clean["age"], bins=age_bins, labels=age_labels, include_lowest=True)
    clean["age_group"] = age_group.astype("object").fillna("Unknown")
    clean["survival_label"] = clean["survived"].map({1: "Survived", 0: "Did not survive"})
    say("New columns created: family_size, age_group, survival_label.")

    # Outliers: flagged using the IQR rule, NOT removed
    q1, q3 = clean["fare"].quantile([0.25, 0.75])
    upper_limit = q3 + 1.5 * (q3 - q1)
    clean["fare_outlier"] = clean["fare"] > upper_limit
    zero_fares = int((clean["fare"] == 0).sum())
    say(f"\nFare outliers (IQR rule, fare above {upper_limit:.2f}): {int(clean['fare_outlier'].sum())} rows.")
    outlier_by_class = clean[clean["fare_outlier"]]["pclass"].value_counts().sort_index()
    say("Fare outliers by passenger class: " + ", ".join(
        f"class {cls} = {count}" for cls, count in outlier_by_class.items()))
    say("Decision: outliers are NOT removed. They are plausible high ticket prices and")
    say("removing them would hide a real part of the story.")
    say(f"Passengers with a fare of 0: {zero_fares}. They are kept, but the reason is not")
    say("recorded in the data, so treat those values with care.")

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)
    say(f"\nCleaned dataset saved to {CLEAN_PATH}  ({clean.shape[0]} rows, {clean.shape[1]} columns)")
    say(f"Missing values left in cleaned data (excluding the original 'age'): "
        f"{int(clean.drop(columns=['age']).isnull().sum().sum())}")
    return clean


# ------------------------------ 4. Analysis ------------------------------
def survival_table(df, column):
    """Passengers, survivors and survival rate (%) for each group in a column."""
    table = df.groupby(column)["survived"].agg(
        passengers="count", survivors="sum", survival_rate="mean")
    table["survival_rate"] = (table["survival_rate"] * 100).round(1)
    return table


def analyse(clean):
    """Group comparisons, distributions and correlations. Returns key numbers."""
    say("\n" + "=" * 70)
    say("5. ANALYSIS")
    say("=" * 70)
    results = {}

    results["overall"] = clean["survived"].mean() * 100
    say(f"Overall survival rate: {results['overall']:.1f}% "
        f"({int(clean['survived'].sum())} of {len(clean)} passengers)")

    for column in ["sex", "pclass", "embarked", "age_group", "alone"]:
        table = survival_table(clean, column)
        results[column] = table
        say(f"\nSurvival by {column}:")
        say(table.to_string())

    results["sex_class"] = clean.pivot_table(
        index="pclass", columns="sex", values="survived", aggfunc="mean") * 100
    say("\nSurvival rate (%) by class and sex:")
    say(results["sex_class"].round(1).to_string())

    results["family"] = survival_table(clean, "family_size")
    say("\nSurvival by family size (small groups are unreliable):")
    say(results["family"].to_string())

    say("\nFare by class (median and mean):")
    results["fare_class"] = clean.groupby("pclass")["fare"].agg(["count", "median", "mean", "max"]).round(2)
    say(results["fare_class"].to_string())

    say("\nAge statistics (real ages only):")
    results["age_by_survival"] = clean.groupby("survival_label")["age"].agg(["count", "mean", "median"]).round(1)
    say(results["age_by_survival"].to_string())

    numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]
    results["corr"] = clean[numeric_columns].corr().round(2)
    say("\nCorrelation matrix (Pearson):")
    say(results["corr"].to_string())
    return results


# ------------------------------ 5. Charts ------------------------------
def save_chart(name):
    """Tidy up and save the current chart, then close it."""
    plt.tight_layout()
    path = os.path.join(CHART_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()


def make_charts(raw, clean, results, missing_table):
    """Create and save 8 charts."""
    os.makedirs(CHART_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    colours = {"Survived": "seagreen", "Did not survive": "indianred"}

    # 1. Missing values (raw data)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=missing_table["missing_percent"], y=missing_table.index,
                     orient="h", color="steelblue")
    ax.set_title("Missing Values by Column (raw data)")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    save_chart("01_missing_values.png")

    # 2. Survival counts by sex
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=clean, x="sex", hue="survival_label", palette=colours)
    ax.set_title("Survival Counts by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Number of passengers")
    ax.legend(title="Outcome")
    for container in ax.containers:
        ax.bar_label(container)
    save_chart("02_survival_by_sex.png")

    # 3. Survival rate by class and sex
    ax = results["sex_class"].plot(kind="bar", figsize=(8, 5), color=["salmon", "steelblue"])
    ax.set_title("Survival Rate by Passenger Class and Sex")
    ax.set_xlabel("Passenger class (1 = first, 3 = third)")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Sex")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%")
    save_chart("03_survival_rate_class_sex.png")

    # 4. Age distribution by outcome (real ages only)
    plt.figure(figsize=(8, 5))
    known_age = clean[clean["age"].notna()]
    ax = sns.histplot(data=known_age, x="age", hue="survival_label", bins=30,
                      multiple="stack", palette=colours)
    ax.set_title("Age Distribution by Survival Outcome (passengers with known age)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of passengers")
    save_chart("04_age_distribution.png")

    # 5. Fare by class (log scale because fares are very skewed)
    plt.figure(figsize=(8, 5))
    plot_data = clean.assign(log_fare=np.log1p(clean["fare"]))
    ax = sns.boxplot(data=plot_data, x="pclass", y="log_fare", color="lightsteelblue")
    ax.set_title("Ticket Fare by Passenger Class (log scale, dots = outliers)")
    ax.set_xlabel("Passenger class (1 = first, 3 = third)")
    ax.set_ylabel("log(1 + fare)")
    save_chart("05_fare_by_class_boxplot.png")

    # 6. Correlation heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(results["corr"], annot=True, fmt=".2f", cmap="coolwarm",
                     vmin=-1, vmax=1, square=True)
    ax.set_title("Correlation Between Numeric Variables")
    save_chart("06_correlation_heatmap.png")

    # 7. Survival rate by age group
    order = ["Child (0-12)", "Teen (13-18)", "Young adult (19-35)",
             "Adult (36-60)", "Senior (61+)", "Unknown"]
    age_table = results["age_group"].reindex(order).dropna()
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=age_table.index, y=age_table["survival_rate"], color="mediumpurple")
    ax.set_title("Survival Rate by Age Group (Unknown = age not recorded)")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    labels = [f"{rate:.0f}%\n(n={int(n)})" for rate, n in
              zip(age_table["survival_rate"], age_table["passengers"])]
    ax.bar_label(ax.containers[0], labels=labels)
    save_chart("07_survival_by_age_group.png")

    # 8. Survival rate by family size
    family = results["family"]
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=family.index.astype(str), y=family["survival_rate"], color="darkorange")
    ax.set_title("Survival Rate by Family Size (n = passengers in group)")
    ax.set_xlabel("Family size (including the passenger)")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 110)
    labels = [f"{rate:.0f}%\nn={int(n)}" for rate, n in
              zip(family["survival_rate"], family["passengers"])]
    ax.bar_label(ax.containers[0], labels=labels, fontsize=8)
    save_chart("08_survival_by_family_size.png")

    say(f"\nCharts saved in '{CHART_DIR}': {sorted(os.listdir(CHART_DIR))}")


# ------------------------------ 6. Findings ------------------------------
def write_findings(raw, clean, results, missing_table, duplicate_rows):
    """Write findings that are calculated from the data (nothing is typed by hand)."""
    sex = results["sex"]["survival_rate"]
    pclass = results["pclass"]["survival_rate"]
    age_known = results["age_group"].drop(index="Unknown", errors="ignore")
    family = results["family"]
    corr_with_survival = results["corr"]["survived"].drop("survived")
    strongest = corr_with_survival.abs().idxmax()
    sex_class = results["sex_class"]
    reliable_family = family[family["passengers"] >= 30]

    lines = [
        "# EDA Findings (generated by eda.py from the actual data)",
        "",
        "## Research questions and answers",
        "",
        f"**Q1. How complete is the data?** The dataset has {len(clean)} rows and "
        f"{clean.shape[1]} columns after cleaning (the raw file has {raw.shape[1]} columns). "
        f"Columns with missing values: "
        + ", ".join(f"{name} ({int(row['missing_count'])}, {row['missing_percent']}%)"
                    for name, row in missing_table.iterrows()) + ". "
        f"There are {duplicate_rows} exact duplicate rows, which were kept because "
        "the data has no ID column to prove they are copies.",
        "",
        f"**Q2. What was the overall survival rate?** {results['overall']:.1f}%.",
        "",
        f"**Q3. Did survival differ by sex?** Survival rate by sex: "
        + ", ".join(f"{name} {rate:.1f}%" for name, rate in sex.items()) + ".",
        "",
        f"**Q4. Did survival differ by passenger class?** "
        + ", ".join(f"class {name}: {rate:.1f}%" for name, rate in pclass.items()) + ". "
        "Class and sex together (survival rate %): "
        + "; ".join(f"class {idx}: " + ", ".join(f"{col} {val:.0f}%" for col, val in row.items())
                    for idx, row in sex_class.iterrows()) + ".",
        "",
        f"**Q5. Did age matter?** Among passengers with a known age, the highest survival rate was "
        f"{age_known['survival_rate'].idxmax()} ({age_known['survival_rate'].max():.1f}%, "
        f"n={int(age_known.loc[age_known['survival_rate'].idxmax(), 'passengers'])}) and the lowest was "
        f"{age_known['survival_rate'].idxmin()} ({age_known['survival_rate'].min():.1f}%, "
        f"n={int(age_known.loc[age_known['survival_rate'].idxmin(), 'passengers'])}). "
        f"Passengers whose age is missing survived at {results['age_group'].loc['Unknown', 'survival_rate']:.1f}%.",
        "",
        f"**Q6. Did family size matter?** Looking only at family sizes with at least 30 passengers, "
        f"the highest survival rate was for family size {reliable_family['survival_rate'].idxmax()} "
        f"({reliable_family['survival_rate'].max():.1f}%) and the lowest for family size "
        f"{reliable_family['survival_rate'].idxmin()} ({reliable_family['survival_rate'].min():.1f}%). "
        "Larger family sizes have very few passengers, so their rates are unreliable.",
        "",
        f"**Q7. What do fares look like?** Median fare by class: "
        + ", ".join(f"class {idx} = {row['median']:.2f}" for idx, row in results["fare_class"].iterrows())
        + f". {int(clean['fare_outlier'].sum())} passengers have unusually high fares (IQR rule), "
        f"and {int((clean['fare'] == 0).sum())} passengers have a fare of 0.",
        "",
        f"**Q8. Which numeric variable has the strongest linear link with survival?** "
        f"{strongest} (correlation {corr_with_survival[strongest]:.2f}).",
        "",
        "## Limitations",
        "- The Titanic dataset is a well-known teaching dataset (a demonstration dataset). "
        "It is real historical data but it is not a model of any current business problem.",
        "- Age is missing for a large part of the passengers, so age results are based on a subset.",
        "- Correlation and group differences show association, not cause and effect.",
        "- The data comes from the seaborn-data copy of the dataset and has no passenger names or "
        "ticket numbers, so duplicates cannot be verified.",
        "- Small groups (for example large families) give unstable percentages.",
        "",
    ]
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "findings.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    with open(os.path.join(REPORT_DIR, "eda_report.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(REPORT_LINES))
    print("\nFindings saved to reports/findings.md and full output saved to reports/eda_report.txt")
    print("\n".join(lines))


# ------------------------------ Main program ------------------------------
def main():
    raw = load_data()
    missing_table, duplicate_rows = inspect_data(raw)
    clean = clean_data(raw)
    results = analyse(clean)
    make_charts(raw, clean, results, missing_table)
    write_findings(raw, clean, results, missing_table, duplicate_rows)


if __name__ == "__main__":
    main()
