# ============================================================
# SMART FAKE NEWS DETECTION SYSTEM
# Model Training and Evaluation
# ============================================================

import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATA_PATH = "data/WELFake_Dataset.csv"
MODEL_DIR = "models"
SCREENSHOT_DIR = "screenshots"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_news_model.joblib"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

print("\n" + "=" * 65)
print("       SMART FAKE NEWS DETECTION SYSTEM")
print("              MODEL TRAINING")
print("=" * 65)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] Loading WELFake dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place WELFake_Dataset.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

required_columns = ["title", "text", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n[2] Cleaning dataset...")

df = df[required_columns].copy()

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df.dropna(subset=["label"], inplace=True)

df["label"] = df["label"].astype(int)

# Remove invalid labels
df = df[df["label"].isin([0, 1])]

# Remove rows with no useful text
df["combined_text"] = (
    df["title"].astype(str).str.strip()
    + " "
    + df["text"].astype(str).str.strip()
)

df["combined_text"] = (
    df["combined_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df[df["combined_text"] != ""]

# Remove duplicate records
df.drop_duplicates(
    subset=["combined_text"],
    inplace=True
)

df.reset_index(drop=True, inplace=True)

print("Cleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[3] Performing exploratory data analysis...")

label_counts = df["label"].value_counts().sort_index()

print("\nLabel Distribution:")
print("Fake News (0):", label_counts.get(0, 0))
print("Real News (1):", label_counts.get(1, 0))

plt.figure(figsize=(8, 5))

plt.bar(
    ["Fake News", "Real News"],
    [
        label_counts.get(0, 0),
        label_counts.get(1, 0)
    ]
)

plt.title("Distribution of Fake and Real News")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "label_distribution.png"
    ),
    dpi=200
)

plt.close()

print("Label distribution chart saved.")


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

print("\n[4] Preparing training and testing data...")

X = df["combined_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\n[5] Initializing machine learning models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB()
}

results = []
trained_pipelines = {}


# ============================================================
# 7. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n[6] Applying TF-IDF and training models...")

for model_name, classifier in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 1),
                min_df=3,
                stop_words="english",
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_pipelines[model_name] = pipeline

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            SCREENSHOT_DIR,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=200
    )

    plt.close()

    print("Confusion matrix saved.")


# ============================================================
# 8. MODEL PERFORMANCE COMPARISON
# ============================================================

print("\n[7] Comparing model performance...")

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison CSV saved.")


# ============================================================
# 9. GENERATE MODEL COMPARISON BAR CHART
# ============================================================

plt.figure(figsize=(10, 6))

plot_df = results_df.set_index("Model")

plot_df.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.75
)

plt.title(
    "Machine Learning Model Performance Comparison",
    fontsize=14
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")

plt.ylim(0, 1.05)

plt.xticks(rotation=0)

plt.legend(
    title="Evaluation Metrics"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "model_comparison_bar_chart.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Model comparison bar chart saved.")


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

print("\n[8] Selecting the best model...")

best_index = results_df["F1 Score"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_pipeline = trained_pipelines[best_model_name]

print("Best Model:", best_model_name)

print(
    "Best F1 Score:",
    round(
        results_df.loc[best_index, "F1 Score"] * 100,
        2
    ),
    "%"
)


# ============================================================
# 11. SAVE COMPLETE MODEL PIPELINE
# ============================================================

print("\n[9] Saving trained model...")

joblib.dump(
    best_pipeline,
    MODEL_PATH
)

print("Model successfully saved at:", MODEL_PATH)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n[10] Classification Report")

best_predictions = best_pipeline.predict(X_test)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("             MODEL TRAINING COMPLETED")
print("=" * 65)

print("Dataset:", DATA_PATH)
print("Total cleaned records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Best model:", best_model_name)
print("Saved model:", MODEL_PATH)

print("\nGenerated Files:")

print("- models/fake_news_model.joblib")
print("- model_comparison.csv")
print("- screenshots/label_distribution.png")
print("- screenshots/logistic_regression_confusion_matrix.png")
print("- screenshots/multinomial_naive_bayes_confusion_matrix.png")
print("- screenshots/model_comparison_bar_chart.png")

print("\nSMART FAKE NEWS DETECTION SYSTEM READY!")
print("=" * 65)