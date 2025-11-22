import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def run_sentiment_pipeline() -> None:
    print("\n==============================")
    print("STAGE 4 SENTIMENT ANALYSIS")
    print("==============================")

    df = pd.read_csv("data/processed/reviews_long.csv")

    # Recreate the cleaned column if EDA not called in same run
    if "text_clean_no_stop" not in df.columns:
        from .eda import clean_text_columns  # local import to avoid circular
        df = clean_text_columns(df)

    # Normalize sentiment labels
    df["sentiment_norm"] = df["sentiment"].replace(
        {"positivo": "positive", "negativo": "negative"}
    )

    # Class balance
    print("\n Sentiment distribution (normalized):")
    print(df["sentiment_norm"].value_counts())
    print(" In the report you will explain how this imbalance may bias the classifier.")

    X = df["text_clean_no_stop"]
    y = df["sentiment_norm"]
    langs = df["lang"]

    X_train, X_test, y_train, y_test, lang_train, lang_test = train_test_split(
        X, y, langs, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Overall metrics
    print("\n Overall sentiment classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Per-language metrics
    for lang_code in ["en", "es"]:
        mask = lang_test == lang_code
        if mask.sum() == 0:
            continue
        print(f"\n Sentiment report for {lang_code.upper()}:")
        print(classification_report(y_test[mask], y_pred[mask], digits=3))

    os.makedirs("outputs/sentiment", exist_ok=True)
    # Save predictions for error analysis
    results = pd.DataFrame(
        {
            "text": X_test,
            "lang": lang_test,
            "true": y_test,
            "pred": y_pred,
        }
    )
    results_path = "outputs/sentiment/sentiment_predictions.csv"
    results.to_csv(results_path, index=False)

    print(f"\n Saved sentiment predictions → {results_path}")
    print(" Sentiment stage complete.")
