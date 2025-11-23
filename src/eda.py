import os
import re
import string
from typing import Tuple

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords if not already
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


def _basic_clean(text: str) -> str:
    """
    Simple normalization used for BOTH languages:
    - lowercasing
    - remove digits
    - remove punctuation
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_stopwords(text: str, lang_code: str) -> str:
    if not isinstance(text, str):
        return ""
    if lang_code == "en":
        lang = "english"
    else:
        lang = "spanish"

    sw = set(stopwords.words(lang))
    tokens = text.split()
    filtered = [t for t in tokens if t not in sw]
    return " ".join(filtered)


def clean_text_columns(df_long: pd.DataFrame) -> pd.DataFrame:
    print(" [EDA] Cleaning texts (basic normalization + stop word removal)…")

    df_long["text_basic_clean"] = df_long["text"].apply(_basic_clean)
    df_long["text_clean_no_stop"] = df_long.apply(
        lambda row: _remove_stopwords(row["text_basic_clean"], row["lang"]),
        axis=1
    )
    return df_long


def run_eda() -> None:
    print("\n==============================")
    print("STAGE  DATA & EDA")
    print("==============================")

    path = "data/processed/reviews_long.csv"
    df = pd.read_csv(path)
    print(f" Loaded long dataset from {path} → {df.shape}")

    # 1) Apply explicit cleaning for BOTH languages (fixes D2 cleaning feedback)
    df = clean_text_columns(df)

    # 2) Sentiment label normalization for later stages
    #    Map ES labels to EN for modelling consistency
    df["sentiment_norm"] = df["sentiment"].replace({
        "positivo": "positive",
        "negativo": "negative"
    })

    # 3) Language-wise stats (fix: include English analysis)
    print("\n Reviews per language:")
    print(df["lang"].value_counts())

    print("\n Wordcount summary by language:")
    print(df.groupby("lang")["wordcount"].describe())

    # Class balance → used later to explain “more positive than negative”
    print("\n  Sentiment distribution (normalized):")
    print(df["sentiment_norm"].value_counts())

    # 4) Plots (both languages)
    os.makedirs("outputs/eda", exist_ok=True)

    # Wordcount boxplot by lang
    plt.figure(figsize=(6, 4))
    df.boxplot(column="wordcount", by="lang")
    plt.title("Wordcount distribution by language")
    plt.suptitle("")
    plt.ylabel("Wordcount")
    plt.savefig("outputs/eda/wordcount_by_lang.png")
    plt.close()

    # Sentiment by lang
    plt.figure(figsize=(6, 4))
    (
        df.groupby(["lang", "sentiment_norm"])["text"]
        .count()
        .unstack(fill_value=0)
        .plot(kind="bar", ax=plt.gca())
    )
    plt.ylabel("# Reviews")
    plt.title("Sentiment distribution by language")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/eda/sentiment_by_lang.png")
    plt.close()

    print(" Saved EDA figures in outputs/eda/")
    df.to_csv("data/processed/reviews_long.csv", index=False)
    print(" EDA finished. Use these outputs + printed stats in the report.")
