import os
import pickle
import pandas as pd
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab")


# ---------------------------------------------------------
# Utility: clean + tokenize
# ---------------------------------------------------------
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens


# ---------------------------------------------------------
# TRAIN ALL N-GRAM MODELS (1, 2, 3) FOR EN & ES
# ---------------------------------------------------------
def run_ngram_models(df, output_dir="outputs/ngram"):
    os.makedirs(output_dir, exist_ok=True)

    models = {}
    languages = ["en", "es"]

    for lang in languages:
        print(f"\n TRAINING {lang.upper()} MODELS...")

        lang_reviews = df[df["lang"] == lang]["text"].dropna().tolist()
        tokenized = [clean_and_tokenize(text) for text in lang_reviews]

        for n in [1, 2, 3]:
            print(f"   → Training {n}-gram for {lang} ...")

            train_data, vocab = padded_everygram_pipeline(n, tokenized)
            model = MLE(n)
            model.fit(train_data, vocab)

            save_path = os.path.join(output_dir, f"{lang}_{n}gram.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

            models[(lang, n)] = model
            print(f"      Saved → {save_path}")

    print("\n ALL N-GRAM MODELS TRAINED AND SAVED.")
    return models


def run_ngram_training(input_csv="data/processed/reviews_long.csv",
                       output_dir="outputs/ngram"):

    print("\n=== N-GRAM TRAINING STAGE ===")

    df = pd.read_csv(input_csv)
    print(f"Loaded dataset: {df.shape[0]} rows")

    run_ngram_models(df, output_dir=output_dir)

    print("=== N-GRAM STAGE COMPLETE ===\n")


if __name__ == "__main__":
    run_ngram_training()
