import os
import pandas as pd


def load_raw_dataset(path: str = "data/multilingual_reviews_cleaned.csv") -> pd.DataFrame:
    print(f" [PREPROCESS] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def reshape_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build long-format dataset with both English and Spanish.
    """

    en_df = pd.DataFrame({
        "text": df["review_en_clean"],
        "tokens": df["review_en_tokens"],
        "wordcount": df["review_en_wordcount"],
        "sentiment": df["sentiment"],   # 'positive'/'negative'
        "lang": "en"
    })

    es_df = pd.DataFrame({
        "text": df["review_es_clean"],
        "tokens": df["review_es_tokens"],
        "wordcount": df["review_es_wordcount"],
        "sentiment": df["sentimiento"],  # 'positivo'/'negativo'
        "lang": "es"
    })

    long_df = pd.concat([en_df, es_df], ignore_index=True)
    print(f"   Long-format: {long_df.shape[0]} rows × {long_df.shape[1]} columns")

    return long_df


def save_processed_data(df_raw: pd.DataFrame, df_long: pd.DataFrame) -> None:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    raw_path = "data/raw/multilingual_reviews_cleaned.csv"
    long_path = "data/processed/reviews_long.csv"

    df_raw.to_csv(raw_path, index=False)
    df_long.to_csv(long_path, index=False)

    print(f"   Saved RAW dataset    → {raw_path}")
    print(f"   Saved LONG dataset   → {long_path}")


def run_preprocessing() -> None:
    df_raw = load_raw_dataset()
    df_long = reshape_to_long_format(df_raw)
    save_processed_data(df_raw, df_long)
    
    return df_raw, df_long
