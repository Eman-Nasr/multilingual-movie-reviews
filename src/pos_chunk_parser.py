from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score


def _load_models() -> Dict[str, "spacy.Language"]:
    print(" Loading spaCy models (en + es)…")
    nlp_en = spacy.load("en_core_web_sm")
    nlp_es = spacy.load("es_core_news_sm")
    return {"en": nlp_en, "es": nlp_es}


def _sample_sentences(df: pd.DataFrame, lang: str, n: int = 200) -> List[str]:
    subset = df[df["lang"] == lang]["text"].dropna().sample(
        n=min(n, df[df["lang"] == lang].shape[0]), random_state=42
    )
    return subset.tolist()


def _pos_baseline(tokens: List[str]) -> List[str]:
    """
    Very naive baseline POS tagger:
    - tag everything as NOUN.
    This gives us something to compare spaCy against.
    """
    return ["NOUN"] * len(tokens)


def _evaluate_pos(nlp, sentences: List[str]) -> Tuple[List[str], List[str]]:
    y_true_all = []
    y_pred_all = []

    for sent in sentences:
        doc = nlp(sent)
        tokens = [t.text for t in doc]
        true_tags = [t.pos_ for t in doc]
        baseline_tags = _pos_baseline(tokens)

        y_true_all.extend(true_tags)
        y_pred_all.extend(baseline_tags)

    return y_true_all, y_pred_all


def _analyse_chunks(nlp, sentences: List[str], lang: str) -> None:
    print(f"\n Example noun chunks for {lang.upper()}:")
    for sent in sentences[:5]:
        doc = nlp(sent)
        print(f"  SENT: {sent}")
        for chunk in doc.noun_chunks:
            print(f"    NP: {chunk.text}")
        print("---")


def run_pos_chunking() -> None:
    print("\n==============================")
    print("STAGE 3 POS TAGGING + PARSING / CHUNKING")
    print("==============================")

    df = pd.read_csv("data/processed/reviews_long.csv")
    nlp_models = _load_models()

    metrics_rows = []

    for lang in ["en", "es"]:
        nlp = nlp_models[lang]
        sents = _sample_sentences(df, lang, n=200)

        print(f"\n Evaluating POS baseline vs spaCy for {lang.upper()} ({len(sents)} samples)…")
        y_true, y_pred = _evaluate_pos(nlp, sents)

        acc = accuracy_score(y_true, y_pred)
        metrics_rows.append({"lang": lang, "metric": "pos_baseline_accuracy", "value": acc})

        print(f"   Baseline accuracy vs spaCy POS (pseudo-gold): {acc:.3f}")

        # Expose variables exactly as mentioned in feedback
        if lang == "en":
            global y_true_en, y_pred_en
            y_true_en, y_pred_en = y_true, y_pred
        else:
            global y_true_es, y_pred_es
            y_true_es, y_pred_es = y_true, y_pred

        # Parsing / chunking analysis
        _analyse_chunks(nlp, sents, lang=lang)

    metrics_df = pd.DataFrame(metrics_rows)
    import os

    os.makedirs("outputs/pos_parsing", exist_ok=True)
    metrics_path = "outputs/pos_parsing/pos_baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n Saved POS baseline metrics → {metrics_path}")
    print(metrics_df)

    print("\n POS + Parsing / Chunking stage complete.")
