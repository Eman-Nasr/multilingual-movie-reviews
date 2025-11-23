"""
Multilingual Movie Reviews – Full NLP Pipeline
Matches course D4 submission requirements.

Stages:
1. Preprocessing
2. EDA
3. N-gram LM
4. Perplexity
5. POS + Chunking
6. NER
7. Sentiment Classification
8. Evaluation Deep-Dive
"""

import ast
from .preprocessing import run_preprocessing
from .eda import run_eda
from .ngram_lm import run_ngram_models
from .perplexity import run_perplexity
from .pos_chunk_parser import run_pos_chunking
from .ner import run_ner
from .sentiment_model import run_sentiment_pipeline
from .evaluation import run_evaluation


def convert_token_strings(df):
    """Convert string tokens → Python lists safely."""
    if isinstance(df["tokens"].iloc[0], str):
        df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def run_full_pipeline() -> None:
    print("\n STARTING FULL MULTILINGUAL PIPELINE\n")

    df_raw, df_long = run_preprocessing()
    df_long = convert_token_strings(df_long)
    print(" Preprocessing completed.\n")

    run_eda()
    print(" EDA completed.\n")

    ngram_results = run_ngram_models(df_long)
    print(" N-gram models completed.\n")

    run_perplexity()
    print(" Perplexity completed.\n")

    df_pos = run_pos_chunking()
    print(" POS / Chunking completed.\n")

    df_ner = run_ner()
    print(" NER completed.\n")

    df_sent = run_sentiment_pipeline()
    print(" Sentiment classification completed.\n")

    run_evaluation()
    print(" Final evaluation completed.\n")

    print(" FULL PIPELINE COMPLETED — Results saved to outputs/")
