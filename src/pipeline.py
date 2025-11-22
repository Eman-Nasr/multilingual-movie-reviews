"""
Orchestrates the full NLP pipeline for D4.

Stages included (matching the course brief):
1. Preprocessing (load, clean, tokenize, format long/wide)
2. EDA (exploratory statistics, language comparison)
3. N-gram LM (train n-gram models)
4. Perplexity (compute PPL by n and by language)
5. POS Tagging + Chunking (baseline + spaCy)
6. NER (actors, directors, titles)
7. Sentiment Classification (LogReg baseline)
8. Evaluation Deep-Dive (metrics, confusion matrix, analysis)

This file is triggered by run.py → run_full_pipeline()
"""
from .preprocessing import run_preprocessing
from .eda import run_eda
from .ngram_lm import run_ngram_models
from .perplexity import run_perplexity
from .pos_chunk_parser import run_pos_chunking
from .ner import run_ner
from .sentiment_model import run_sentiment_pipeline
from .evaluation import run_evaluation


def run_full_pipeline() -> None:
    print("\n STARTING FULL MULTILINGUAL PIPELINE\n")

    df_raw, df_long = run_preprocessing()     
    print(" Preprocessing done.\n")


    run_eda(df_long)                          

 
    ngram_results = run_ngram_models(df_long)  
    print(" N-gram language models completed.\n")


    run_perplexity(ngram_results)              
    print(" Perplexity evaluation completed.\n")


    run_pos_chunking(df_long)                  
    print(" POS / chunking completed.\n")

    ner_df = run_ner(df_long)
    print(" NER completed.\n")

    sentiment_df = run_sentiment_pipeline(df_long)
    print(" Sentiment classification completed.\n")


    run_evaluation(df_long, sentiment_df, ner_df)
    print(" Final evaluation completed.\n")

    print("\n Pipeline finished. Use outputs/ for figures & CSVs in the report.")
