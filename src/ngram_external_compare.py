# ngram_external_compare.py

import nltk
from nltk.corpus import brown, udhr
import re
from collections import Counter, defaultdict
import math
import pandas as pd

nltk.download("brown")
nltk.download("udhr")


# ----------------------------------------------------
# CLEAN TOKENS
# ----------------------------------------------------
def clean_tokens(tokens):
    cleaned = []
    for t in tokens:
        t = t.lower()
        t = re.sub(r"[^a-záéíóúüñ]+", "", t)
        if t:
            cleaned.append(t)
    return cleaned


# ----------------------------------------------------
# BUILD N-GRAM MODEL
# ----------------------------------------------------
def build_ngram(tokens, n=2):
    model = defaultdict(Counter)
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i+n])
        nxt = tokens[i+n]
        model[context][nxt] += 1
    return model


# ----------------------------------------------------
# PERPLEXITY
# ----------------------------------------------------
def perplexity(model, tokens, n=2, smooth=1):
    vocab = set(tokens)
    log_prob = 0
    denom = 0

    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i+n])
        nxt = tokens[i+n]
        count_context = sum(model[context].values())
        count_next = model[context][nxt]

        prob = (count_next + smooth) / (count_context + smooth * len(vocab))

        if prob <= 0:
            return float("inf")

        log_prob += -math.log(prob)
        denom += 1

    return math.exp(log_prob / denom)


# ----------------------------------------------------
# LOAD EXTERNAL DATASETS
# ----------------------------------------------------
def load_external():
    # English — Brown corpus (50k tokens)
    brown_tokens = clean_tokens(brown.words()[:50000])

    # Spanish — UDHR Spanish version
    udhr_raw = udhr.raw("Spanish_Espanol-Latin1")
    es_tokens = clean_tokens(udhr_raw.split())

    return brown_tokens, es_tokens


# ----------------------------------------------------
# RUN COMPARISON
# ----------------------------------------------------
def run_external_comparison():
    brown_tokens, es_tokens = load_external()

    rows = []

    for lang, toks in [
        ("external_en_brown", brown_tokens),
        ("external_es_udhr", es_tokens)
    ]:
        for n in [1, 2, 3]:
            model = build_ngram(toks, n)
            ppl = perplexity(model, toks, n)
            rows.append({
                "language": lang,
                "ngram": n,
                "perplexity": ppl
            })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/perplexity/external_perplexity.csv", index=False)
    print(df)
    print("\nExternal perplexity comparison completed.")


# ----------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------
if __name__ == "__main__":
    run_external_comparison()
