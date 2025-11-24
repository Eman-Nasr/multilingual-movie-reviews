import spacy
import random
import pandas as pd
import nltk
from nltk import CFG

# ------------------------------------------------------------
# CFG GRAMMAR (simple demo grammar)
# ------------------------------------------------------------

GRAMMAR = CFG.fromstring("""
S -> NP VP
NP -> DT NN | DT JJ NN | PRP
VP -> VBD NP | VBZ NP | VBP NP | VBD | VBZ | VBP
DT -> 'the' | 'a'
JJ -> 'good' | 'bad' | 'great'
NN -> 'movie' | 'story' | 'actor'
PRP -> 'i' | 'you'
VBD -> 'liked' | 'hated' | 'watched'
VBZ -> 'is' | 'was'
VBP -> 'am' | 'are'
""")

parser = nltk.ChartParser(GRAMMAR)


# ------------------------------------------------------------
# CFG PARSER ANALYSIS
# ------------------------------------------------------------

def analyse_cfg_parser(sentences, lang):
    print(f"\n========== CFG PARSING EXAMPLES ({lang.upper()}) ==========")

    for sent in sentences[:3]:
        tokens = [t.lower() for t in sent.split() if t.isalpha()]
        print(f"\nSENTENCE: {sent}")
        print(f"TOKENS: {tokens}")

        try:
            trees = list(parser.parse(tokens))
            if trees:
                for tree in trees:
                    print(tree)
            else:
                print("   (No valid parse under this grammar)")
        except Exception as e:
            print(f"   Parser error: {e}")

        print("----------------------------------------------------------")


# ------------------------------------------------------------
# POS + NOUN CHUNK ANALYSIS
# ------------------------------------------------------------

def analyse_chunks(nlp, sentences, lang):
    print(f"\n========== NOUN CHUNKS ({lang.upper()}) ==========")

    for sent in sentences[:3]:
        doc = nlp(sent)
        print(f"\nSENT: {sent}")
        for chunk in doc.noun_chunks:
            print(" NP:", chunk.text)

        print("----------------------------------------------------------")


# ------------------------------------------------------------
# MAIN FUNCTION CALLED BY PIPELINE
# ------------------------------------------------------------

def run_pos_and_parsing(df_long):
    print("\n==============================")
    print("STAGE 3 POS TAGGING + PARSING / CHUNKING")
    print("==============================")

    # Load spaCy models
    print(" Loading spaCy models (en + es)…")
    nlp_en = spacy.load("en_core_web_sm")
    nlp_es = spacy.load("es_core_news_sm")

    # Sample sentences for analysis
    en_sents = df_long[df_long["lang"] == "en"]["text"].sample(3, random_state=42).tolist()
    es_sents = df_long[df_long["lang"] == "es"]["text"].sample(3, random_state=42).tolist()

    # POS baseline accuracy (simple baseline model)
    baseline_rows = []

    for lang, nlp, sents in [("en", nlp_en, en_sents), ("es", nlp_es, es_sents)]:

        print(f"\n Evaluating POS baseline vs spaCy for {lang.upper()} (200 samples)…")

        sample_df = df_long[df_long["lang"] == lang].sample(200, random_state=2)
        gold_tags = []
        baseline_tags = []

        for text in sample_df["text"]:
            doc = nlp(text)
            gold_tags.extend([t.pos_ for t in doc])

            # very crude baseline: all words tagged as NOUN
            baseline_tags.extend(["NOUN"] * len(doc))

        correct = sum(g1 == g2 for g1, g2 in zip(gold_tags, baseline_tags))
        acc = correct / len(gold_tags)

        print(f"   Baseline accuracy vs spaCy POS (pseudo-gold): {acc:.3f}")
        baseline_rows.append([lang, "pos_baseline_accuracy", acc])

        # noun chunks
        analyse_chunks(nlp, sents, lang)

        # CFG parsing
        analyse_cfg_parser(sents, lang)

    # Save POS evaluation to CSV
    df_metrics = pd.DataFrame(baseline_rows, columns=["lang", "metric", "value"])
    df_metrics.to_csv("outputs/pos_parsing/pos_baseline_metrics.csv", index=False)

    print("\n POS + Parsing / Chunking stage complete.")
    print(" POS / Chunking completed.\n")
