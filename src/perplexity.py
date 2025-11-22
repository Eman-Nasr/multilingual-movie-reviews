import os
import pickle
import pandas as pd
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab")


# ---------------------------------------------------------
# Clean + tokenize utility
# ---------------------------------------------------------
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    return word_tokenize(text)


# ---------------------------------------------------------
# Compute perplexity safely
# ---------------------------------------------------------
def compute_ppl(model, tokens, n):
    try:
        padded = list(padded_everygram_pipeline(n, [tokens]))[0]
        return model.perplexity(tokens)
    except:
        return None


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def run_perplexity(
    input_csv="data/processed/reviews_long.csv",
    model_dir="outputs/ngram",
    output_dir="outputs/perplexity"
):
    print("\n=== PERPLEXITY STAGE ===")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # small sample (balanced)
    df_sample = (
        df.groupby("lang")
          .head(15)
          .reset_index(drop=True)
    )

    results = []

    for lang in ["en", "es"]:
        print(f"\n Computing perplexity for {lang.upper()} ...")

        reviews = df_sample[df_sample["lang"] == lang]["review_clean"]

        for n in [1, 2, 3]:
            model_path = os.path.join(model_dir, f"{lang}_{n}gram.pkl")

            if not os.path.exists(model_path):
                print(f" Missing model → {model_path}")
                continue

            model = pickle.load(open(model_path, "rb"))

            # average perplexity for n=1/2/3
            ppl_scores = []
            for text in reviews:
                tokens = clean_and_tokenize(text)
                if tokens:
                    ppl = compute_ppl(model, tokens, n)
                    if ppl is not None:
                        ppl_scores.append(ppl)

            avg_ppl = sum(ppl_scores) / len(ppl_scores) if ppl_scores else None

            results.append({
                "lang": lang,
                "ngram": n,
                "perplexity": avg_ppl
            })

            print(f"   → {n}-gram ppl = {avg_ppl}")

    # Save results
    df_out = pd.DataFrame(results)
    df_out_path = os.path.join(output_dir, "perplexity_by_lang.csv")
    df_out.to_csv(df_out_path, index=False)

    print(f"\n Saved perplexity results → {df_out_path}")
    print("=== PERPLEXITY COMPLETE ===\n")

    return df_out


if __name__ == "__main__":
    run_perplexity()
