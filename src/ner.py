import spacy
import pandas as pd
import os


# -------------------------
# Load multilingual spaCy models
# -------------------------

def load_spacy_models():
    print("\n[NER] Loading spaCy models...")

    try:
        nlp_en = spacy.load("en_core_web_sm")
    except:
        raise RuntimeError("English spaCy model missing. Run: python -m spacy download en_core_web_sm")

    try:
        nlp_es = spacy.load("es_core_news_sm")
    except:
        raise RuntimeError("Spanish spaCy model missing. Run: python -m spacy download es_core_news_sm")

    print("[NER] Models loaded successfully.")

    return nlp_en, nlp_es


# -------------------------
# Extract relevant entities
# -------------------------

def extract_entities(text, lang, nlp_en, nlp_es):
    model = nlp_en if lang == "en" else nlp_es
    doc = model(text)

    people = []
    movies = []
    orgs = []

    for ent in doc.ents:
        if ent.label_ in ["PERSON"]:
            people.append(ent.text)
        elif ent.label_ in ["WORK_OF_ART"]:
            movies.append(ent.text)
        elif ent.label_ in ["ORG"]:
            orgs.append(ent.text)

    return people, movies, orgs


# -------------------------
#  NER pipeline
# -------------------------

def run_ner(input_path="data/processed/reviews_long.csv",
            output_path="outputs/ner/ner_results.csv"):

    print("\n========== NER STAGE ==========")

    os.makedirs("outputs/ner", exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"[NER] Loaded {len(df)} reviews.")

    # Load models
    nlp_en, nlp_es = load_spacy_models()

    all_people = []
    all_movies = []
    all_orgs = []

    for i, row in df.iterrows():
        text = str(row["text"])
        lang = row["lang"]

        people, movies, orgs = extract_entities(text, lang, nlp_en, nlp_es)

        all_people.append(people)
        all_movies.append(movies)
        all_orgs.append(orgs)

    df["people"] = all_people
    df["movies"] = all_movies
    df["organizations"] = all_orgs

    df.to_csv(output_path, index=False)

    print(f"[NER] Saved NER results → {output_path}")
    print("[NER] Example extracted entities:")
    print(df[["lang", "text", "people", "movies"]].head())

    return df


if __name__ == "__main__":
    run_ner()
