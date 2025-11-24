**multilingual-movie-reviews** 
This project implements a complete multilingual Natural Language Processing (NLP) workflow for English (EN) and Spanish (ES) movie reviews.
The pipeline processes raw text, performs linguistic analysis, builds classical NLP models, and evaluates sentiment classification performance.

The project follows a fully modular architecture: each stage is isolated into its own Python file and orchestrated by pipeline.py and run.py.


# 1. Create a virtual environment
python -m venv .venv

2. Activate the environment

**Windows:**

* .venv\Scripts\activate


**macOS/Linux:**

* source .venv/bin/activate

# 3. Install project requirements
* pip install -r requirements.txt

# 4. Download required spaCy models
* python -m spacy download en_core_web_sm
* python -m spacy download es_core_news_sm

# Run the Full Pipeline
* python run.py


# This runs all stages automatically:

* Preprocessing
* EDA (with visualizations)
* N-gram Language Models + Perplexity
* POS Tagging + Chunking
* NER (English + Spanish)
* Sentiment Classification

# Final Evaluation

Outputs are saved into:

outputs/
├── eda/
├── ngram/
├── pos_parsing/
├── ner/
├── sentiment/
└── evaluation/
------------------------------------------------
**Project Features**
# 1. Preprocessing

* Load and clean the multilingual dataset
* Normalize text (lowercasing, punctuation removal)
* Generate a long-format dataset
* Save processed outputs

* Files:
preprocessing.py

# 2. Exploratory Data Analysis (EDA)

* Word count statistics
* Distribution of reviews per language
* Sentiment balance
* Automatic figure generation

* Files:
eda.py

* Outputs:
outputs/eda/*.png

# 3. N-Gram Language Modeling

* Train 1-gram, 2-gram, 3-gram models for EN and ES
* Save models using pickle
* Use them to compute perplexity

* Files:
ngram_lm.py, perplexity.py

* Outputs:
outputs/ngram/*.pkl

outputs/perplexity/perplexity_by_lang.csv

# 4. POS Tagging + Parsing / Chunking

* Uses spaCy models
* Computes baseline POS accuracy
* Extracts noun chunks
* Saves results

* Files:
pos_chunk_parser.py

* Outputs:
outputs/pos_parsing/

# 5. Named Entity Recognition (NER)

* Extracts people, movie titles, locations, organisations
* Saves a full entity table

* Files:
ner.py

* Outputs:
outputs/ner/ner_results.csv

# 6. Classical Sentiment Classification

* Bag-of-words + Logistic Regression
* Trains one combined multilingual model
* Predicts probabilities and labels
* Saves prediction CSV

* Files:
sentiment_model.py

* Output:
outputs/sentiment/sentiment_predictions.csv
(columns: text, lang, true, pred)

# 7. Final Evaluation

* Computes accuracy, precision, recall, F1
* Evaluates EN and ES separately
* Produces final sentiment confusion matrices

* Files:
evaluation.py

* Outputs:
outputs/evaluation/*
-------------------------------

**Full Project Structure**

project/
│
├── data/
│   ├── multilingual_reviews_cleaned.csv
│   ├── processed/
│   └── raw/
│
├── outputs/
│   ├── eda/
│   ├── evaluation/
│   ├── ner/
│   ├── ngram/
│   ├── perplexity/
│   ├── pos_parsing/
│   └── sentiment/
│
├── src/
│   ├── eda.py
│   ├── evaluation.py
│   ├── ner.py
│   ├── ngram_lm.py
│   ├── perplexity.py
│   ├── pipeline.py
│   ├── pos_chunk_parser.py
│   ├── preprocessing.py
│   └── sentiment_model.py
│
├── run.py
├── requirements.txt
└── README.md