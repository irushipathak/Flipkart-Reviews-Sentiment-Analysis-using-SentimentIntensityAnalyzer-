
Flipkart Product Reviews — Sentiment Analysis (VADER)

Project: Sentiment analysis of Flipkart product reviews using NLTK's SentimentIntensityAnalyzer (VADER).

---

Project summary
This repository contains code and data to explore and evaluate sentiment analysis on Flipkart product reviews using the rule-based VADER sentiment analyzer (nltk.sentiment.SentimentIntensityAnalyzer). The goal is to:

- Load Flipkart reviews (CSV), compute sentiment scores (compound / pos / neg / neu) per review.
- Map numeric ratings (Rate) to ground-truth sentiment labels and evaluate VADER using accuracy, F1-score and a confusion matrix.
- Produce simple visualizations (bar plots and confusion matrix) and a small exploratory analysis.

This implementation is intentionally lightweight — VADER works well for social / short-text sentiment and is a fast baseline before moving to ML models.

---

Contents
README.txt
data/Flipkart.csv            # (place extracted CSV here)
code/sentiment_analysis.py   # your main python notebook/script
notebooks/analysis.ipynb     # optional jupyter notebook
requirements.txt
results/                     # generated figures / metrics
LICENSE

---

Getting started (one-time setup)

1. Extract the provided .rar
   - Place the Flipkart.csv inside a data/ folder at the repository root (data/Flipkart.csv).

2. Create a Python environment (recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

3. If you don't have requirements.txt, install the minimal packages:

pip install pandas numpy matplotlib seaborn scikit-learn nltk

4. NLTK data (run once) — VADER and tokenizers require additional NLTK data. In a Python shell or at the top of your notebook/script run:

import nltk
nltk.download('vader_lexicon')
# Optional (only needed if your code uses tokenization/POS/NER):
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

---

How to run

Option A — Jupyter Notebook
1. Open notebooks/analysis.ipynb (or create a new notebook). Split the script into logical cells (imports, data load, scoring, evaluation, plots).
2. Run cells with Shift + Enter. Use InteractiveShell.ast_node_interactivity = "all" if you want every statement in a cell to display.

Option B — Command line (run script)
Assuming code/sentiment_analysis.py is the main script and data/Flipkart.csv exists:

python code/sentiment_analysis.py

The script can be written to save outputs (CSV with scores, plots in results/, and a simple text file with metrics).

---

Example: Minimal usage (script snippet)

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# load
df = pd.read_csv('data/Flipkart.csv', encoding='ISO-8859-1')

# safe summary
df['Summary'] = df['Summary'].fillna('').astype(str)

sia = SentimentIntensityAnalyzer()
scores = df['Summary'].apply(lambda t: sia.polarity_scores(str(t))).tolist()
scores_df = pd.DataFrame(scores)
merged = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

# map Rate -> true label
def rate_to_label(r):
    r = float(r)
    if r >= 4:
        return 'positive'
    elif r == 3:
        return 'neutral'
    else:
        return 'negative'

merged['true_label'] = merged['Rate'].apply(rate_to_label)

# predicted label from compound
merged['pred_label'] = merged['compound'].apply(lambda c: 'positive' if c>=0.05 else ('negative' if c<=-0.05 else 'neutral'))

# compute metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report
print('Accuracy:', accuracy_score(merged['true_label'], merged['pred_label']))
print('F1 (weighted):', f1_score(merged['true_label'], merged['pred_label'], average='weighted'))
print(classification_report(merged['true_label'], merged['pred_label']))

---

Notes & tips
- Encoding issues: If you see weird characters (non-UTF), try encoding='ISO-8859-1' when reading the CSV or errors='replace'.
- VADER thresholds: Default thresholds used here are compound >= 0.05 → positive; <= -0.05 → negative; otherwise neutral. You can tune thresholds for your dataset.
- Using full review text: Summary is often short — consider using the Review (full-text) column for better signals.
- Imbalance: Ratings may be skewed toward positive; consider using stratified sampling or class-weighted metrics if you train ML models.

---

Project improvements (future)
- Train a supervised classifier (TF-IDF + Logistic Regression / SVM / LGBM) and compare to VADER baseline.
- Use more advanced pre-processing: emoji handling, negation handling, lemmatization, spelling correction.
- Add cross-validation and hyper-parameter search for supervised models.

---

Requirements (example requirements.txt)
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk

---

License & Contact
This repository is provided for educational purposes. Add a license (e.g., MIT) if you plan to publish.
If you want changes to the README (more examples, badges, contributor list, or a short abstract for a report), tell me what to add and I will update it.
