
Flipkart Product Reviews — Sentiment Analysis

This project demonstrates sentiment analysis of Flipkart product reviews using NLP preprocessing,
NLTK’s VADER SentimentIntensityAnalyzer, and a simple Multinomial Naive Bayes classifier.

---

Contents
--------
- Flipkart_Sentiment_Analysis.ipynb   # Jupyter Notebook with full workflow
- data/FlipkartData.csv               # Input dataset (provide your own)
- flipkart_with_vader_and_clean.csv   # Example output (preprocessed + VADER scores)

---

Setup
-----
1. Install Python 3.8+ and Jupyter Notebook (or JupyterLab).
2. Install dependencies:

    pip install pandas numpy matplotlib seaborn scikit-learn nltk

3. Download NLTK data (run once inside Python):

    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')

---

Usage
-----
1. Open `Flipkart_Sentiment_Analysis.ipynb` in Jupyter Notebook or JupyterLab.
2. Update the path to your dataset in the cell:

       DATA_PATH = r'D:\projects\FlipkartData.csv'

3. Run all cells in order to:
   - Preprocess the reviews (stopwords removal, stemming, negation handling)
   - Compute VADER sentiment scores (compound, pos, neu, neg)
   - Vectorize reviews (bag of words, n-grams)
   - Train and evaluate a Multinomial Naive Bayes classifier
   - Display confusion matrix and metrics

4. The notebook saves an augmented dataset with preprocessed text and VADER scores to:

       flipkart_with_vader_and_clean.csv

---

Notes
-----
- Ensure the dataset has `Summary` and `Sentiment` columns.
- The notebook maps `positive` reviews to 1 and `negative` to 0 for binary classification.
- Use `ngram_range=(1,2)` in CountVectorizer for unigrams + bigrams.
- Adjust `max_features` to balance accuracy vs memory usage.

---

Future Improvements
-------------------
- Extend labels to three classes: positive, neutral, negative.
- Experiment with TF-IDF features.
- Try advanced models (Logistic Regression, SVM, LSTM, Transformers).
- Handle class imbalance with resampling or class weights.

---

License
-------
This project is provided for educational purposes. Add an open-source license (e.g., MIT) if publishing.
