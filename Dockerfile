FROM python:3.9


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# download pre-trained embedding model
RUN python -c 'from sentence_transformers import SentenceTransformer; s = SentenceTransformer("all-mpnet-base-v2");'

# download generic news data
RUN python -c 'from sklearn.datasets import fetch_20newsgroups; fetch_20newsgroups(return_X_y=True);'

# Entity extraction requirements
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words');"


COPY OpenNews/utils OpenNews/utils
COPY OpenNews/ingestion OpenNews/ingestion

COPY OpenNews/analysis OpenNews/analysis
COPY OpenNews/flask_app OpenNews/flask_app

COPY configs configs
