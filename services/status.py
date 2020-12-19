import pandas as pd
from sklearn.model_selection import train_test_split

from services.globals import *


def remove_stopwords(text, stopwords):
    useful = [w for w in text if w not in stopwords]
    return useful


def getDoc(document):
    d = []
    for doc in document:
        d.append(getStem(doc))
    return d


def getStem(review):
    review = review.lower()
    tokens = tokenizer.tokenize(review)
    removed_stopwords = [w for w in tokens if w not in stop_words]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_review = ' '.join(stemmed_words)
    return clean_review

