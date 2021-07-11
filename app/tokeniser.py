import sys
import re
import pandas as pd
import nltk
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from sqlalchemy import create_engine
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """
    Loads data from SQLite Database and convert to dataframe

    Args:
        database_filepath: file path to SQLite database

    Returns:
        X: Feature(s) of the messages dataframe that we use to predict or classify a message
        Y: Target - the class that the message belongs to
        category_names: class labels for the different classes
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y, y.columns

def tokenize(text):
    """
    converts text into individual words (tokens) that can be vectorised

    Args:
        text: Text document

    Returns:
       clean_tokens: list of clean tokens, that was tokenized, lower cased, stripped,
       and lemmatized
    """
    # normalized text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)
    # Replace url with a url placeholder string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    # Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    # List of clean tokens
    return clean_tokens