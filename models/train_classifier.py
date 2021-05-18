import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import os
import pickle
import joblib
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from copy import deepcopy
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """
    Loads data from SQL Database

    Args:
        database_filepath: path to SQL database

    Returns:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for the 36 categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df_clean', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Processed text data

    Args:
        text (list): Text to be processed

    Returns:
       clean_tokens (list): list of clean tokens, that was tokenized, lower cased, stripped, 
       and lemmatized
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def build_model():
    """
    Build a Machine Learning pipeline with AdaBoost classifier GridSearch.
    Args:
        none
    Returns:
        ML pipeline (ml_model): that has gone through tokenization, count vectorization and
        TFIDTransofmration
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42))))
    ])
    parameters = {'vect__max_df': [0.90, 1.0],
                  'vect__min_df': [0.05, 0.1],
#                   'tfidf__use_idf':[True, False],
                  'classifier__estimator__learning_rate':[0.5, 1.5]
                 }
    cv = GridSearchCV(pipeline, param_grid= parameters, verbose=5, n_jobs = -1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance

    Args:
        model (ml_model): model to be evaluated
        X_test (dataframe): Input features, testing set
        Y_test (dataframe): Input target, testing set
        category_names (list): List of the categories 
    """
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, y_pred, target_names = category_names))

    

def save_model(model, model_filepath):
    """
    Save into a pickle file
    Args:
        model (ml_model):  Model to be saved
        model_filepath : path of the output 
    """

    pickle.dump(model, open(model_filepath, 'wb'))

def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        

        # Try to load saved gridSearchCV result if it exists
        try:
            print('Trying to load model...')
            model = joblib.load("gridSearchResult.pkl")
        except FileNotFoundError:    
            print('Model Not Found')
            print('Building model...')
            model = build_model()
        
            print('Training model...')
            model.fit(X_train, Y_train)
            
            print('Saving model...')
            joblib.dump(model, 'gridSearchResult.pkl')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()