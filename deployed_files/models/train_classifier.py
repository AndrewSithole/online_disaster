import sys
import re
import pandas as pd
import nltk
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

def create_doc_term_matrix(data_list, vectoriser):
    """
    Method for visualising tokens during development
    Args:
        data_list: list of tokenised words (e.g. bag of words)
        vectoriser: the model used to transform the data (e.g. tfidfvectorizer)

    Returns:
        dataframe: a labelled dataframe delineating the transformed bag of words in a human readable form
    """
    doc_term_matrix = vectoriser.fit_transform(data_list)
    return pd.DataFrame(doc_term_matrix.toarray(), columns=vectoriser.get_feature_names())


def build_model():
    """
    Build a Machine Learning model for classifying messages
    Args:
        none
    Returns:
        pipeline: a pipeline that can be used to fit and predict
    """

    # Using TfidfVectorizer which combines the power of CountVectorizer and TfidfTransformer on one go
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42))))
    ])

    parameters = {'classifier__estimator__base_estimator__criterion': ['gini', 'entropy'],
                  'classifier__estimator__base_estimator__max_depth': [2, 4, 6, 8, 10, 12],
                  "classifier__estimator__n_estimators": [1, 2, 3]
                  }

    grid_search = GridSearchCV(pipeline, param_grid=parameters, verbose=2)

    return grid_search


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate model performance

    Args:
        model: model to be evaluated
        X_test: features from the testing set
        y_test: target from the testing set
        category_names: column list (with the class columns)
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    Save the model
    Args:
        model: the created model
        model_filepath: file path to save to
    """

    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        print('Best Criterion:',
              model.best_estimator_.get_params()['classifier__estimator__base_estimator__criterion'])
        
        print()
        print(model.best_estimator_.get_params()['classifier'])

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()