import json
import pandas as pd
import joblib
import pickle
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

import plotly.graph_objects as graph_objects

import plotly.utils as pl_utils

from sqlalchemy import create_engine

app = Flask(__name__)


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

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        if name == 'tokenize':

            return tokenize

        return super().find_class(module, name)

# load data
engine = create_engine('sqlite:////home/andysee/disasterapp.andysee.com/data/messages.db')
df = pd.read_sql_table('messages', engine)

# load model
model = CustomUnpickler(open('/home/andysee/disasterapp.andysee.com/models/classifier.pkl', 'rb')).load()
#model = joblib.load("/home/andysee/disasterapp.andysee.com/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                graph_objects.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=pl_utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()