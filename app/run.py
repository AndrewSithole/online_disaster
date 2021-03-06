import json
import plotly
import re
import pandas as pd
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Scatter
from tokeniser import tokenize

import nltk

from sqlalchemy import create_engine

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

app = Flask(__name__)


def tokenizew(text):
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


# load data
engine = create_engine('sqlite:///./messages.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Number of Messages per Category
    message_counts = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1).sum().sort_values()
    message_names = list(message_counts.index)

    # Top ten categories count
    top_category_count = df.iloc[:, 4:].sum().sort_values(ascending=False)[1:11]
    top_category_names = list(top_category_count.index)

    # extract categories
    category_map = df.iloc[:, 4:].corr().values
    category_names = list(df.iloc[:, 4:].columns)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # Added more visuals inspired by this post: https://katba-caroline.com/disaster-response-message-classification-pipelines-nltk-flask/
    graphs = [
        {
            'data': [
                Bar(
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
        },

        {
            'data': [
                Bar(
                    x=message_counts,
                    y=message_names,
                    orientation='h',

                )
            ],

            'layout': {
                'title': 'Number of Messages per Category',

                'xaxis': {
                    'title': "Number of Messages"

                },
            }
        },

        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            'layout': {
                'title': 'Top Ten Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },

        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=category_map
                )
            ],

            'layout': {
                'title': 'Correlation Heatmap of Categories',
                'xaxis': {'tickangle': -45}
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    prediction = model.predict([query])
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        prediction=prediction,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
