
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Merges data from 2 sources (messages_filepath and categories_filepath)
     on id returns the resulting dataframe
    Args:
        messages_filepath: path to the messages csv
        categories_filepath: path to the categories csv
       
    Returns:
        df: the merged dataset
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    return df


def get_last_int(string_value):
    """
    function to get last character and return binary (int) value
    we return binary because the classification_report function expects binary values from our y_test
    think of it as part of the data cleaning process
    Args:
        string_value:

    Returns:
        binary int
    """
    return 0 if(int(string_value[-1])==0) else 1

def clean_data(df):
    """
    Cleans the dataframe

    Args:
        df: input dataframe

    Returns:
        df: output dataframe with duplicates removed and feature columns properly formatted
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    columns = []
    for cat in list(categories.loc[0]):
        columns.append(cat[:(len(cat)) - 2])
    categories.columns = columns

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(get_last_int)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)
    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Saves dataframe as sqlite database
    Args:
        df: dataframe
        database_filename: intended database file name (including path)
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()