
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Takes two CSV files imports them as pandas dataframe and merges into a single df
    Args:
        messages_filepath(dataframe): messages_filepath: path to messages data
        categories_filepath(dataframe): path to categories data
       
    Returns:
        df(dataframe): dataset combining messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    Cleans the dataset

    Args:
        df(dataframe): dataset combining messages and categories data

    Returns:
        df(dataframe): Cleaned dataset
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = [category_name.split('-')[0] for category_name in row.values]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df['related'] = df['related'].replace(2,0)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save df as sqlite db
    Args:
        df(dataframe): cleaned dataset
        database_filename (dataframe): database name
    """
  engine = create_engine('sqlite:///{}'.format(database_filename))
  df.to_sql('df_clean', engine, index=False, if_exists='replace')

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