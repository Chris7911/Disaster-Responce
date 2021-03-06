import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # list category's names from the first row
    row = categories.iloc[[0]]
    category_colnames = list(row.apply(lambda x: x.str.split('-').str.get(0)[0]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to 0 or 1
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str.split('-').str.get(1))

    # replace categories column in df with new category columns
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    # save the clean dataset to an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    table_name = '{}'.format(database_filename.split('.')[0])
    engine.execute("DROP TABLE IF EXISTS {}".format(table_name))
    df.to_sql(table_name, engine, index=False)


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