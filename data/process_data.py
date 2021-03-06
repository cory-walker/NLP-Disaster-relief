# Import --------------------------------
import sys
import pandas as pd
from sqlalchemy import create_engine

# Functions -----------------------------

def load_data(messages_filepath, categories_filepath):
    # Loads data from the specified files and produces a dataframe
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    
    categories = categories["categories"].str.split(";", expand = True)
    
    row = categories.iloc[0]
    category_colnames = row.str[:-2].tolist()
    
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis = 1)
    
    return df

def clean_data(df):
    # Cleans the dataframe by dropping duplicates, and NaNs
    # Original text is NaN for everything except direct messages. This can be ignored
    
    df.drop_duplicates(subset = 'id', inplace = True)
    df["original"].fillna(" ", inplace=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df


def save_data(df, database_filename):
    # Save the dataframe to a database
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("messages", engine, index=False)  


def main():
    #Main ETL code
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {} ...'
              .format(messages_filepath, categories_filepath), end='')
        df = load_data(messages_filepath, categories_filepath)
        print('Complete')
        
        print('Cleaning data...', end='')
        df = clean_data(df)
        print('Complete')
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath), end='')
        save_data(df, database_filepath)
        print('Complete')
        
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