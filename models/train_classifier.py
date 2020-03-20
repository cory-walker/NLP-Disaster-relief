# Imports ------------------------------------------------
import sys
import pandas as pd
import numpy as np
import re
import nltk

from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Functions ------------------------------------------------

def load_data(database_filepath):
    # Load data from the SQL lite database and produce
    # messages, category values, and category names
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    categories = list(df.columns[4:])
    return X, Y, categories


def tokenize(text):
    # Turns messages into word tokens
    
    #Remove common stop words
    stop_words = stopwords.words("english")
    
    #Remove non-letters and non-numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize, then lowercase and remove extra whitespace in tokens
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word.lower().strip() not in stop_words]

    return tokens

def build_model():
    # Builds the Multi output classifer random forest model
    
    #Create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))
    ])
    
    #Set parameters for grid search
    parameters = {
         #'tfidf__smooth_idf':[True, False],
         #'clf__estimator__estimator__C': [1, 2, 5],
         #'vect__ngram_range': ((1, 1), (1, 2))
         #'vect__max_df': (0.5, 0.75, 1.0),
         #'vect__max_features': (None, 5000, 10000),
         #'tfidf__use_idf': (True, False),
         #'clf__n_estimators': [50, 100, 200],
         #'clf__min_samples_split': [2, 3, 4],
    }

    # Generate model
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # Produce accuracy measures for the model
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))
    print("\nAccuracy")
    for i in range(Y_test.shape[1]):
        print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], Y_pred[:,i])))


def save_model(model, model_filepath):
    # Saves the model
    
    dump(model, model_filepath)


def main():
    # Main process, to keep everything running in order
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {} ...'.format(database_filepath), end='')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
        print('Complete')
        
        print('Building model...', end='')
        model = build_model()
        print('Complete')
        
        print('Training model...', end='')
        model.fit(X_train, Y_train)
        print('Complete')
        
        print('Evaluating model...', end='')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Complete')
        
        print('Saving model...\n    MODEL: {} ...'.format(model_filepath), end='')
        save_model(model, model_filepath)
        print('Complete')
        
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

# Main ------------------------------------------------

if __name__ == '__main__':
    main()