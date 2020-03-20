# Disaster Response Pipeline Project



### Project Summary

This project provides Natural Language Processing (NLP) analysis on disaster response messages provided by <a href="https://www.figure-eight.com/">Figure Eight</a> in order to help classification for disaster response teams

A website is also provided where you can enter a message into the text box, and then recieve the classification determined by the machine learning algorithm. The main page also includes two live calculated bar charts concerning the training data set message genre and categories 

### Files
<ul>
  <li>data</li>
  <ul>
    <li>DisasterResponse.db <i>SQLlite database of combined message data and categories</i></li>
    <li>disaster_categories.csv <i>Message category raw data</i></li>
    <li>disaster_messages.csv <i>Message raw text</i></li>
    <li>process_data <i>Python script for processing the raw data and generating the DisasterResponse.db</i></li>
  </ul>
  <li>models</li>
  <ul>
    <li>classifier.pkl <i>Saved message classification model</i></li>
    <li>train_classifier.py <i>Python script for training the classifier model</i></li>
  </ul>
  <li>app</li>
  <ul>
    <li>templates <i>HTML templates folder</i></li>
    <li>run.py <i>Python script for website generation</i></li>
  </ul>
</ul>
  

### Required libraries
<ul>
  <li>nltk 3.2.5</li>
  <li>numpy 1.12.1</li>
  <li>pandas 0.23.3</li>
  <li>scikit-learn 0.19.1</li>
  <li>sqlalchemy 1.2.18</li>
</ul>

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. For the Udacity workspace: Get WORKSPACEID and WORKSPACEDOMAAIN by running env | grep WORK

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/ or http://WORKSPACEID-3001.WORKSPACEDOMAIN in the Udacity workspace

### Acknowledgements

I would like to thank the following groups for their part in this project:
<a href="https://www.figure-eight.com/">Figure Eight</a>: Dataset
<a href="https://www.udacity.com/">Udacity</a>: Training, advice, and review
<a href="https://stackoverflow.com/">stackoverflow</a>: Infinite wisdom




