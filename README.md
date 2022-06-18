# Disaster-Response-Pipeline


### ETL Pipeline

In a Python script, process_data.py:

- Loads the messages and categories datasets<br/>
- Merges the two datasets<br/>
- Cleans the data<br/>
- Stores it in a SQLite database

### ML Pipeline

In a Python script, train_classifier.py:

- Loads data from the SQLite database<br/>
- Splits the dataset into training and test sets<br/>
- Builds a text processing and machine learning pipeline<br/>
- Trains and tunes a model using GridSearchCV<br/>
- Outputs results on the test set<br/>
- Exports the final model as a pickle file


### Project Motivation

In this project, I apply skills I learned in Data Engineering Section to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.


#### Instructions: 
- Run run.py directly if DisasterResponse.db and claasifier.pkl already exist.

- Run the following commands in the project's root directory to set up your database and model.

#### To run ETL pipeline that cleans data and stores in database python 

- data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

#### To run ML pipeline that trains classifier and saves 

- python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

#### Run the following command in the app's directory to run your web app. 

- python run.py
