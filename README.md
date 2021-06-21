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

### Flask Web App
