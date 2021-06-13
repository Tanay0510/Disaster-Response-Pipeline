# Disaster-Response-Pipeline


### ETL Pipeline

In a Python script, process_data.py:

Loads the messages and categories datasets </n>
Merges the two datasets </n>
Cleans the data
Stores it in a SQLite database

### ML Pipeline

In a Python script, train_classifier.py:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

### Flask Web App
