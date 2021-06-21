#import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, \
f1_score, precision_score, recall_score, make_scorer

from custom_transformer import DisasterWordExtrator, replace_urls, tokenize

import pickle


def load_data(database_filepath):
    
	 # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # Create dataframe by quering database
    df = pd.read_sql("SELECT * from messages", engine)
    
    # Feature selection
    X = df['message']
    
    # Choosing column names for multiobjective classification
    category_names=df.drop(['id','message','original','genre'], axis=1).columns
    
    # Target values to predict
    Y =df[category_names] 
    
    return X, Y, category_names


def build_model():
	"""
	INPUT:
		- None
		
	OUTPUT:
		- pipeline - A machine learning pipeline
	"""
	
	# model pipeline
	pipeline = Pipeline([
	('features', FeatureUnion([
        
		('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                                    ('tfdif', TfidfTransformer())
                                    ])),

		('disaster_words', DisasterWordExtrator())
        ])), 

	('clf', MultiOutputClassifier(estimator = RandomForestClassifier(n_jobs=-1)))
    ])
    	parameters = {'clf__estimator__max_features':['sqrt', 0.5],
              'clf__estimator__n_estimators':[50, 100]}

        cv = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 5, n_jobs = 10)
   
        return cv
    
    
def get_scores(y_true, y_pred):
	"""
	Returns the accuracy, precision and recall and f1 scores of the two same shape numpy 
	arrays `y_true` and `y_pred`.
	INPUTS:
		- y_true - Numpy array object - A (1 x n) vector of true values
		- y_pred - Numpy array object - A (1 x n) vector of predicted values
        
	OUPUT:
		- dict_scores - Python dict - A dictionary of accuracy, precision and recall and f1 
		scores of `y_true` and `y_pred`.
	"""
    
	# Compute the accuracy score of y_true and y_pred
	accuracy = accuracy_score(y_true, y_pred)
    
	# Compute the precision score of y_true and y_pred
	precision = round(precision_score(y_true, y_pred, average='micro'))
    
	# Compute the recall score of y_true and y_pred
	recall = recall_score(y_true, y_pred, average='micro')
    
	# Compute the recall score of y_true and y_pred
	f_1 = f1_score(y_true, y_pred, average='micro')
    
	# A dictionary of accuracy, precision and recall and f1 scores of `y_true` and `y_pred`
	dict_scores = {
		'Accuracy': accuracy, 
		'Precision': precision, 
		'Recall': recall, 
		'F1 Score': f_1
	}
    
	return dict_scores


def evaluate_model(model, X_test, Y_test, category_names):  
    
	"""
	INPUT:
		- model - A machine learning pipeline
		- X_test - Numpy array - A vector of str objects
		- Y_test - Numpy array - A matrix of zeros and ones.
		- category_names - Numpy array - A vector of str objects
	OUTPUT:
		- df - Pandas DataFrame - A DataFrame of the accuracy, precision, recall, and 
		f1 scores for each category in category_names. 
	"""
    
	y_pred = model.predict(X_test)
	print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
	"""
	Saves the machine learning pipeline `model` to disk with the name model_filepath
	INPUTS:
		model - A machine learning Pipeline object
		model_filepath - A Python str object - the name of the input `model` saved on disk
		
	OUTPUT:
		None
	"""
	
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n    DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
		print('Building model...')
		model = build_model()
        
		print('Training model...')
		model.fit(X_train, Y_train)
        
		print('Evaluating model...')
		evaluate_model(model, X_test, Y_test, category_names)

		print('Saving model...\n    MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)

		print('Trained model saved!')

	else:
		print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
	main()
