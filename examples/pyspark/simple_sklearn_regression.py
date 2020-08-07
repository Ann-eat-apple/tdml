import os
os.environ['DATAFRAME'] = 'pyspark'
import tdml
import copy
from sklearn.linear_model import LinearRegression, ElasticNet

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

conf = SparkConf()
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

if __name__ == "__main__":
	path = '../data/GOOG.csv'
	df = spark.read.format('csv').option("header","true").option("inferSchema","true").load(path)

	# Drop Date and Adj Close fields
	df = df.drop('Date', 'Adj Close')

	# Predicting the Close price
	ds = tdml.Dataset(df, label='Close')
	ds.transform()

	# Split the dataset into train and test sets with seed 0
	ds.train_test_split(seed=0)

	# Print a brief view of the dataset
	print(ds)

	X_train, y_train = ds.train_x, ds.train_y
	X_test, y_test = ds.test_x, ds.test_y

	log = "The coefficient of determination of {} data is {}"
	for model in [LinearRegression, ElasticNet]:
		print("Using {}:".format(model.__name__))
		m = model().fit(X_train, y_train)
		train_score = m.score(X_train, y_train)
		print(log.format("train", round(train_score, 5)))
		test_score = m.score(X_test, y_test)
		print(log.format("test", round(test_score, 5)))
	print()

	# Split to train / val / test three sets
	ds.train_val_test_split(seed=0)
	print(ds)

	X_train, y_train = ds.train_x, ds.train_y
	X_val, y_val = ds.val_x, ds.val_y
	X_test, y_test = ds.test_x, ds.test_y

	max_score = -1
	best_model = None

	for model in [LinearRegression, ElasticNet]:
		print("Using {}:".format(model.__name__))
		m = model().fit(X_train, y_train)
		val_score = m.score(X_val, y_val)
		print(log.format("val", round(val_score, 5)))
		if(val_score > max_score):
			max_score = val_score
			best_model = copy.deepcopy(m)
	print("{} has the highest val score and it has test score: {}".\
		format(best_model.__class__.__name__, round(best_model.score(X_test, y_test), 5)))
