import os
os.environ['TDML_DATAFRAME'] = 'pyspark'
import tdml
import copy
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

conf = SparkConf()
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

if __name__ == "__main__":
	path = '../data/dataset_31_credit-g.csv'
	df = spark.read.format('csv').option("header","true").option("inferSchema","true").load(path)
	ds = tdml.Dataset(df, label='class')
	ds.transform()
	ds.train_test_split()

	# check elements in the dataset
	print(ds)

	# check the mapping of categorical feature
	# print(ds.feature_mapping)
	
	# check the mapping of categorical label
	# print(ds.label_mapping)

	lr = LogisticRegression(solver='liblinear')
	lr.fit(ds.train_x, ds.train_y)
	print("Logistic regression accuracy: {}".format(lr.score(ds.test_x, ds.test_y)))

	sgd = SGDClassifier()
	sgd.fit(ds.train_x, ds.train_y)
	print("SGD accuracy: {}".format(lr.score(ds.test_x, ds.test_y)))

	svm = make_pipeline(StandardScaler(), SVC())
	svm.fit(ds.train_x, ds.train_y)
	print("SVM accuracy: {}".format(svm.score(ds.test_x, ds.test_y)))