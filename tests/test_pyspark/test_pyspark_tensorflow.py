import os
import torch
os.environ['MLLIB'] = 'tensorflow'
os.environ['DATAFRAME'] = 'pyspark'
import tdml
import unittest

import pyspark
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

conf = SparkConf()
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

class TestPySpark(unittest.TestCase):

	def test_simple(self):
		path = "data/simple.csv"
		df_pyspark = spark.read.format('csv').option("header","true").option("inferSchema","true").load(path)

		ds = tdml.Dataset(df_pyspark, label='Evaluation')
		ds.transform()
		self.assertEqual(ds.feature.shape, (10, 4))
		self.assertEqual(len(ds.label), 10)
		self.assertEqual(len(ds.label_mapping), 3)

	def test_goog(self):
		path = "data/GOOG.csv"
		df_pyspark = spark.read.format('csv').option("header","true").option("inferSchema","true").load(path)
		df_pyspark.drop('Date', 'Adj Close')

		ds = tdml.Dataset(df_pyspark, label='Close')
		ds.transform()
		self.assertEqual(ds.feature.shape, (252, 6))
		self.assertEqual(len(ds.label), 252)

	def test_simple_text(self):
		path = "data/simple_text.csv"
		df_pyspark = spark.read.format('csv').option("header","true").option("inferSchema","true").load(path)

		ds = tdml.Dataset(df_pyspark, label='Broken', text="Description")
		ds.transform()
		self.assertEqual(len(ds.feature_mapping['Description']), 16)
		self.assertEqual(ds.feature_to_idx('Description'), (4, 20))
		self.assertEqual(ds.idx_to_feature((4, 20)), 'Description')

if __name__ == '__main__':
	unittest.main()