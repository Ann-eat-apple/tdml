import os
import numpy as np
import tdml
import unittest
from utils import *

class TestDatasetNumPy(unittest.TestCase):

	def test_no_label(self):
		df = generate_simple_pd_dataframe()

		# no label, only feature
		ds = tdml.Dataset(df)
		ds.transform()
		self.assertEqual(ds.feature.shape, (10, 5))
		self.assertEqual(ds.num_sample, 10)
		self.assertEqual(ds.num_feature, 5)
		self.assertEqual(ds.num_label, 0)

	def test_label(self):
		df = generate_simple_pd_dataframe()

		# categorical label
		ds = tdml.Dataset(df, label='Evaluation')
		ds.transform()
		self.assertEqual(ds.feature.shape, (10, 4))
		self.assertEqual(ds.num_sample, 10)
		self.assertEqual(ds.num_feature, 4)
		self.assertEqual(ds.num_label, 3)

		# boolean label
		ds = tdml.Dataset(df, label='Dark')
		ds.transform()
		self.assertEqual(ds.feature.shape, (10, 4))
		self.assertEqual(ds.num_sample, 10)
		self.assertEqual(ds.num_feature, 4)
		self.assertEqual(ds.num_label, 2)

	def test_specified_feature(self):
		df = generate_simple_pd_dataframe()

		# one feature
		ds = tdml.Dataset(df, feature='Brand')
		ds.transform()
		self.assertEqual(ds.feature.shape, (10, 1))
		self.assertEqual(ds.num_sample, 10)
		self.assertEqual(ds.label, None)
		self.assertEqual(ds.num_feature, 1)
		self.assertEqual(ds.num_label, 0)

		# two features
		ds = tdml.Dataset(df, feature=['Brand', 'Price'])
		ds.transform()
		self.assertEqual(ds.feature.shape, (10, 2))
		self.assertEqual(ds.num_sample, 10)
		self.assertEqual(ds.label, None)
		self.assertEqual(ds.num_feature, 2)
		self.assertEqual(ds.num_label, 0)

	def test_to_string_before_split(self):
		df = generate_simple_pd_dataframe()

		# no label, only feature
		ds = tdml.Dataset(df)
		ds.transform()
		to_string = 'Dataset(feature=[10, 5], feature_mapping=2)'
		self.assertEqual(to_string, str(ds))
		
		# categorical label
		ds = tdml.Dataset(df, label='Evaluation')
		ds.transform()
		to_string = 'Dataset(label=[10], feature=[10, 4], label_mapping=3, feature_mapping=1)'
		self.assertEqual(to_string, str(ds))

		# boolean label
		ds = tdml.Dataset(df, label='Dark')
		ds.transform()
		to_string = 'Dataset(label=[10], feature=[10, 4], feature_mapping=2)'
		self.assertEqual(to_string, str(ds))

	def test_split_two_label(self):
		df = pd.read_csv('data/GOOG.csv')
		df = df.drop(columns=['Date', 'Adj Close'])
		
		ds = tdml.Dataset(df, label='Close')
		ds.transform()
		ds.train_test_split(seed=0)

		to_string = 'Dataset(label=[252], feature=[252, 4], train_x=[201, 4], train_y=[201], test_x=[51, 4], test_y=[51])'
		self.assertEqual(to_string, str(ds))
		self.assertEqual(ds.train_x.shape, (201, 4))
		self.assertEqual(ds.test_x.shape, (51, 4))
		self.assertEqual(ds.val_x, None)
		self.assertEqual(ds.train_y.shape, (201, ))
		self.assertEqual(ds.test_y.shape, (51, ))
		self.assertEqual(ds.val_y, None)

	def test_split_two_no_label(self):
		df = pd.read_csv('data/GOOG.csv')
		df = df.drop(columns=['Date', 'Adj Close'])
		
		ds = tdml.Dataset(df, feature=['Open', 'High'])
		ds.transform()
		ds.train_test_split(seed=0)

		to_string = 'Dataset(feature=[252, 2], train_x=[201, 2], test_x=[51, 2])'
		self.assertEqual(to_string, str(ds))
		self.assertEqual(ds.train_x.shape, (201, 2))
		self.assertEqual(ds.test_x.shape, (51, 2))
		self.assertEqual(ds.val_x, None)
		self.assertEqual(ds.train_y, None)
		self.assertEqual(ds.test_y, None)
		self.assertEqual(ds.val_y, None)

	def test_split_three_label(self):
		df = pd.read_csv('data/GOOG.csv')
		df = df.drop(columns=['Date', 'Adj Close'])
		
		ds = tdml.Dataset(df, label='Close')
		ds.transform()
		ds.train_val_test_split(seed=0)
		to_string = 'Dataset(label=[252], feature=[252, 4], train_x=[201, 4], train_y=[201], test_x=[26, 4], test_y=[26], val_x=[25, 4], val_y=[25])'
		self.assertEqual(to_string, str(ds))
		self.assertEqual(ds.train_x.shape, (201, 4))
		self.assertEqual(ds.test_x.shape, (26, 4))
		self.assertEqual(ds.val_x.shape, (25, 4))
		self.assertEqual(ds.train_y.shape, (201, ))
		self.assertEqual(ds.test_y.shape, (26, ))
		self.assertEqual(ds.val_y.shape, (25, ))

	def test_split_three_no_label(self):
		df = pd.read_csv('data/GOOG.csv')
		df = df.drop(columns=['Date', 'Adj Close'])
		
		ds = tdml.Dataset(df)
		ds.transform()
		ds.train_val_test_split()

		to_string = 'Dataset(feature=[252, 5], train_x=[201, 5], test_x=[26, 5], val_x=[25, 5])'
		self.assertEqual(to_string, str(ds))
		self.assertEqual(ds.train_x.shape, (201, 5))
		self.assertEqual(ds.test_x.shape, (26, 5))
		self.assertEqual(ds.val_x.shape, (25, 5))
		self.assertEqual(ds.train_y, None)
		self.assertEqual(ds.test_y, None)
		self.assertEqual(ds.val_y, None)

	def test_label_mapping(self):
		df = pd.read_csv('data/GOOG.csv')
		df = df.drop(columns=['Date', 'Adj Close'])
		ds = tdml.Dataset(df, label='Close')
		ds.transform()
		self.assertEqual(ds.label_mapping, None)

		df = generate_simple_pd_dataframe()
		ds = tdml.Dataset(df, label='Evaluation')
		ds.transform()
		self.assertEqual(len(ds.label_mapping), 3)
		self.assertEqual(len(ds.feature_mapping), 1)

		ds = tdml.Dataset(df)
		ds.transform()
		self.assertEqual(ds.label_mapping, None)
		self.assertEqual(len(ds.feature_mapping), 2)

	def test_prespecified_split_two(self):
		df = generate_simple_pd_dataframe()

		ds = tdml.Dataset(df, label="Evaluation")
		ds.transform()
		indices = np.arange(10)
		train_split = indices[:7]
		test_split = indices[7:]
		ds.train_test_split(train_split=train_split, test_split=test_split)
		self.assertEqual(ds.train_x.shape, (7, 4))
		self.assertEqual(ds.test_x.shape, (3, 4))
		self.assertEqual(ds.train_y.shape, (7, ))
		self.assertEqual(ds.test_y.shape, (3, ))

		ds = tdml.Dataset(df)
		ds.transform()
		ds.train_test_split(train_split=train_split, test_split=test_split)
		self.assertEqual(ds.train_x.shape, (7, 5))
		self.assertEqual(ds.test_x.shape, (3, 5))
		self.assertEqual(ds.train_y, None)
		self.assertEqual(ds.test_y, None)

	def test_prespecified_split_three(self):
		df = generate_simple_pd_dataframe()

		ds = tdml.Dataset(df, label="Evaluation")
		ds.transform()
		indices = np.arange(10)
		train_split = indices[:6]
		val_split = indices[6:8]
		test_split = indices[8:]
		ds.train_val_test_split(train_split=train_split, val_split=val_split, test_split=test_split)
		self.assertEqual(ds.train_x.shape, (6, 4))
		self.assertEqual(ds.val_x.shape, (2, 4))
		self.assertEqual(ds.test_x.shape, (2, 4))
		self.assertEqual(ds.train_y.shape, (6, ))
		self.assertEqual(ds.val_y.shape, (2, ))
		self.assertEqual(ds.test_y.shape, (2, ))

		ds = tdml.Dataset(df)
		ds.transform()
		ds.train_val_test_split(train_split=train_split, val_split=val_split, test_split=test_split)
		self.assertEqual(ds.train_x.shape, (6, 5))
		self.assertEqual(ds.val_x.shape, (2, 5))
		self.assertEqual(ds.test_x.shape, (2, 5))
		self.assertEqual(ds.train_y, None)
		self.assertEqual(ds.val_y, None)
		self.assertEqual(ds.test_y, None)

	def test_reshuffle(self):
		df = generate_simple_pd_dataframe()

		ds = tdml.Dataset(df, label="Evaluation")
		ds.transform()
		indices = np.arange(10)
		train_split = indices[:6]
		val_split = indices[6:8]
		test_split = indices[8:]
		ds.train_val_test_split(train_split=train_split, val_split=val_split, test_split=test_split)
		ds.reshuffle(seed=1)
		self.assertTrue(not np.array_equal(ds._reshuffle_indices, train_split))
		self.assertEqual(len(ds._reshuffle_indices), len(train_split))

	def test_text(self):
		# Default transformation
		df = pd.read_csv('data/simple_text.csv')
		ds = tdml.Dataset(df, label='Broken', text="Description")
		ds.transform()

		self.assertTrue('Description' in ds.feature_mapping)
		self.assertEqual(ds.idx_to_feature((4, 20)), 'Description')
		self.assertEqual(ds.feature_to_idx('Description'), (4, 20))

		# Customized transformation
		ds = tdml.Dataset(df, text="Description")
		ds.transform(text_transform=simple_embedding, val=-1, dim=8)

		self.assertTrue('Description' in ds.feature_mapping)
		self.assertEqual(ds.idx_to_feature((5, 13)), 'Description')
		self.assertEqual(ds.feature_to_idx('Description'), (5, 13))

if __name__ == '__main__':
	unittest.main()