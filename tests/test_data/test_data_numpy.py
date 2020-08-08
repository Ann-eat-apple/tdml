import os
import tdml
import unittest
import numpy as np

class TestDataNumPy(unittest.TestCase):

	def test_data(self):
		array = np.array([[1,2,3,4], [5,6,7,8]])
		data = tdml.Data(array)

		self.assertEqual(data.shape, (2, 4))
		self.assertEqual(data.num_sample, 2)
		self.assertEqual(str(data), "Data(data=[2, 4])")

		reshuffle_indices = np.array([1, 0])
		reshuffle_array = np.array([[5,6,7,8], [1,2,3,4]])
		data.reshuffle(reshuffle_indices)
		self.assertTrue(np.array_equal(data.data, reshuffle_array))

	def test_feature(self):
		array = np.array([[1,2,3,4], [5,6,7,8]])
		feature = tdml.Feature(array)

		self.assertEqual(feature.num_feature, 4)

	def test_label(self):
		array = np.array([0,0,1,2,1,3,2,1,1,1,1,0])
		label = tdml.Label(array)

		self.assertEqual(label.num_label, 4)

if __name__ == '__main__':
	unittest.main()