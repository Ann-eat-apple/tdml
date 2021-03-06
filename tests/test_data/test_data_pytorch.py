import os
os.environ['TDML_FRAMEWORK'] = 'pytorch'
import tdml
import unittest
import torch

class TestDataPyTorch(unittest.TestCase):

	def test_data(self):
		array = torch.tensor([[1,2,3,4], [5,6,7,8]])
		data = tdml.Data(array)

		self.assertEqual(data.shape, (2, 4))
		self.assertEqual(data.num_sample, 2)
		self.assertEqual(str(data), "Data(data=[2, 4])")

		reshuffle_indices = torch.tensor([1, 0])
		reshuffle_array = torch.tensor([[5,6,7,8], [1,2,3,4]])
		data.reshuffle(reshuffle_indices)
		self.assertTrue(torch.all(torch.eq(data.data, reshuffle_array)))

	def test_feature(self):
		array = torch.tensor([[1,2,3,4], [5,6,7,8]])
		feature = tdml.Feature(array)

		self.assertEqual(feature.num_feature, 4)

	def test_label(self):
		array = torch.tensor([0,0,1,2,1,3,2,1,1,1,1,0])
		label = tdml.Label(array)

		self.assertEqual(label.num_label, 4)

if __name__ == '__main__':
	unittest.main()