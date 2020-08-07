import pandas as pd

def generate_simple_pd_dataframe():
	cars = {
			'Brand': ['Audi', 'Toyota', 'Ford', 'Audi', 'Mercedes', 'Mercedes', 'Audi', 'Audi', 'Toyota', 'Audi'],
			'Price': [30000, 25000, 27000, 35000, 55000, 60000, 31000, 31000, 23000, 32000],
			'Used Year': [2, 5, 6, 3, 1, 3, 2, 2, 4, 4],
			'Evaluation': ['Good', 'OK', 'Bad', 'Good', 'Good', 'Good', 'OK', 'Good', 'OK', 'Good'],
			'Dark': [True, False, False, True, True, False, False, False, False, False]
		   }
	df = pd.DataFrame(cars, columns = ['Brand', 'Price', 'Used Year', 'Evaluation', 'Dark'])
	return df

def simple_embedding(sentences, val, dim):
	words = set()
	processed_sentences = []
	embeds = []
	for sentence in sentences:
		embeds.append([val] * dim)
	return embeds, {}