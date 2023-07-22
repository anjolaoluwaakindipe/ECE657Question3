# import required packages
import os
from tensorflow import keras
import random
from nltk.corpus import stopwords
import numpy as np
from sklearn import metrics
import utils



class NLP_Training:
	def __init__(self, file_path:str):
		self._model = keras.models.load_model(file_path)
		pass

	def load_testing_set(self,pos_path:str, neg_path:str):
		""" 
		Loads the testing set from a file path 
		and set the and sets the input and output testing set
		"""
		test_x, test_y = utils.load_preprocess_samples_list(pos_path=pos_path,  neg_path= neg_path)

		# create a token to tokenize the sentences
		with open('tokenizer.json') as jsonfile:
			tokenizer_json = jsonfile.read()

		token = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

		# assigning train_x to the tokenize version of all the text
		test_x = token.texts_to_sequences(test_x)

		# Padding each text
		test_x = keras.preprocessing.sequence.pad_sequences(test_x, maxlen=1000, padding="post")

		# assign train_x and train_y (as numpy arrays)
		self._test_x, self._test_y = test_x, np.array(test_y) 

	def test(self):
		loss, accuracy = self._model.evaluate(self._test_x, self._test_y)
		print("Test Accuracy: ",accuracy)

if __name__ == "__main__": 
	# 1. Load your saved model
	nlp_training = NLP_Training("./models/NLP_model.model") 

	# 2. Load your testing data
	nlp_training.load_testing_set(pos_path="./data/aclImdb/test/pos", neg_path="./data/aclImdb/test/neg")

	# 3. Run prediction on the test data and print the test accuracy
	nlp_training.test()

	pass