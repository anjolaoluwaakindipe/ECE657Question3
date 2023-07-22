# import required packages
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
class RNN_Training:
	def __init__(self, file_path:str):
		self._model = keras.models.load_model(file_path)
		pass
	
	def load_testing_set(self, file_path:str):
		""" 
		Loads the testing set from a file path 
		and set the and sets the input and output testing set
		"""
		testing_sample = pd.read_csv(file_path)
		self._X_test = testing_sample.values[:,1:13].reshape((testing_sample.shape[0],3,4))
		self._Y_test = testing_sample.values[:,13]
	
	def test(self):
		""" 
		Uses the model to predict and creates 
		a prediction array from the input set
		then compares it to the output set.It
		prints out the final loss and plots the
		predicted values against the actual value
		"""
		Y_pred = self._model.predict(self._X_test)
		loss = mean_squared_error(self._Y_test, Y_pred)
		print('Loss:', loss)
		# Plot the true and predicted values
		plt.plot(self._Y_test, label='Actual Values', linewidth=1.5)
		plt.plot(Y_pred,label='Predicted Values', linewidth=0.5)
		plt.xlabel('Data Point')
		plt.ylabel('Value')
		plt.legend()
		plt.show()
		pass


if __name__ == "__main__":
	# 1. Load your saved model
	rnn_training = RNN_Training("./models/RNN_model.model")

	# 2. Load your testing data
	rnn_training.load_testing_set("./data/test_data_RNN.csv")

	# 3. Run prediction on the test data and output required plot and loss
	rnn_training.test()