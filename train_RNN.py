# import required packages
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
def create_dataset_csv():
	# load the data set
	prices_with_dates = pd.read_csv("./q2_dataset.csv")

	# remove Close/Last and Date column
	prices = prices_with_dates.drop([" Close/Last", "Date"], axis=1)

	# normalize price values between 0 and 1 and return as np array
	scaler = MinMaxScaler()
	prices_np =  scaler.fit_transform(prices)

	# get the characters needed to build the train.csv and test.csv
	timestep = 3
	features = prices_np.shape[1]
	num_prices = len(prices_np)

	# create the input and output from the prices
	X = []
	Y = []
	for i in range(num_prices - timestep):
		X.append(prices_np[i:i+timestep][:])
		Y.append(prices_np[i+timestep][1])
	
	# Convert Input and output into numpy arrays
	X_np = np.array(X)
	Y_np = np.array(Y)

	sample_size = X_np.shape[0]
	# shuffle samples
	indices = np.arange(sample_size)

	np.random.shuffle(indices)
	X_np = X_np[indices]
	Y_np = Y_np[indices]
	
	# split samples to test and train set
	training_samples = (sample_size * 70) // 100
	X_train_pre = X_np[:training_samples]
	X_test_pre = X_np[training_samples:]
	Y_train_pre = Y_np[:training_samples]
	Y_test_pre = Y_np[training_samples:]

	# Flatten input for both training and testing set
	X_train_flat = X_train_pre.reshape((X_train_pre.shape[0], -1))
	X_test_flat = X_test_pre.reshape((X_test_pre.shape[0], -1))

	# Volume Open High Low
	column_names = ["Volume1", "Open1", "High1", "Low1","Volume2", "Open2", "High2", "Low2","Volume3", "Open3", "High3", "Low3"]
	X_train_df = pd.DataFrame(X_train_flat, columns=column_names)
	X_test_df = pd.DataFrame(X_test_flat, columns=column_names)
	Y_train_df = pd.DataFrame(Y_train_pre, columns=["NextDayOpening"])
	Y_test_df = pd.DataFrame(Y_test_pre, columns=["NextDayOpening"])
	train = pd.concat([X_train_df, Y_train_df], axis=1)
	test = pd.concat([X_test_df, Y_test_df], axis=1)

	train.to_csv("./data/train_data_RNN.csv")
	test.to_csv("./data/test_data_RNN.csv")

	pass


class RNN_Network:
	def __init__(self, file_path:str) -> None:
		training_sample = pd.read_csv(file_path)
		self._X_train = training_sample.values[:,1:13].reshape((training_sample.shape[0],3,4))
		self._Y_train = training_sample.values[:,13]
		self._model = RNN_network = keras.models.Sequential()
		self._model.add(keras.layers.LSTM(units=64, return_sequences=True,input_shape=(3, 4)))
		self._model.add(keras.layers.LSTM(units=64, return_sequences=True))
		self._model.add(keras.layers.LSTM(units=64))
		self._model.add(keras.layers.Dense(1))
		self._model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		pass
	
	def train(self, batch_size: int, epochs:int, verbose:int = 1):
		history = self._model.fit(self._X_train, self._Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
		self._model.summary()
		print(f"Final Loss: {history.history['loss'][-1]}")
		pass
	def save_model(self,file_path:str):
		self._model.save(file_path)
		pass



if __name__ == "__main__": 
	# Create train and testing dataset
	create_dataset_csv()
	# 1. load your training data
	network = RNN_Network("./data/train_data_RNN.csv")
	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss
	network.train(10, 100, 1)

	# 3. Save your model
	network.save_model("./models/RNN_model.model")	