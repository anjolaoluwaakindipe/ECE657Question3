# import required packages
import os
from tensorflow import keras
import random
from nltk.corpus import stopwords
import numpy as np
import utils


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

class NLP_Network:
	def __init__(self) -> None:
		self._train_x = []
		self._train_y = []
		self._vocabulary_size = 0
		pass

	def load_training_data(self, pos_path:str, neg_path:str):
		train_x , train_y = utils.load_preprocess_samples_list(pos_path=pos_path, neg_path=neg_path)
		print(type(train_x))
		# create a token to tokenize the sentences
		token = keras.preprocessing.text.Tokenizer(num_words=6000)
		token.fit_on_texts(train_x)

		vocabulary_size = len(token.word_index) + 1

		tokenizer_json = token.to_json()

		with open('tokenizer.json', 'w') as json_file:
			json_file.write(tokenizer_json)

		# assign train_x to the tokenize version of all the text
		train_x = token.texts_to_sequences(train_x)

		print(type(train_x))

		# Padding each text
		train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=1000, padding="post")

		self._train_x, self._train_y, self._vocabulary_size=  train_x, np.array(train_y), vocabulary_size

	
	def train(self):
		# Create NLP Network
		self._model = keras.Sequential()
		self._model.add(keras.layers.Embedding(self._vocabulary_size, 100, input_length=1000, mask_zero=True))
		# self._model.add(keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
		# self._model.add(keras.layers.SpatialDropout1D(rate=0.2))
		# self._model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.5))
		# self._model.add(keras.layers.GRU(128,
        #                 activation='tanh', return_sequences=False))
		self._model.add(keras.layers.SimpleRNN(128, activation='tanh',return_sequences= False, recurrent_dropout=0.2))
		# self._model.add(keras.layers.Dropout(0.2))
		# self._model.add(keras.layers.GlobalAveragePooling1D())
		self._model.add(keras.layers.Dense(64, activation='relu'))
		self._model.add(keras.layers.Dropout(0.2))
		# self._model.add(keras.layers.Dense(64, activation='relu'))
		self._model.add(keras.layers.Dense(1, activation="sigmoid"))

		# Compile model
		self._model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
		
		# Create EarlyStopping
		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

		# Start training the model
		history = self._model.fit(self._train_x, self._train_y, epochs=10, batch_size=32, validation_split=0.3, callbacks=[early_stop])
		# train_loss = history.history['loss']

		# Get training Accuracy and print 
		train_accuracy = history.history['accuracy'][-1]
		# print('Training Loss:', train_loss)
		print('Final Training Accuracy:', train_accuracy)

		pass
	
	def save_model(self, directory: str):
		self._model.save(directory)
		pass


if __name__ == "__main__": 
	# 1. load your training data
	network = NLP_Network()
	network.load_training_data(pos_path="./data/aclImdb/train/pos", neg_path="./data/aclImdb/train/neg")

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy
	network.train()
	

	# 3. Save your model
	network.save_model("./models/NLP_model.model")	
	pass