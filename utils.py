# import required packages
from tensorflow import keras
import string
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import os
import random
nltk.download('punkt')
nltk.download('stopwords')

def word_preprocessor(text:str)-> str:
	# make text lower case
	text = text.lower()

	# Remove html tag
	text = re.sub(r'<br\s*><br\s*>', ' ', text)

	# remove all punctuation in text
	text = text.translate(str.maketrans("", "", string.punctuation))

	# remove stopping words that have no meaning
	stop_words = set(stopwords.words('english'))
	text_with_no_stop_words = ""
	for word in nltk.word_tokenize(text):
		if word.lower() not in stop_words:
			text_with_no_stop_words += " " + word
	text = text_with_no_stop_words.strip()
	# pass
	# text = " ".join([word for word in nltk.word_tokenize(text) if word.lower() not in stop_words])

	# stem words in text e.g running -> run
	stemmer = nltk.stem.PorterStemmer()
	text = " ".join([stemmer.stem(word) for word in nltk.word_tokenize(text)])
	return text


def load_preprocess_samples_list(pos_path: str, neg_path:str):
	pos_x = []
	pos_y = []
	neg_x = []
	neg_y = []

	# Reads positive reviews and assigns them 1
	for file_name in os.listdir(pos_path):
		path = os.path.join(pos_path, file_name)
		text = ""
		with open(path, 'r', encoding='utf-8') as f:
			text = f.read()
		# print(text)
		text = word_preprocessor(text)
		# print(text)
		pos_x.append(text)
		pos_y.append(1)
		break
		
	# Read negative reviews and assigns them 0
	for file_name in os.listdir(neg_path):
		path = os.path.join(neg_path, file_name)
		# print(path)
		text = ""
		with open(path, 'r', encoding='utf-8') as f:
			text = f.read()
		# print(text)
		text = word_preprocessor(text)
		# print(text)
		neg_x.append(text)
		neg_y.append(0)
		break
	
	# joins positve and negative texts and labels
	sample_x = [*pos_x, *neg_x]
	sample_y = [*pos_y, *neg_y]

	#  shuffles the postive and negative reviews and thier labels
	combined_x_y = list(zip(sample_x, sample_y))
	random.shuffle(combined_x_y)
	sample_x, sample_y = zip(*combined_x_y)
	sample_x = list(sample_x)
	sample_y = list(sample_y)
	return sample_x, sample_y