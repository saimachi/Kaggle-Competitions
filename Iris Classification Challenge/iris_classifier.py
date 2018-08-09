import numpy 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

numpy.random.seed(7) #ensures that the stochastic process of training a NN can be reproduced
train_file = pd.read_csv('iris.csv')
model = Sequential()
#defining feature columns
feature_cols = [
	'sepal length',
	'sepal width',
	'petal length',
	'petal width']
#converting labels from string to integer
def one_hot(flower_type):
	if flower_type == 'Iris-setosa':
		return 0
	elif flower_type == 'Iris-versicolor':
		return 1
	else:
		return 2

def add_one_hot_encoding():
	train_file['encoded'] = [one_hot(flower) for flower in train_file.species]
add_one_hot_encoding()
#one-hot encoding allows for multi-class predictions (every output of the NN is the result of the softmax function being called)
X = train_file[feature_cols]
y = train_file['encoded'] #initial labels
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y) #creates a one-hot vector
#model architecture: 4 inputs -> 8 hidden nodes (1 layer) -> 3 outputs
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #we have built our model

batch_size = 5
model.fit(X, dummy_y, epochs=150, batch_size=batch_size)

#Our model has been trained (98% accurate)! To make predictions, call model.predict(x, batch_size=None, verbose=0, steps=None), where
#x is a numpy array that contains the features of the plants to be predicted. if batch_size equals none, it defaults to
#32. Finally, steps represents the number of batches of samples needed to be completed before declaring the
#training round complete. model.predict(args) will return a numpy array of predictions.

