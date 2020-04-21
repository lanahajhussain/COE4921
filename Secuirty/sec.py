import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_intrusion = pd.read_csv('Train_data.csv')

df_intrusion.protocol_type= [1 if each == "icmp"
                               else 2 if each == "udp" 
                               else 0 
                               for each in df_intrusion.protocol_type]

df_intrusion.xAttack= [1 if each == "dos"
                               else 2 if each == "probe" 
                               else 3 if each == "u2r"
                               else 4 if each == "r2l"
                               else 0 
                               for each in df_intrusion.xAttack]

df_intrusion_t = pd.read_csv('test_data.csv')


df_intrusion_t=df_intrusion_t.drop(["Unnamed: 0"], axis=1)

df_intrusion_t.protocol_type= [1 if each == "icmp"
                               else 2 if each == "udp" 
                               else 0 
                               for each in df_intrusion_t.protocol_type]
df_intrusion_t.xAttack= [1 if each == "dos"
                               else 2 if each == "probe" 
                               else 3 if each == "u2r"
                               else 4 if each == "r2l"
                               else 0 
                               for each in df_intrusion_t.xAttack]


# Create two pandas arrays -- one for X and one for Y to get ready 
# for neural network

# convert the array into a numpy array
arr_train = df_intrusion.to_numpy()
arr_test = df_intrusion_t.to_numpy()

# separate X and Y
X_train = arr_train[:,0:41]
Y_train = arr_train[:,41]


# separate X and Y
X_test = arr_test[:,0:41]
Y_test = arr_test[:,41]


# Prepare Y values for one-hot encoding

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(X_train[:])
# encoded_X_train= encoder.transform(X_train)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_X_train = np_utils.to_categorical(encoded_X_train)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train= encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y_test= encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_Test = np_utils.to_categorical(encoded_Y_test)

from keras.models import Sequential
from keras import layers
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
import keras
from keras.models import Sequential
from keras.layers import Dense

# define the keras model
model = Sequential()
model.add(Dense(41, input_dim=41, activation='relu'))
model.add(Dense(41, activation='relu'))
model.add(Dense(5, activation = 'softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, dummy_y_train, validation_split=0.25, epochs=50, batch_size=32, verbose=1)

from keras.utils import plot_model
import pydotplus
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot
plot_model(model, to_file='model.png', show_shapes=True,)


predictions=model.predict(X_test, batch_size=32)
print(predictions)

from sklearn.metrics import classification_report
# evaluate the network
print("[INFO] evaluating network...")
print(classification_report(dummy_y_Test.argmax(axis=1),
                            predictions.argmax(axis=1)))


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(dummy_y_Test.argmax(axis=1), predictions.argmax(axis=1))
print(confusion_matrix)