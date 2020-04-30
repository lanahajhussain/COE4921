# ------------------------------------------------------------- #
# ------------------------- IMPORTING ------------------------- #
# ------------------------------------------------------------- #
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
import keras
from keras.models import Sequential
from keras.layers import Dense



import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#########ROC#######
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, roc_curve
#Grid Search 
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier





# ------------------------------------------------------------- #
# ------------------- Data Pre-processing---------------------- #
# ------------------------------------------------------------- #

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

# ------------------------------------------------------------- #
# ---------------------- Encoding Data ------------------------- #
# ------------------------------------------------------------- #
# Prepare Y values for one-hot encoding

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

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


# ------------------------------------------------------------- #
# ---------------------- Create Model ------------------------- #
# ------------------------------------------------------------- #
num_epochs=50
batch_size=32

def create_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(41, input_dim=41, activation='relu'))
    model.add(Dense(41, activation='relu'))
    model.add(Dense(5, activation = 'softmax'))
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# create model for grid search
model = KerasClassifier(build_fn=create_model)
########################################################
# Use scikit-learn to grid search
# activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] # softmax, softplus, softsign
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# weight_constraint = [1, 2, 3, 4, 5]
# neurons = [1, 5, 10, 15, 20, 25, 30]
# init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
##############################################################
# grid search epochs, batch size, and learning rate
epochs = [16, 50, 100]  #32,25,75, 128, 150, 175, 200, 225, 250]  # add 50, 100, 150 etc
batch_size = [16, 32]  #, 128, 150, 175]  # add 5, 10, 20, 40, 60, 80, 100 etc
# learn_rate = [0.01, 0.001, 0.0005]
param_grid = dict(epochs=epochs, batch_size=)
# , learn_rate=learn_rate)
##############################################################
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, dummy_y_train)
##############################################################
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
##############################################################