from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_validate


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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


#########ROC#######
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, roc_curve

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
num_epochs=16
batch_size=32
def create_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(41, input_dim=41, activation='relu'))
    model.add(Dense(41, activation='relu'))
    model.add(Dense(5, activation = 'softmax'))


    #   base_model = Sequential()
    #   base_model.add(Flatten(input_shape=input_shape))  # this converts our 3D feature maps to 1D feature vectors
    #   base_model.add(Dense(256, activation='relu'))
    #   base_model.add(Dense(256, activation='relu'))
    #   base_model.add(Dense(128, activation='relu'))
    #   base_model.add(Dense(64, activation='relu'))
    #   base_model.add(Dense(3, activation='softmax'))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ------------------------------------------------------------- #
# ---------------------- Sampling Data ------------------------- #
# ------------------------------------------------------------- #
#Find best oversampling ratio 

ratios = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
normal_f = []
dos_f = []
probe_f=[]
u2r_f=[]
r2l_f=[]
class_names=['normal','dos','probe','u2r','r2l']
model=create_model()
### fit (train) the classifier on the training features and labels
model.fit(X_train, dummy_y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size, verbose=1)
y_pred=model.predict(X_test, batch_size=32)

hehe = classification_report(dummy_y_Test.argmax(axis=1),y_pred.argmax(axis=1), target_names=class_names, output_dict=True)
print(hehe)

normal_f.append(hehe['normal']['f1-score'])
dos_f.append(hehe['dos']['f1-score'])
probe_f.append(hehe['probe']['f1-score'])
u2r_f.append(hehe['u2r']['f1-score'])
r2l_f.append(hehe['r2l']['f1-score'])

for i in ratios:

    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy=i, random_state=1)
    #print(Counter(Y))
    X_ov, Y_ov = oversample.fit_resample(X_train, dummy_y_train)
    #print(Counter(Y_ov))

    under = RandomUnderSampler(sampling_strategy=1, random_state=1)
    X_un, Y_un = under.fit_resample(X_ov, Y_ov)
    #print(Counter(Y_un))


    model=create_model()
    model.fit(X_un, Y_un, validation_split=0.2, epochs=num_epochs, batch_size=batch_size, verbose=1)
    y_pred=model.predict(X_test, batch_size=32)
    scores = cross_validate(estimator=model, X=X_train, y=dummy_y_train, cv=10,return_train_score=True)

#     class_names=['normal','dos','probe','u2r','r2l']

    hehe = classification_report(dummy_y_Test.argmax(axis=1),y_pred.argmax(axis=1), target_names=class_names, output_dict=True)
    normal_f.append(hehe['normal']['f1-score'])
    dos_f.append(hehe['dos']['f1-score'])
    probe_f.append(hehe['probe']['f1-score'])
    u2r_f.append(hehe['u2r']['f1-score'])
    r2l_f.append(hehe['r2l']['f1-score'])
    
    
ratios = [0, 0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
fig, ax = plt.subplots()
ax.set_xlabel("Oversampling Ratio")
ax.set_ylabel("F1-score")
ax.set_title("F1-Score vs. oversampling ratio")

ax.plot(ratios, normal_f, marker='o', label="normal",
        drawstyle="steps-post")
ax.plot(ratios, dos_f, marker='o', label=" dos",
        drawstyle="steps-post")
ax.plot(ratios, probe_f, marker='o', label=" probe_f",
        drawstyle="steps-post")
ax.plot(ratios, u2r_f, marker='o', label=" u2r_f",
        drawstyle="steps-post")
ax.plot(ratios, r2l_f, marker='o', label=" r2l_f",
        drawstyle="steps-post")
ax.legend()
plt.savefig('sampling.png')
