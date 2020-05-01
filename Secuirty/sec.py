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
# ----------------- Balancing Dataset------------------------ #
# ------------------------------------------------------------- #
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
 # define oversampling strategy
sm = SMOTE(sampling_strategy={3:10000,4:10000}, random_state=7)
X_new,Y_new=sm.fit_resample(X_train, Y_train)
# oversample = RandomOverSampler(sampling_strategy=0.1, random_state=1)
# X_new, Y_new = oversample.fit_resample(X_train, dummy_y_train)
print(Counter(Y_new))
# under = RandomUnderSampler(sampling_strategy={'0:1000,1:1000,2:100'}, random_state=1)
# X_un, Y_un = under.fit_resample(X_ov, Y_ov)

# ------------------------------------------------------------- #
# ---------------------- Encoding Data ------------------------- #
# ------------------------------------------------------------- #
# Prepare Y values for one-hot encoding

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_new)
encoded_Y_train= encoder.transform(Y_new)
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
model=create_model()
history = model.fit( X_new,dummy_y_train, validation_split=0.25, epochs=num_epochs, batch_size=batch_size, verbose=1)

# ------------------------------------------------------------- #
# ----------------- Model Visualization------------------------ #
# ------------------------------------------------------------- #

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True,)



# ------------------------------------------------------------- #
# ----------------- Performance Metrics------------------------ #
# ------------------------------------------------------------- #


predictions=model.predict(X_test, batch_size=batch_size)
print(predictions)

from sklearn.metrics import classification_report
# evaluate the network
print("[INFO] evaluating network...")
print(classification_report(dummy_y_Test.argmax(axis=1),
                            predictions.argmax(axis=1)))



# ------------------------------------------------------------- #
# ----------------- Confusion Matrix--------------------------- #
# ------------------------------------------------------------- #
class_names=['normal','dos','probe','u2r','r2l']
# Plot non-normalized confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

matrix = confusion_matrix(dummy_y_Test.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)

plt.figure(5)

df_cm = pd.DataFrame(matrix,  index = [i for i in class_names],
                  columns = [i for i in class_names])
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.savefig('conf.png')

# ------------------------------------------------------------- #
# ----------------------- ROC CURVE --------------------------- #
# ------------------------------------------------------------- #

# Plot linewidth.
lw = 2
n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(dummy_y_Test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(dummy_y_Test.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('roc.png')



# ------------------------------------------------------------- #
# ----------------------- PR CURVE --------------------------- #
# ------------------------------------------------------------- #
plt.figure(3)
# precision recall curve
precision = dict()
recall = dict()
pr_auc = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(dummy_y_train[:, i],
                                                            predictions[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='PR curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], pr_auc[i]))
    
    # plt.plot([1, 0], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.savefig('PR.png')

# ------------------------------------------------------------- #
# ---------------- Training Validation CURVE ------------------ #
# ------------------------------------------------------------- #
# plot the training loss and accuracy (if --plot output file specified in arguments)
plt.figure(4)
N = np.arange(0, num_epochs)
plt.style.use("ggplot")
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["acc"], label="train_acc")
plt.plot(N, history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch \#")
print("Plot of training/validation loss and accuracy saved!")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('acc.png')
