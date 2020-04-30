# ------------------------------------------------------------- #
# ------------------------- IMPORTING ------------------------- #
# ------------------------------------------------------------- #

import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras import layers
from keras.layers.core import Dense
from keras.optimizers import SGD

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor


from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from keras.layers import GlobalAveragePooling2D
from keras.models import Model

# to prevent CUDA error
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# ------------------------------------------------------------- #
# ------------------------- PARAMETERS ------------------------ #
# ------------------------------------------------------------- #

# configure model and training parameters
image_size = 299
# test_split = 0.2       # no explicitly defined test split; handled by k-fold validation
# validation_split = 0   # no validation split; each fold only has train/test
learning_rate = 0.05
optimizer = SGD(lr=learning_rate)
num_epochs = 75
training_batch_size = 48
testing_batch_size = training_batch_size

# ------------------------------------------------------------- #
# ----------------------- ARGUMENT PARSING -------------------- #
# ------------------------------------------------------------- #


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
ap.add_argument("-r", "--ROC", required=True,
                help="path to output ROC plot")
ap.add_argument("-z", "--ROCz", required=True,
                help="path to output zoomed ROC plot")
ap.add_argument("-pr", "--PR", required=True,
                help="path to output  PR plot")
ap.add_argument("-conf", "--CONF", required=True,
                help="path to output  Confusion Matrix")
args = vars(ap.parse_args())

# ------------------------------------------------------------- #
# -------------------------- DATA PREP ------------------------ #
# ------------------------------------------------------------- #

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
import time
timestamp = int(time.time()*1000.0)
random.seed(timestamp)
random.shuffle(imagePaths)

count = 0
# loop over the input images
for imagePath in imagePaths:
    count = count + 1
    print("Processing image: " + str(imagePath) + ", #" + str(count) + "\n")    
    image = cv2.imread(imagePath)  
    image = cv2.resize(image, (image_size, image_size))  # .flatten()
    data.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

print("Processed " + str(count) + " images!\n")

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=1)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# ------------------------------------------------------------- #
# ---------------------------- MODEL -------------------------- #
# ------------------------------------------------------------- #
def create_InceptionV3():
    # Getting the InceptionV3 model 
    inceptionV3 = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))
    model = Sequential()
    model.add(inceptionV3)
#     model.add(layers.Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers:
        layer.trainable = True

    from contextlib import redirect_stdout

    model.summary()

    # initialize our initial learning rate and # of epochs to train for
    INIT_LR = 0.01

    opt = SGD(lr=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    return model
# create inception model function


model = create_InceptionV3()

# train the neural network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_split=0.25,
              epochs=num_epochs, batch_size=training_batch_size)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=testing_batch_size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# ------------------------------------------------------------- #
# ----------------------- ROC CURVE --------------------------- #
# ------------------------------------------------------------- #

#########ROC#######
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, roc_curve
class_names=['donkey','fox','ghost']




# Plot linewidth.
lw = 2
n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), predictions.ravel())
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
plt.savefig(args["ROC"])

#plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig(args["ROCz"])
#plt.show()

# ------------------------------------------------------------- #
# -------------------- Freeze Model --------------------------- #
# ------------------------------------------------------------- #
# save model and weights into .h5 file (if --model output file specified in arguments)
if args["model"] is not None:
  model.save(args["model"])
  print("Model architecture and weights saved onto specified file!")


# ------------------------------------------------------------- #
# ----------------------- PR CURVE --------------------------- #
# ------------------------------------------------------------- #
if args["PR"] is not None:
    plt.figure(3)
    # precision recall curve
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(testY[:, i],
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
    plt.savefig(args["PR"])

# ------------------------------------------------------------- #
# ---------------- Training Validation CURVE ------------------ #
# ------------------------------------------------------------- #
# plot the training loss and accuracy (if --plot output file specified in arguments)
if args["plot"] is not None:
    plt.figure(4)
    N = np.arange(0, num_epochs)
    plt.style.use("ggplot")
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch \#")
    print("Plot of training/validation loss and accuracy saved!")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["plot"])

# ------------------------------------------------------------- #
# ----------------- Confusion Matrix--------------------------- #
# ------------------------------------------------------------- #

# Plot non-normalized confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

matrix = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)

plt.figure(5)

df_cm = pd.DataFrame(matrix,  index = [i for i in class_names],
                  columns = [i for i in class_names])
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.savefig(args["CONF"])

# ------------------------------------------------------------- #
# ----------------- Model Visualization------------------------ #
# ------------------------------------------------------------- #

from keras.utils import plot_model

plot_model(model, to_file='inceptionV3.png', show_shapes=True,)