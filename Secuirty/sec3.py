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
# ------------------------- Data Cleaning--------------------- #
# ------------------------------------------------------------- #

df_intrusion = pd.read_csv('Train_data.csv')
df_intrusion_t = pd.read_csv('test_data.csv')

df_intrusion=df_intrusion.drop(["dst_host_srv_rerror_rate"], axis=1)
df_intrusion=df_intrusion.drop(["dst_host_rerror_rate"], axis=1)
df_intrusion=df_intrusion.drop(["srv_diff_host_rate"], axis=1)
df_intrusion=df_intrusion.drop(["srv_rerror_rate"], axis=1)
df_intrusion=df_intrusion.drop(["rerror_rate"], axis=1)

df_intrusion=df_intrusion.drop(["is_guest_login"], axis=1)
df_intrusion=df_intrusion.drop(["is_host_login"], axis=1)
df_intrusion=df_intrusion.drop(["num_outbound_cmds"], axis=1)
df_intrusion=df_intrusion.drop(["num_access_files"], axis=1)
df_intrusion=df_intrusion.drop(["num_shells"], axis=1)
df_intrusion=df_intrusion.drop(["num_file_creations"], axis=1)
df_intrusion=df_intrusion.drop(["num_root"], axis=1)
df_intrusion=df_intrusion.drop(["su_attempted"], axis=1)
df_intrusion=df_intrusion.drop(["root_shell"], axis=1)

df_intrusion=df_intrusion.drop(["num_compromised"], axis=1)
df_intrusion=df_intrusion.drop(["num_failed_logins"], axis=1)
df_intrusion=df_intrusion.drop(["hot"], axis=1)
df_intrusion=df_intrusion.drop(["land"], axis=1)
df_intrusion=df_intrusion.drop(["wrong_fragment"], axis=1)
df_intrusion=df_intrusion.drop(["duration"], axis=1)
df_intrusion=df_intrusion.drop(["urgent"], axis=1)




df_intrusion_t=df_intrusion_t.drop(["dst_host_srv_rerror_rate"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["dst_host_rerror_rate"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["srv_diff_host_rate"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["srv_rerror_rate"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["rerror_rate"], axis=1)

df_intrusion_t=df_intrusion_t.drop(["is_guest_login"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["is_host_login"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["num_outbound_cmds"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["num_access_files"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["num_shells"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["num_file_creations"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["num_root"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["su_attempted"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["root_shell"], axis=1)

df_intrusion_t=df_intrusion_t.drop(["num_compromised"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["num_failed_logins"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["hot"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["land"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["wrong_fragment"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["duration"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["Unnamed: 0"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["urgent"], axis=1)


df_intrusion=df_intrusion.drop(["src_bytes"], axis=1)
df_intrusion=df_intrusion.drop(["dst_bytes"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["src_bytes"], axis=1)
df_intrusion_t=df_intrusion_t.drop(["dst_bytes"], axis=1)



df_intrusion_t=df_intrusion_t[df_intrusion_t.xAttack != 'u2r']
df_intrusion=df_intrusion[df_intrusion.xAttack != 'u2r']


print(df_intrusion.shape)
print(df_intrusion_t.shape)



# ------------------------------------------------------------- #
# ------------------- Data Pre-processing---------------------- #
# ------------------------------------------------------------- #


df_intrusion.protocol_type= [1 if each == "icmp"
                               else 2 if each == "udp" 
                               else 0 
                               for each in df_intrusion.protocol_type]

df_intrusion_t.protocol_type= [1 if each == "icmp"
                               else 2 if each == "udp" 
                               else 0 
                               for each in df_intrusion_t.protocol_type]

df_intrusion.xAttack= [1 if each == "dos"
                               else 2 if each == "probe" 
                               else 3 if each == "r2l"
                               else 0 
                               for each in df_intrusion.xAttack]

df_intrusion_t.xAttack= [1 if each == "dos"
                               else 2 if each == "probe" 
                               else 3 if each == "r2l"
                               else 0 
                               for each in df_intrusion_t.xAttack]



# Create two pandas arrays -- one for X and one for Y to get ready 
# for neural network

# convert the array into a numpy array
arr_train = df_intrusion.to_numpy()
arr_test = df_intrusion_t.to_numpy()

# separate X and Y
X_train = arr_train[:,0:18]
Y_train = arr_train[:,18]


# separate X and Y
X_test = arr_test[:,0:18]
Y_test = arr_test[:,18]

# ------------------------------------------------------------- #
# ----------------- Feature Selection ------------------------- #
# ------------------------------------------------------------- #


# Dimensionality Reduction 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler


 


scaler=StandardScaler()#instantiate
scaler.fit(X_train) # compute the mean and standard which will be used in the next command
X_scaled_train=scaler.transform(X_train)# fit and transform can be applied together and I leave that for simple exercise
scaler.fit(X_test) # compute the mean and standard which will be used in the next command
X_scaled_test=scaler.transform(X_test)# fit and transform can be applied together and I leave that for simple exercise

# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1
print ("after scaling minimum", X_scaled_train.min(axis=0) )
print ("after scaling minimum", X_scaled_test.min(axis=0) )

Ratio=[]
Ratio1=[]
for i in range(1,18):
    pcatest = PCA(n_components=i) 
    pcatest.fit(X_scaled_train)
    PCA(copy=True, iterated_power='auto', n_components=i, random_state=None, 
    svd_solver='auto', tol=0.0, whiten=False)
    Ratio.append(round(pcatest.explained_variance_ratio_.sum(),i)*100)
    Ratio1.append(pcatest.explained_variance_ratio_[i-1]*100)
#     pca.explained_variance_ratio_
    print('\n n_components= ',i,pcatest.explained_variance_ratio_) 
#     print(pca.singular_values_) 
print()
print(Ratio1)

plt.figure(50)
components= np.linspace(1,17, num=17)
plt.figure(figsize=(30,10))
ax = sns.barplot(x=components, y=Ratio)
ax.set(title="Cumulative sum of variance",xlabel="Number of princple componenets",ylabel="Explained Variance Ratio")
plt.savefig('pca1.png')
plt.figure(55)

plt.figure(figsize=(20,5))
ax2 = sns.barplot(x=components, y=Ratio1)
ax2.set(title='Amount of variance explained by each PC',xlabel="Number of princple componenets",ylabel="Explained Variance Ratio")
plt.savefig('pca2.png')














pca2=PCA(0.90)
pca2.fit(X_scaled_train)
pca2.fit(X_scaled_test)

print(pca2.n_components_)
print(pca2.explained_variance_ratio_) 
print(pca2.singular_values_)
 
X_train_pca=pca2.transform(X_scaled_train) 
X_test_pca=pca2.transform(X_scaled_test) 


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
sm = SMOTE(sampling_strategy={3:20000,2:20000}, random_state=1)
X_ov,Y_ov=sm.fit_resample(X_train_pca, Y_train)
print(Counter(Y_ov))

under = RandomUnderSampler(sampling_strategy={0:20000,1:20000}, random_state=1)
X_new, Y_new = under.fit_resample(X_ov, Y_ov)
print(Counter(Y_new))

# oversample = RandomOverSampler(sampling_strategy=0.1, random_state=1)
# X_new, Y_new = oversample.fit_resample(X_ov, Y_ov)

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

print(dummy_y_train)
print(dummy_y_Test)
# ------------------------------------------------------------- #
# ---------------------- Create Model ------------------------- #
# ------------------------------------------------------------- #
num_epochs=20
batch_size=16
def create_model():
# define the keras model
    model = Sequential()
    model.add(Dense(9, input_dim=9, activation='sigmoid'))
    # model.add(layers.Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(4, activation = 'softmax'))


    # model = Sequential()
    # model.add(Dense(18, input_dim=18, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(4, activation = 'softmax'))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model=create_model()
history = model.fit(X_new,dummy_y_train, validation_split=0.20, epochs=num_epochs, batch_size=batch_size, verbose=1)

# ------------------------------------------------------------- #
# ----------------- Model Visualization------------------------ #
# ------------------------------------------------------------- #

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True,)



# ------------------------------------------------------------- #
# ----------------- Performance Metrics------------------------ #
# ------------------------------------------------------------- #


predictions=model.predict(X_test_pca, batch_size=batch_size)
print(dummy_y_Test.argmax(axis=1))
print(predictions.argmax(axis=1))

from sklearn.metrics import classification_report
# evaluate the network
print("[INFO] evaluating network...")
print(classification_report(dummy_y_Test.argmax(axis=1),
                            predictions.argmax(axis=1)))



# ------------------------------------------------------------- #
# ----------------- Confusion Matrix--------------------------- #
# ------------------------------------------------------------- #
class_names=['normal','dos','probe','r2l']
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
    precision[i], recall[i], _ = precision_recall_curve(dummy_y_Test[:, i],
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
plt.plot(N, history.history["accuracy"], label="train_accuracy")
plt.plot(N, history.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch \#")
print("Plot of training/validation loss and accuracy saved!")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('acc.png')
