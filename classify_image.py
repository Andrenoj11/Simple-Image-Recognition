#!/usr/bin/env python
# coding: utf-8

# In[7]:


# In[4]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os


# In[5]:


def extract_color_stats(image):
    #split the input image into its respective RGB color channels
    #and then create a feature vector with 6 values: the mean and
    #standard deviation for eac hof the 3 channels, respectively
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
               np.std(G), np.std(B)]
    #return our set of features
    return features
    


# In[6]:


models = {'knn': KNeighborsClassifier(n_neighbors = 1),
         'naive_bayes': GaussianNB(),
         'logit': LogisticRegression(solver='lbfgs', multi_class='auto'),
         'svm': SVC(kernel='linear'),
         'decision_tree': DecisionTreeClassifier(),
         'random_forest': RandomForestClassifier(n_estimators=100),
         'mlp': MLPClassifier()}


# In[ ]:


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', type=str, default='3scenes',
               help='path to directory containing the "3 scenes" dataset') #untuk mengeset folder
ap.add_argument('-m', '--model', type=str, default='knn',
               help='type of python machine learning model to use')
ap.add_argument('-a', '--test_a', type=str, default='test_a', 
               help='path to directory containing the test_a dataset')
args = vars(ap.parse_args())


# In[ ]:


print('[INFO] extracting image features...')
imagePaths = paths.list_images(args['dataset'])
data = []
labels = []

#loop over our input images
for imagePath in imagePaths:
    #load the input image from disk, compute color channel
    #statistics, and then update our data list
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    
    #extract the class label from the file path and update the label list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    
#encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#perform a training and testing split, using 75% of the data for
#training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

#train the model
print("[INFO] using '{}' model".format(args['model']))
model = models[args['model']]
model.fit(trainX, trainY)

#make predictions on our data and show a classification report
print('[INFO] evaluating...')
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))


# cucubo se lu In[32]:
target = args['test_a']
dat = []

#loop over our input images
if target != '':
    #load the input image from disk, compute color channel
    #statistics, and then update our data list
    image = Image.open(target)
    
    featuress = extract_color_stats(image)
    dat.append(featuress)
    pred = model.predict(dat)
    #extract the class label from the file path and update the label list
   
    
print('nama :', str(target))
print('pred :', str(le.classes_[pred]))







