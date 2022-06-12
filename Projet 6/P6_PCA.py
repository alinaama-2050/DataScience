
# ************************************************************************************************
# Description : PCA
# Author : ANA
# We used an example to show practically how PCA can help to visualize a
# High dimension dataset, reduces computation time, and avoid overfitting
#
# https://machinelearningknowledge.ai/complete-tutorial-for-pca-in-python-sklearn-with-example/
#
# *************************************************************************************************

import sklearn
import numpy as np
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from bioinfokit.analys import get_data
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time
# logging mngt
import logging
import datetime

def format_percentage(value):
    '''
    Format a percentage with 1 digit after comma
    '''
    return "{0:.1f}%".format(value * 100)

def writeHtml(df, dir, filename):
    '''
        Write Html page
    '''
    html = df.to_html()
    df.to_html(classes='table table-striped')
    # write html to file
    text_file = open(dir + filename + ".html", "w")
    text_file.write(html)
    text_file.close()

now = datetime.datetime.now()
logging.basicConfig(filename='log\Projetct6_PCA.log', level=logging.INFO)
logging.info('Started : ' + now.strftime("%Y-%m-%d %H:%M:%S"))

# Read dataset
df= pd.read_csv(r"data/pd_speech_features.csv")
print(df.head())
print(df.shape)


# Other Info EDA :
files_description = pd.DataFrame(
    columns=["Nb lignes", "Nb colonnes", "Taux remplissage moyen", "Doublons", "Description"], index=["pd_speech_features.csv"])

# Filling the total number rows in each file
files_description["Nb lignes"] = [len(df.index)]

# Filling the number of columns in each file
files_description["Nb colonnes"] = [len(df.columns)]

# Filling the fill-percentile of each file
files_description["Taux remplissage moyen"] = [format_percentage(df.notna().mean().mean())]

# Filling the number of duplicate keys for each file
files_description["Doublons"] = [df.duplicated(subset=["id"]).sum()]

# Finally adding a short description for each file
files_description["Description"] = ["pd_speech_features.csv"]

print(files_description)
writeHtml(files_description, 'data/', 'files_description_pd_speech_features.html')

df['class'].value_counts()
array = df.values
X = array[:,0:755]
Y = array[:,754]

# Rescale Data
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[:5,:])



# PCA
pca = PCA(n_components = 3)
fit = pca.fit(rescaledX)
print("Explained Variance: %s" % fit.explained_variance_ratio_)
# Dump components relations with features:
print(fit.components_)
data_scaled = pd.DataFrame(rescaledX,columns = df.columns)
# Dump components relations with features:
print(pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2','PC-3']).head())
logging.info(pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2','PC-3']).head())

# Display PCA
# positive and negative values in component loadings reflects the positive and negative
# correlation of the variables with the PCs.

principalComponents = pca.fit_transform(rescaledX)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
print(finalDf.head())


# get scree plot (for scree or elbow test)
# Scree plot will be saved in the same directory with name screeplot.png
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot PCA')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# PCA
#pca2 = PCA(n_components = 20)
#fit2 = pca2.fit(rescaledX)
#print("Explained Variance: %s" % fit.explained_variance_ratio_)
# Dump components relations with features:
#PC_values2 = np.arange(pca2.n_components_) + 1
#plt.plot(PC_values2, pca2.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
#plt.title('Scree Plot PCA 2')
#plt.xlabel('Principal Component')
#plt.ylabel('Variance Explained')
#plt.show()

# Plot PC1 / PC2
plt.figure(figsize=(7,7))
plt.scatter(finalDf['principal component 1'],finalDf['principal component 2'], c=finalDf['class'],cmap='prism', s =5)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

# Plot PC1 / PC3
plt.figure(figsize=(7,7))
plt.scatter(finalDf['principal component 1'],finalDf['principal component 3'], c=finalDf['class'],cmap='prism', s =5)
plt.xlabel('pc1')
plt.ylabel('pc3')
plt.show()

# Plot PC2 / PC3
plt.figure(figsize=(7,7))
plt.scatter(finalDf['principal component 2'],finalDf['principal component 3'], c=finalDf['class'],cmap='prism', s =5)
plt.xlabel('pc2')
plt.ylabel('pc3')
plt.show()

# 3-D Printing
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9,9))
axes = Axes3D(fig)
axes.set_title('PCA Representation', size=14)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')

axes.scatter(finalDf['principal component 1'],finalDf['principal component 2'],finalDf['principal component 3'],c=finalDf['class'], cmap = 'prism', s=10)
plt.show()

# Feature importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(rescaledX,Y)
print(model.feature_importances_)

# ***********************************************************************
# Improve Speed and Avoid Overfitting of ML Models with PCA using Sklearn

X = df.drop('class',axis=1).values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
scaler.fit(X_test)
# Apply transform to both the training set and the test set.
#
X_train_pca = scaler.transform(X_train)
X_test_pca = scaler.transform(X_test)


st=time.time()
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train,y_train)
y_train_hat =logisticRegr.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_hat)*100
# Creating Logistic Regression Model without PCA
print('"Accuracy for our Training dataset without PCA is: %.4f %%' % train_accuracy)
print("Wall time (s): ----%.4f----"%(time.time()-st))
print(time.time()-st)

st=time.time()
y_test_hat=logisticRegr.predict(X_test)
test_accuracy=accuracy_score(y_test,y_test_hat)*100
print("Accuracy for our Testing dataset without PCA is : {:.3f}%".format(test_accuracy) )
print("Wall time (s): ----%.4f----"%(time.time()-st))
print(time.time()-st)

# Creating Logistic Regression Model with PCA
st=time.time()
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train_pca,y_train)

y_train_hat =logisticRegr.predict(X_train_pca)
train_accuracy = accuracy_score(y_train, y_train_hat)*100
print('"Accuracy for our Training dataset with PCA is: %.4f %%' % train_accuracy)
print("Wall time (s): ----%.4f----"%(time.time()-st))
print(time.time()-st)


st=time.time()
y_test_hat=logisticRegr.predict(X_test_pca)
test_accuracy=accuracy_score(y_test,y_test_hat)*100
print("Accuracy for our Testing dataset with PCA is : {:.3f}%".format(test_accuracy) )
print("Wall time (s): ----%.6f----"%(time.time()-st))
print(time.time()-st)

# Implement the PCA
# PCA help to visualize a high dimension dataset, reduces computation time,
# and avoid overfitting.