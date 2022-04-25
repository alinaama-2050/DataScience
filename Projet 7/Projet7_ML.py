#############################################################################################
#
# Description : This program used for Project 7 - ML Part
# Lead Scoring
#
# Author :
# Date : 08/04/2022
#
#
#############################################################################################


# Import libraries
import pandas as pd
import numpy as np# read csv data
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import statsmodels.api as sm
from statsmodels.compat import lzip
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import missingno as ms
import joblib
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pickle
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
plt.style.use('ggplot')

def func(pct):
    return "{:1.1f}%".format(pct)

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


def plot_frequency_and_percentage(feature, leads_df, category_df, another_row=False, height=8, ylabels=[]):
    if another_row:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, height * 2))
    else:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)

    ## plot the frequency plot for each category in the required column
    ax1.set_title('Frequency Plot of {0}'.format(feature), color='blue')
    ax1.set_ylabel(feature)
    ax1.set_xlabel('count')
    sns.countplot(y=feature, data=category_df.sort_values(by=feature), ax=ax1, color='green');
    if len(ylabels) > 0:
        ax1.set_yticklabels(ylabels);

    ## plot the value percentage in each sub-category wrt the label
    ax2.set_title('Lead Converted label %', color='blue')
    ax2.set_ylabel(feature)
    ax2.set_xlabel('percentage')
    leads_df.iloc[1].sort_values().plot(kind='barh', ax=ax2, color='orange');
    if len(ylabels) > 0:
        ax2.set_yticklabels(ylabels)


def check_rfe(X_df,y_df,col_list):
    logreg_model1 = LogisticRegression()
    logreg_model1.fit(X_df[col_list],y_df)
    # Use RFE for feature selection
    rfe1 = RFE(logreg_model1,15)
    rfe1.fit(X_train_scaled[col_list],y_df)
    #List all features with importance/ranking
    print(list(zip(col_list,rfe1.support_,rfe1.ranking_)))
    # Plot pixel ranking - Added Ana
    ranking = rfe1.ranking_
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking feautures with RFE")
    plt.show()


def build_statsmodel(X_df, y_df):
    # add constant. Its because intercept might not pass through origin
    # By default its not added in stats model
    # we have to add constant explicitly
    X_train_sm = sm.add_constant(X_df)
    lr = sm.GLM(y_df, X_train_sm, family=sm.families.Binomial()).fit()

    ## Print the params obtained
    print('************ feature - coefficients *****************')
    print(round(lr.params, 4))
    print('*****************************************************')
    print()
    print()
    ## Print stats model summary
    print(lr.summary())
    return lr, X_train_sm


def check_vif(X_df):
    vif = pd.DataFrame()
    vif['Features'] = X_df.columns
    vif['VIF'] = [variance_inflation_factor(X_df.values,i) for i in range(X_df.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


def createfinaldf_and_checkscores(y_df, y_pred_df):
    y_pred_final_df = pd.DataFrame({'Converted': y_df.values, 'Converted_Prob': y_pred_df})
    y_pred_final_df['LeadId'] = y_df.index

    # Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
    ## this means cut-off set is 0.5
    y_pred_final_df['predicted'] = y_pred_final_df.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
    # print(y_train_pred_final.head())

    # Confusion matrix
    cm = confusion_matrix(y_pred_final_df['Converted'], y_pred_final_df['predicted'])
    # print(conf_matrix)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d');  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['unsuccessfull', 'successfull']);
    ax.yaxis.set_ticklabels(['unsuccessfull', 'successfull']);
    plt.show();

    # Let's check the overall accuracy.
    print('Accuracy score: {0}'.format
          (accuracy_score(y_pred_final_df['Converted'], y_pred_final_df['predicted'])))
    # Let's check sensitivity
    TP = cm[1, 1]  # true positive
    TN = cm[0, 0]  # true negatives
    FP = cm[0, 1]  # false positives
    FN = cm[1, 0]  # false negatives
    sensitivity = TP / float(TP + FN)
    print('Sensitivity score: {0}'.format(sensitivity))
    # Lets check f1-score
    print('f1-score: {0}'.format
          (f1_score(y_pred_final_df['Converted'], y_pred_final_df['predicted'])))
    # Lets check precision score
    print('Precision score: {0}'.format
          (precision_score(y_pred_final_df['Converted'], y_pred_final_df['predicted'])))
    # Lets check Recall Score
    print('Recall score: {0}'.format
          (recall_score(y_pred_final_df['Converted'], y_pred_final_df['predicted'])))

    return y_pred_final_df, cm

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return None




def plot_feature_importance(importance,names,model_type):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# Read ML Dataframe

df = pd.read_csv("data\LeadsFranceML.csv")

# First make subset of dataframes based on datatype - object and numerical
object_cols = [col for col in df.select_dtypes(include=np.object).columns]
numerical_cols = [col for col in df.select_dtypes(include=np.number).columns]
## Remove Converted from Numerical Columns list

print(numerical_cols)
#print(object_cols )

# Observations from above correlation map
#     TotalWebsiteTime, LeadOrigin_Lead Add Form, LastActivity_SMS Sent,
#     Occupation_Working Professional, LastNotableActivity_SMS Sent features
#     have positive correlation with target variable Converted
#     There are many dummy variables that are correlated with each other


# Divide into X and Y for model building
df = df.dropna()
#df.reset_index()
X = df.drop(columns = ['Converted'])
y = df['Converted']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

X_test.to_csv(r'D:\Projet Perso\Ali\Data Scientist\Projet 7\data\Leads_X_test.csv',index=False)

print(X_train.head())
# Feature Scaling
# Back to Table of Contents
# Apply StandardScalar on all continuous numerical features

sc = StandardScaler()
#sc = MinMaxScaler()

#Create X_train_scaled, X_test_scaled
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()


#Fit and transform Train
#numerical_cols = numerical_cols.remove('Converted')
X_train_scaled = sc.fit_transform(X_train)

#Transform Test (No Fit)
X_test_scaled = sc.transform(X_test)


# Plot Feature Importance - DTC

dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train_scaled, y_train)
y_pred_gini = dt_clf_gini.predict(X_test_scaled)

plot_feature_importance(dt_clf_gini.feature_importances_,df.drop(columns = ['Converted']).columns,'DTC-CART')

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )

# Plot Feature Importance - LR



# Plot Feature Importance

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(dt_clf_gini, open("model_lead_dt.pkl","wb"))


# Evaluate algorithms: baseline
num_folds = 10
seed = 7
scoring = 'accuracy'

models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model,X_train_scaled,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"%(name, cv_results.mean(),cv_results.std())
    print(msg)


fig = pyplot.figure(figsize = (10,10))
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Save Model :
pickle.dump(models[0], open('models/model_Lead_LR.pkl', 'wb'))
pickle.dump(models[3], open('models/model_Lead_CART.pkl', 'wb'))

# Standardize data

pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()),('LR',LogisticRegression())])))
pipelines.append(('ScaledLDA',Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5))])))
pipelines.append(('ScaledNB',Pipeline([('Scaler',StandardScaler()),('NB',GaussianNB())])))
pipelines.append(('ScaledSVM',Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_results=cross_val_score(model,X_train_scaled,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)

fig = pyplot.figure(figsize= (10,10))
fig.suptitle('Scaled algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Algorithm tuning
# Tunning KNN

scaler = StandardScaler().fit(X_train_scaled)
rescaledX = scaler.transform(X_train_scaled)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold =KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(rescaledX,y_train)
print("Best: %f using %s"%(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev,param in zip(means,stds,params):
    print("%f (%f) with :%r"%(mean,stdev,param))

# Tuning SVM

scaler = StandardScaler().fit(X_train_scaled)
rescaledX = scaler.transform(X_train_scaled)
c_values = [0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0]
kernel_values = ['linear','poly','rbf','sigmoid']
param_grid = dict(C=c_values,kernel = kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result=grid.fit(rescaledX,y_train)
print("Best: %f using %s"%(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev,param in zip(means,stds,params):
    print("%f (%f) with: %r" %(mean,stdev,param))

# Ensemble methods

ensembles = []
ensembles.append(('AB',AdaBoostClassifier()))
ensembles.append(('GBM',GradientBoostingClassifier()))
ensembles.append(('RF',RandomForestClassifier()))
ensembles.append(('ET',ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits = num_folds,random_state=seed)
    cv_results = cross_val_score(model, rescaledX,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)

fig = pyplot.figure(figsize = (10,10))
fig.suptitle('Ensemble algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



# Finalize model
#y_train, y_test

scaler = StandardScaler().fit(X_train_scaled)
rescaledX = scaler.transform(X_train_scaled)
model = SVC(C=1.5)
model.fit(rescaledX,y_train)
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Confusion matrix SVM
import seaborn as sns
cf_matrix = confusion_matrix(y_test,predictions)
ax = plt.subplot()
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix SVM with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# Confusion matrix DTC

modelDTC = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
modelDTC.fit(rescaledX,y_train)
rescaledValidationX = scaler.transform(X_test)
predictionsmodelDTC = modelDTC.predict(rescaledValidationX)
print(accuracy_score(y_test,predictionsmodelDTC))
print(confusion_matrix(y_test,predictionsmodelDTC))
print(classification_report(y_test,predictionsmodelDTC))

cf_matrix = confusion_matrix(y_test,predictionsmodelDTC)
ax = plt.subplot()
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix DTC with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

# Feature importance :

# get importance
from matplotlib import pyplot
importance = modelDTC.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
