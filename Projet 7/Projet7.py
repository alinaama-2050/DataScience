#############################################################################################
#
# Description : This program used for Project 7
# Lead Scoring
#
# Author :
# Date : 08/04/2022
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

# Read csv file
data = pd.read_csv("data\LeadsFranceDate.csv")
# Start EDA
print(data.head())
print(data['Country'].unique())
print(data['City'].unique())
print (data.info())



# Filling the fill-percentile of each file
# We use the mean() function twice to calculate the mean for each columns, and then the mean for the whole file
# We create a table containing all information about the 2 files
files_description = pd.DataFrame(
    columns=["Nb lignes", "Nb colonnes", "Taux remplissage moyen", "Valeurs null", "Doublons", "Description"],
    index=["Leads_France.csv"])


df = data
# Filling the total number rows in each file
files_description["Nb lignes"] = [len(df.index)]

# Filling the number of columns in each file
files_description["Nb colonnes"] = [len(df.columns)]

#  Null Values
files_description["Valeurs null"] = [df.isnull().sum().sum()]

# Filling the fill-percentile of each file
# We use the mean() function twice to calculate the mean for each columns, and then the mean for the whole file
files_description["Taux remplissage moyen"] = [format_percentage(df.notna().mean().mean())]

# FIlling the number of duplicate keys for each file

files_description["Doublons"] = [df.duplicated().sum()]

# Finally adding a short description for each file
files_description["Description"] = ["Leads_France.csv"]

print(files_description)
writeHtml(files_description, 'D:/Projet Perso/Ali/Data Scientist/Projet 7/data/', 'files_description.html')

# Rename Column

df.rename(columns={'Converted':'Converted',
                            'Prospect ID':'ProspectId',
                            'Lead Number':'LeadNumber',
                            'Lead Origin':'LeadOrigin',
                            'Lead Source':'LeadSource',
                            'Do Not Email':'DndEmail',
                            'Do Not Call':'DndCall',
                            'TotalVisits':'TotalVisits',
                            'Total Time Spent on Website':'TotalWebsiteTime',
                            'Page Views Per Visit':'PagesPerVisit',
                            'Last Activity':'LastActivity',
                            'Country':'Country',
                            'Specialization':'Specialization',
                            'How did you hear about our brand':'HowHeard',
                            'What is your current occupation':'Occupation',
                            'What matters most to you in choosing our product':'MattersMost',
                            'Search':'Search',
                            'Magazine':'Magazine',
                            'Newspaper Article':'PaperArticle',
                            'X Education Forums':'EducationForum',
                            'Newspaper':'Newspaper',
                            'Digital Advertisement':'DigitalAd',
                            'Through Recommendations':'Recommendation',
                            'Receive More Updates About Our Products':'ReceiveUpdate',
                            'Tags':'Tags',
                            'Lead Quality':'LeadQuality',
                            'Update me on Supply Chain Content':'UpdateSupplyChain',
                            'Get updates on DM Content':'UpdateDMContent',
                            'Lead Profile':'LeadProfile',
                            'City':'City',
                            'Asymmetrique Activity Index':'ActivityIndex',
                            'Asymmetrique Profile Index':'ProfileIndex',
                            'Asymmetrique Activity Score':'ActivityScore',
                            'Asymmetrique Profile Score':'ProfileScore',
                            'I agree to pay the amount through cheque':'ChequePayment',
                            'A free copy of our White Book':'FreeCopy White Book',
                            'Last Notable Activity':'LastNotableActivity'},
                   inplace=True)

# Bring target column 'Converted' in front
col_list = list(df.columns)
col_list.insert(0,col_list.pop(col_list.index('Converted')))
df = df.loc[:,col_list]

# EDA
plt.title("Null Values Plot")
#ms.bar(df, color = 'slategrey')
ms.bar(df, color = 'slategrey')
plt.show()


# Check Values with Select :

print(df[df == 'Select'].count())
# Remplacement des colonnes ayant Select par nan

df.replace('Select',np.NaN,inplace=True)
print(df[df == 'Select'].count())

# Look for missing value percentage
print(df[df.columns[df.isnull().any()]].isnull().sum())

# Tableau de % des valeurs manquantes :

#In terms of missing%
percent_missing = df.isnull().sum() * 100 / len(df)
print(percent_missing)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df.sort_values(by=['percent_missing'], ascending=False))



# Data Cleaning
# Back to Table of Contents
# 6.1 Lets drop columns with more than 30% missing values

#Get the list of columns with missing% > 30%


missing_data_cols = list(missing_value_df[missing_value_df['percent_missing']>30]['column_name'])
print(missing_data_cols)

print('number of columns before dropping high pcnt missing value columns are: {0}'.format(df.shape[1]))
df.drop(columns=missing_data_cols,inplace=True)
print('number of columns after dropping high pcnt missing value columns are: {0}'.format(df.shape[1]))



# Lets check for column Country / City
df['City'].value_counts()
plt.figure(figsize=(8,8))
plt.title('frequency plot of City',color='blue')
sns.countplot(y='City',hue='Converted',data=df);
plt.show()

# frequency plot of What matters most
df['MattersMost'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of What matters most',color='blue')
sns.countplot(y='MattersMost',hue='Converted',data=df)
plt.show()


# Check values distribution for last variables :
# LeadSource
print(df['LeadSource'].value_counts())
# Create a sub category
df['LeadSource'] = df['LeadSource'].replace(
    ['bing','google','Click2call','Social Media','Press_Release','Live Chat','WeLearn','youtubechannel',
     'welearnblog_Home','NC_EDM','testone','Pay per Click Ads','blog'],'Others')
print(df['LeadSource'].value_counts())


# Lets check for column TotalVisits



# Total Stat - for outlayers viz
print(df['TotalVisits'].describe([0,0.05,0.25,0.5,0.75,0.9,0.95,0.99]))

print('TotalVisits 99th percentile value: {0}'.format(df['TotalVisits'].quantile(0.99)))
plt.figure(figsize=(6,6));
sns.countplot(y=df[df['TotalVisits']>17]['TotalVisits'].sort_values());
plt.show()

print('shape of dataframe before dropping rows: {0}'.format(df.shape))
print('Total number of rows to be dropped: {0}'.format(df[df['TotalVisits']>55]['TotalVisits'].count()))
df.drop(index=df[df['TotalVisits']>55].index,inplace=True)
print('shape of dataframe after dropping rows: {0}'.format(df.shape))

# box Plot - NbCar Name
sns.boxplot(x=df["Converted"], y=df["TotalVisits"])
plt.show()


# Page per Visit :
sns.boxplot(df['PagesPerVisit'])
plt.show()

print(df['PagesPerVisit'].describe([0,0.05,0.25,0.5,0.75,0.9,0.95,0.99]))

print('TotalVisits 99th percentile value: {0}'.format(df['PagesPerVisit'].quantile(0.99)))
plt.figure(figsize=(6,6));
sns.countplot(y=df[df['PagesPerVisit']>9]['PagesPerVisit'].sort_values());
plt.show()

#  Lets check for column LastActivity

plt.figure(figsize=(6,6))
plt.title('frequency plot of Last Activity',color='blue')
sns.countplot(y='LastActivity',hue='Converted',data=df)
plt.show()

# Since below categories have very less observations so lets combine the below categories and create new category value - Others
#
#     Unreachable
#     Unsubscribed
#     Had a Phone Conversation
#     Approached upfront
#     View in browser link Clicked
#     Email Marked Spam
#     Email Received
#     Resubscribed to emails
#     Visited Booth in Tradeshow

df['LastActivity'] = df['LastActivity'].replace(['Unreachable','Unsubscribed','Had a Phone Conversation','Approached upfront','View in browser link Clicked',
     'Email Marked Spam','Email Received','Resubscribed to emails','Visited Booth in Tradeshow'],'Others')

print(df['LastActivity'].value_counts())

## replace null values
print('Number of observations with null values in column LastActivity: {0}'.format( df['LastActivity'].isnull().sum()))

# Use SimpleImputer class to impute missing values
imp = SimpleImputer(missing_values=np.NaN, strategy= 'most_frequent')
imp.fit(df[['LastActivity']])
print('Most frequent value is : {0}'.format(imp.statistics_[0]))
df['LastActivity'] = imp.transform(df[['LastActivity']]).ravel()
print('Number of observations with null values in column LastActivity after imputation: {0}'.format(df['LastActivity'].isnull().sum()))


# Drop columns with only one categorical value
# Observation from above cells
# Since below listed columns dont add any variance in data so it wont be helpful in modeling. Based on this we can drop these columns
# Magazine
# ReceiveCourseUpdate
# UpdateSupplyChain
# UpdateDMContent
# ChequePayment

print(df['Magazine'].value_counts())
print(df['ReceiveUpdate'].value_counts())
print(df['UpdateSupplyChain'].value_counts())
print(df['UpdateDMContent'].value_counts())
print(df['ChequePayment'].value_counts())
print('number of columns before dropping column "Country" are: {0}'.format(df.shape[1]))
df.drop(columns=['Magazine','ReceiveUpdate','UpdateSupplyChain','UpdateDMContent','ChequePayment'], inplace=True)


# Analysis
# Converted



fig, axes = plt.subplots(1,1, figsize=(8,6))
# We do a pie plot on the axes
labels = ['Unsuccessful','Successful']
axes.pie(x=df["Converted"].value_counts(),labels=labels, autopct=lambda pct: func(pct), shadow=True)
plt.axis('equal')
plt.title('% of Unsuccessful Leads vs. Successful Converted Leads',color='blue')
plt.show()


plt.figure(figsize=(5,5))
sns.countplot(df['Converted']);
plt.title('No of Unsuccessful Leads vs. Successful Converted Leads',color='blue')
plt.xticks(np.arange(2),('Unsuccessful','Successful'));
#plt.legend(['0 - non-Converted','1 - Converted']);
plt.show()



# Lets plot Lead Origin

df['LeadOrigin'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of Lead Origin',color='blue')
sns.countplot(y='LeadOrigin',hue='Converted',data=df);
plt.show()

crosstab_df = pd.crosstab(df['Converted'],
            df['LeadOrigin']).apply(lambda x: round((x/x.sum())*100,2), axis=0)
print(crosstab_df)
plot_frequency_and_percentage('LeadOrigin',crosstab_df,df.sort_values(by='LeadOrigin',ascending=False),True,6)
plt.show()



# Lets plot Lead Source


df['LeadSource'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of Lead Source',color='blue')
sns.countplot(y='LeadSource',hue='Converted',data=df);
plt.show()

crosstab_df = pd.crosstab(df['Converted'],
            df['LeadSource']).apply(lambda x: round((x/x.sum())*100,2), axis=0)
print(crosstab_df)

plot_frequency_and_percentage('LeadSource',crosstab_df,df.sort_values(by='LeadSource',ascending=False),True,6)
plt.show()

# Lets plot Segmentation
df['Segmentation'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of Segmentation',color='blue')
sns.countplot(y='Segmentation',hue='Converted',data=df);
crosstab_df = pd.crosstab(df['Converted'],df['Segmentation']).apply(lambda x: round((x/x.sum())*100,2), axis=0)
print(crosstab_df)
plot_frequency_and_percentage('Segmentation',crosstab_df,df.sort_values(by='Segmentation',ascending=False),True,6)
plt.show()

# Lets plot Last Acitivity
df['LastActivity'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of Last Activity',color='blue')
sns.countplot(y='LastActivity',hue='Converted',data=df);
plt.show()
crosstab_df = pd.crosstab(df['Converted'],df['LastActivity']).apply(lambda x: round((x/x.sum())*100,2), axis=0)
print(crosstab_df)

# Lets plot Do not email
df['DndEmail'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of Do not Email',color='blue')
sns.countplot(y='DndEmail',hue='Converted',data=df);
plt.show()

crosstab_df = pd.crosstab(df['Converted'],df['DndEmail']).apply(lambda x: round((x/x.sum())*100,2), axis=0)
crosstab_df

plot_frequency_and_percentage('DndEmail',crosstab_df,df.sort_values(by='DndEmail',ascending=False),False,6)
plt.show()


# Observations from above categorical features

print('number of columns before dropping column "Country" are: {0}'.format(df.shape[1]))
df.drop(columns=['DndCall','Search','PaperArticle','Forums','Newspaper','DigitalAd','Recommendation'],inplace=True)
print('number of columns after dropping column "Country" are: {0}'.format(df.shape[1]))

# Lets plot feature - a free copy of Mastering The Interview

df['FreeCopy White Book'].value_counts()
plt.figure(figsize=(6,6))
plt.title('frequency plot of a free copy of mastering the interview',color='blue')
sns.countplot(y='FreeCopy White Book',hue='Converted',data=df);
plt.show()

crosstab_df = pd.crosstab(df['FreeCopy White Book'],df['FreeCopy White Book']).apply(lambda x: round((x/x.sum())*100,2), axis=0)
crosstab_df
plot_frequency_and_percentage('FreeCopy White Book',crosstab_df,df.sort_values(by='FreeCopy White Book',ascending=False),False,6)
plt.show()


# Data Preparation for modeling

# LeadNumber and ProspectId are case identifiers and dont add much value to modelling so will drop these columns
# Createddate as well need to be removed
# drop Createddate *********************************************************
#df.drop(columns = ['CreatedDate'])
print('No. of columns before dropping 2 columns: {0}'.format(df.shape[1]))
df.drop(columns = ['ProspectId','LeadNumber', 'CreatedDate'],inplace=True)
print('No. of columns after dropping 2 columns: {0}'.format(df.shape[1]))

print(df.head())


# First make subset of dataframes based on datatype - object and numerical
object_cols = [col for col in df.select_dtypes(include=np.object).columns]
numerical_cols = [col for col in df.select_dtypes(include=np.number).columns]
## Remove Converted from Numerical Columns list

print('Categorical Columns: \n{0}\n'.format(object_cols))
print('Numerical Columns: \n{0}'.format(numerical_cols))
numerical_cols.remove('Converted')
print('Categorical Columns: \n{0}\n'.format(object_cols))
print('Numerical Columns: \n{0}'.format(numerical_cols))
# Apply get_dummies
# The get_dummies() function is used to convert categorical variable into dummy/indicator variables.

#one hot encoding for categorical variables
df = pd.get_dummies(data=df,columns=object_cols,drop_first=True)
print(df.sample(10))

plt.figure(figsize=(20,11))
sns.heatmap(df.corr(),annot=True);
plt.show()

leadscore_corr = df.corr()
leadscore_corr.style.apply(lambda x:
                           ["background: lightblue" if abs(v) > 0.5
                            else
                            ("background: lightyellow" if abs(v) > 0.3
                             else "" ) for v in x], axis = 1)
#leadscore_corr.style.apply(lambda x: ["background: lightblue" if abs(v) > 0.5 else "" for v in x], axis = 1)
# Set Data into csv
df.to_csv(r'data\LeadsFranceML.csv',index=False)
