

###############################################################################
#
# Description : This program is used for Credit Scoring 
# from 1 csv file containing German Credit data Set Forked
# Author : Ali Naama forked from Applied DS using Pyspark
# Date : 02/05/2022
"""
Credit Scoring - German Credit data Set Forked
Core Functionality:
0. Data Exploration
1. Performs feature selection
2. Develops machine learning models
3. Validates the models on hold out datasets
4. Picks the best algorithm to deploy based on user selected statistics (ROC, KS, Accuracy)
"""
###############################################################################


from pyspark import SparkContext,HiveContext,Row,SparkConf
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import *
from pyspark.mllib.stat import *
from pyspark.ml.feature import *
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import count, when, isnan, col
from pyspark.ml.feature import MinMaxScaler
from sklearn.metrics import roc_curve,auc
import numpy as np
import pandas as pd
import subprocess
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import functions as func
from datetime import *
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.types import *
from dateutil.relativedelta import relativedelta
import datetime
from datetime import date
import string
import os
import pickle
import sys
import time
import numpy
from pyspark import SparkConf, SparkContext
from pyspark.sql import functions as F
from pyspark.sql.functions import approxCountDistinct, countDistinct
from pyspark.ml.classification import DecisionTreeClassifier
#
# Import libraries
import pandas as pd
import numpy as np# read csv data
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno as ms
import findspark
findspark.init()
from pandas import ExcelWriter
import glob
# include extra def
from helper import *

# 0. Init Spark Session

spark = SparkSession \
    .builder \
    .appName("Credit_Score") \
    .getOrCreate()

# Read csv file
# By setting inferSchema=true , Spark will automatically go through the csv file and infer the
# schema of each column. This requires an extra pass over the file which will result in reading a file with inferSchema
# set to true being slower. But in return the dataframe will most likely have a correct schema given its input

S3_DATA_SOURCE_PATH = 's3://creditscoringbucket/data/lsdata_german.csv'
S3_DATA_OUTPUT_FOLDER = 's3://creditscoringbucket/output'
df = spark.read.csv(S3_DATA_SOURCE_PATH , inferSchema=True, header=True)

#df = spark.read.csv("data/data_german.csv", inferSchema=True, header=True)

# change value of score to get binary : 0 or 1
df = df.withColumn('Score', F.when(F.col("Score") == '2', 1).otherwise(0))
print(df.head())

pandasDF = df.toPandas()
print(pandasDF)


# EDA - "Null Values Plot
# Length of the data

df.count()

df.describe().toPandas() # missing and cardinality
# variable types :
df.dtypes
df.printSchema()

# single variable group by
df.groupBy('Score').count().show()

plt.title("Null Values Plot")
#ms.bar(df, color = 'slategrey')
ms.bar(pandasDF, color = 'blue')
plt.savefig( 'image/Chart Null Values Plot_' +  'data.png', bbox_inches='tight')
plt.close()
#plt.show()

# 1. Print missing_value_calculation :

print(missing_value_calculation(df))

final_missing_df, vars_selected = missing_calculation(df)
print(final_missing_df)
print(vars_selected)

# 2. Identify variable type :
# Identify the data type for each variables
char_vars, num_vars = identify_variable_type(df)
print(char_vars)
print(num_vars)

# 3. cardinality check

cardinality_df, cardinality_vars = cardinality_calculation(df)
print(cardinality_df)
print(cardinality_vars)

#    4. Convert Categorical to Numerical using Label encoders
df, char_labels = category_to_index(df, char_vars)
print(char_labels)
print(df)
df.dtypes

# 4.1 select only num vars
df = df.select([c for c in df.columns if c not in char_vars])

df = rename_columns(df, char_vars)
df.dtypes
#6. Assemble - individual variables into single feature vector
print(df.dtypes)

df.groupBy('Score').count().show()
#exclude target variable and select all other feature vectors
target_variable_name = "Score"
features_list = df.columns
dffull = df
features_list.remove(target_variable_name)
# apply the function on our dataframe

df = assemble_vectors(df, features_list, target_variable_name)
df.show()
df.schema["features"].metadata["ml_attr"]["attrs"]

for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
    features_df = pd.DataFrame(v)

print(features_df)

# 5. feature importance :


dt = DecisionTreeClassifier(featuresCol='features', labelCol=target_variable_name)
dt_model = dt.fit(df)
dt_model.featureImportances

#temporary output rf_output
dt_output = dt_model.featureImportances
features_df['Decision_Tree'] = features_df['idx'].apply(lambda x: dt_output[x] if x in dt_output.indices else 0)
print(features_df)
writeHtml(features_df, 'data/', 'features_df.html')
# Draw Importance Df features
draw_feature_importance(features_df, 'image/')


# 6. Test / Test : Split

print('vars selected')
vars_selectedn = df.columns
X = df.cache()
target_column_name = target_variable_name
Y = dffull.select(target_column_name)
Y = Y.cache()
joinedDF = join_features_and_target(X, Y)
joinedDF = joinedDF.cache()
print('Features and targets are joined')
#

train_size=0.7
valid_size=0.2 #Train -Test difference is used for test data
seed=2308
train, valid, test = train_valid_test_split(df, train_size, valid_size, seed)
train = train.cache()
valid = valid.cache()
test = test.cache()
print('Train, valid and test dataset created')

x = train.columns
x.remove(target_column_name)
feature_count = len(x)
print(feature_count)

if feature_count > 30:
    print('# No of features - ' + str(feature_count) + '.,  Performing feature reduction before running the model.')


sel_train = train
sel_train = sel_train.cache()
# # Variable Reduction for more than 30 variables in the feature set using Random Forest

from pyspark.ml.classification import  RandomForestClassifier

rf = RandomForestClassifier(featuresCol="features",labelCol = target_column_name)
mod = rf.fit(sel_train)
varlist = ExtractFeatureImp(mod.featureImportances, sel_train, "features")
print(varlist)


# 7. Modeling 

run_logistic_model = 1
run_randomforest_model = 1
run_boosting_model = 1
run_neural_model = 1

loader_model_list = []
dataset_list = ['train','valid','test']
datasets = [train,valid,test]

print(train.count())
print(test.count())



models_to_run = []

# 8. apply the transformation done on train dataset

x = 'features'
y = target_column_name
mdl_ltrl = 'credit_scoring_testrun' #An unique literal or tag to represent the model

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from metrics_calculator import *
import sys
import time
# import __builtin__ as builtin
import builtins
import numpy

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import RandomForestClassifier

# from sklearn.externals import joblib
import joblib

def logistic_model(train, x, y):
    lr = LogisticRegression(featuresCol = x, labelCol = y, maxIter = 10)
    lrModel = lr.fit(train)
    return lrModel

# Model


if run_randomforest_model:

    clf = RandomForestClassifier(featuresCol='features', labelCol='Score')
    clf_model = clf.fit(train)
    doutput = clf_model.featureImportances
    print(doutput)
    varlist = ExtractFeatureImp(clf_model.featureImportances, train, "features")
    print(varlist)
    features_df['RandomForestClassifier'] = features_df['idx'].apply(lambda x: doutput[x] if x in doutput.indices else 0)
    print(features_df)
    # Draw Importance Df features
    draw_feature_importance_xtd(features_df, 'image/', 'RandomForestClassifier')
    print(clf_model.toDebugString)
    train_pred_result = clf_model.transform(train)
    test_pred_result = clf_model.transform(test)

    train_cm, train_acc, train_miss_rate, train_precision,train_recall, train_f1, train_roc, train_pr = evaluation_metrics(train_pred_result, target_variable_name)
    test_cm, test_acc, test_miss_rate, test_precision, test_recall, test_f1, test_roc, test_pr = evaluation_metrics(test_pred_result, target_variable_name)

    print('Train accuracy - ', train_acc, ', Test accuracy - ', test_acc)
    print('Train misclassification rate - ', train_miss_rate, ', Test misclassification rate - ', test_miss_rate)
    print('Train precision - ', train_precision, ', Test precision - ', test_precision)
    print('Train recall - ', train_recall, ', Test recall - ', test_recall)
    print('Train f1 score - ', train_f1, ', Test f1 score - ', test_f1)
    print('Train ROC - ', train_roc, ', Test ROC - ', test_roc)
    print('Train PR - ', train_pr, ', Test PR - ', test_pr)

    make_confusion_matrix_chart(train_cm, test_cm)

    plot_roc_pr(train_pred_result, target_variable_name, 'roc', train_roc, 'Train ROC')
    plot_roc_pr(test_pred_result, target_variable_name, 'roc', test_roc, 'Test ROC')
    plot_roc_pr(train_pred_result, target_variable_name, 'pr', train_pr, 'Train Precision-Recall curve')

    #models_to_run.append('RandomForest')
    #loader_model_list.append(RandomForestClassifier)


if run_logistic_model:
        lrModel = logistic_model(train, x, y) #build model
        #lrModel.write().overwrite().save('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/logistic_model.h5') #save model object
        print("Logistic model developed")
        model_type = 'Logistic'
        l = []

        for i in datasets:
            l += model_validation( mdl_ltrl, i, y, lrModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot( mdl_ltrl, model_type) #ks charts
        #joblib.dump(l,'/home/' + user_id + '/' + 'mla_' + mdl_ltrl  + '/logistic_metrics.z') #save model metrics
        models_to_run.append('logistic')
        loader_model_list.append(LogisticRegressionModel)

