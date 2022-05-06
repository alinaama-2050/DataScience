
"""
Helper - All module need from main
Credit Scoring - German Credir data Set Forked
Core Functionality:
0. Data Exploration
1. Performs feature selection
2. Develops machine learning models
3. Validates the models on hold out datasets
4. Picks the best algorithm to deploy based on user selected statistics (ROC, KS, Accuracy)
5. Produces pseudo score code
"""


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
from metrics_calculator import *
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import IntegerType, DoubleType


def rename_columns(X, char_vars):
    mapping = dict(zip([i+ '_index' for i in char_vars], char_vars))
    X = X.select([col(c).alias(mapping.get(c, c)) for c in X.columns])
    return X

def missing_value_calculation(X, miss_per=0.75):

    missing = X.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in X.columns])
    missing_len = X.count()
    final_missing = missing.toPandas().transpose()
    final_missing.reset_index(inplace=True)
    final_missing.rename(columns={0:'missing_count'},inplace=True)
    final_missing['missing_percentage'] = final_missing['missing_count']/missing_len
    vars_selected = final_missing['index'][final_missing['missing_percentage'] <= miss_per]
    return vars_selected


def missing_calculation(df, miss_percentage=0.45):
    # checks for both NaN and null values
    missing = df.select(*[count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    length_df = df.count()
    ## convert to pandas for efficient calculations
    final_missing_df = missing.toPandas().transpose()
    final_missing_df.reset_index(inplace=True)
    final_missing_df.rename(columns={0: 'missing_count'}, inplace=True)
    final_missing_df['missing_percentage'] = final_missing_df['missing_count'] / length_df

    # select variables with cardinality of 1
    vars_selected = final_missing_df['index'][final_missing_df['missing_percentage'] >= miss_percentage]

    return final_missing_df, vars_selected


def identify_variable_type(X):

    l = X.dtypes
    char_vars = []
    num_vars = []
    for i in l:
        if i[1] in ('string'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars



def categorical_to_index(X, char_vars):
    chars = X.select(char_vars)
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",handleInvalid="keep") for column in chars.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(chars)
    X = char_labels.transform(X)
    return X, char_labels


def category_to_index(df, char_vars):
    char_df = df.select(char_vars)
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep") for c in char_df.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(char_df)
    df = char_labels.transform(df)
    return df, char_labels


#assemble individual columns to one column - 'features'
def assemble_vectors(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    stages = [assembler]
    #select all the columns + target + newly created 'features' column
    selectedCols = [target_variable_name, 'features']
    #use pipeline to process sequentially

    # MinMaxScaler Transformation
    #scaler = MinMaxScaler(inputCol=features_list + "_Vect", outputCol='features' +  "_Scaled")
    pipeline = Pipeline(stages=stages)
    #assembler model
    assembleModel = pipeline.fit(df)
    #apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)

    return df

"""
Note: approxCountDistinct and countDistinct can be used interchangeably. Only difference is the computation time. 
"approxCountDistinct" is useful for large datasets 
"countDistinct" for small and medium datasets.
"""
def cardinality_calculation(df, cut_off=1):
    cardinality = df.select(*[approxCountDistinct(c).alias(c) for c in df.columns])

    ## convert to pandas for efficient calculations
    final_cardinality_df = cardinality.toPandas().transpose()
    final_cardinality_df.reset_index(inplace=True)
    final_cardinality_df.rename(columns={0: 'Cardinality'}, inplace=True)

    # select variables with cardinality of 1
    vars_selected = final_cardinality_df['index'][final_cardinality_df['Cardinality'] <= cut_off]

    return final_cardinality_df, vars_selected

# The module below is used to write HTML table from df
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



# The module below is used to draw the feature importance plot
def draw_feature_importance(importance_df, destdir):

    importance_df = importance_df.sort_values('Decision_Tree')
    plt.figure(figsize=(15,15))
    plt.title('Feature Importances')
    plt.barh(range(len(importance_df['Decision_Tree'])), importance_df['Decision_Tree'], align='center')
    plt.yticks(range(len(importance_df['Decision_Tree'])), importance_df['name'])
    plt.ylabel('Variable Importance')
    plt.savefig(destdir + 'Features selected for modeling.png', bbox_inches='tight')
    plt.close()
    return None

# The module below is used to draw the feature importance plot
def draw_feature_importance_xtd(importance_df, destdir, col):

    importance_df = importance_df.sort_values(col)
    plt.figure(figsize=(15,15))
    plt.title('Feature Importances')
    plt.barh(range(len(importance_df[col])), importance_df[col], align='center')
    plt.yticks(range(len(importance_df[col])), importance_df['name'])
    plt.ylabel('Variable Importance')
    plt.savefig(destdir + col + 'Features selected for modeling.png', bbox_inches='tight')
    plt.close()
    return None

#    Train, Valid and Test data creator

def train_valid_test_split(df, train_size=0.4, valid_size=0.3,seed=12345):

    train, valid, test = df.randomSplit([train_size, valid_size,1-train_size-valid_size], seed=12345)
    return train,valid,test



#    6. Join X and Y vector using a monotonically increasing row id

def join_features_and_target(X, Y):

    X = X.withColumn('id', F.monotonically_increasing_id())
    Y = Y.withColumn('id', F.monotonically_increasing_id())
    joinedDF = X.join(Y,'id','inner')
    joinedDF = joinedDF.drop('id')
    return joinedDF




#    8. Assembling vectors

def assembled_vectors(train,list_of_features_to_scale,target_column_name):

    stages = []
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='features')
    stages=[assembler]
    selectedCols = [target_column_name,'features'] + list_of_features_to_scale

    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(train)

    train = assembleModel.transform(train).select(selectedCols)
    return train

#assemble individual columns to one column - 'features'
def assemble_vectors2(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    stages = [assembler]
    #select all the columns + target + newly created 'features' column
    selectedCols = [target_variable_name, 'features'] + features_list
    #use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    #assembler model
    assembleModel = pipeline.fit(df)
    #apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)

    return df, assembleModel, selectedCols

def logistic_model(train, x, y):
    lr = LogisticRegression(featuresCol = x, labelCol = y, maxIter = 10)
    lrModel = lr.fit(train)
    return lrModel

# The module below is used to calculate the feature importance for each variables based on the Random Forest output. The feature importance is used to reduce the final variable list to 30.
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    """
    Takes in a feature importance from a random forest / GBT model and map it to the column names
    Output as a pandas dataframe for easy reading
    rf = RandomForestClassifier(featuresCol="features")
    mod = rf.fit(train)
    ExtractFeatureImp(mod.featureImportances, train, "features")
    """

    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['Importance_Score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('Importance_Score', ascending = False))


def evaluation_metrics(df, target_variable_name):
    pred = df.select("prediction", target_variable_name)
    pred = pred.withColumn(target_variable_name, pred[target_variable_name].cast(DoubleType()))
    pred = pred.withColumn("prediction", pred["prediction"].cast(DoubleType()))
    metrics = MulticlassMetrics(pred.rdd.map(tuple))
    # confusion matrix
    cm = metrics.confusionMatrix().toArray()
    acc = metrics.accuracy #accuracy
    misclassification_rate = 1 - acc #misclassification rate
    precision = metrics.precision(1.0) #precision
    recall = metrics.recall(1.0) #recall
    f1 = metrics.fMeasure(1.0) #f1-score
    #roc value
    evaluator_roc = BinaryClassificationEvaluator(labelCol=target_variable_name, rawPredictionCol='rawPrediction', metricName='areaUnderROC')
    roc = evaluator_roc.evaluate(df)
    evaluator_pr = BinaryClassificationEvaluator(labelCol=target_variable_name, rawPredictionCol='rawPrediction', metricName='areaUnderPR')
    pr = evaluator_pr.evaluate(df)
    return cm, acc, misclassification_rate, precision, recall, f1, roc, pr



# Generate ROC chart
from sklearn import metrics
def draw_roc_plot(mdl_ltrl, y_score, y_true, model_type, data_type):

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title(str(model_type) + ' Model - ROC for ' + str(data_type) + ' data' )
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc = 'lower right')
    print( str(model_type) + '_' + str(model_type) + ' Model - ROC for ' + str(data_type) + ' data.png')
    plt.savefig( 'image/' +' Model - ROC for ' + str(data_type) + ' data.png', bbox_inches='tight')
    plt.close()


def draw_ks_plot(mdl_ltrl, model_type):

    writer = ExcelWriter(str(model_type) + '_KS_Charts.xlsx')


    writer.save()

def draw_confusion_matrix(mdl_ltrl, y_pred, y_true, model_type, data_type):

        AccuracyValue = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
        PrecisionValue = metrics.precision_score(y_pred=y_pred, y_true=y_true)
        RecallValue = metrics.recall_score(y_pred=y_pred, y_true=y_true)
        F1Value = metrics.f1_score(y_pred=y_pred, y_true=y_true)

        plt.title(str(model_type) + ' Model - Confusion Matrix for ' + str(
            data_type) + ' data \n \n Accuracy:{0:.3f}   Precision:{1:.3f}   Recall:{2:.3f}   F1 Score:{3:.3f}\n'.format(
            AccuracyValue, PrecisionValue, RecallValue, F1Value))
        cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        sns.heatmap(cm, annot=True, fmt='g');  # annot=True to annotate cells
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        print( str(  model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data.png')
        plt.savefig( 'image/' + mdl_ltrl +  str(model_type) +  ' Model - Confusion Matrix for ' + str(data_type) + ' data.png', bbox_inches='tight')
        plt.close()



# Model validation

def model_validation(mdl_ltrl, data, y, model, model_type, data_type):

        start_time = time.time()
        pred_data = model.transform(data)
        print('model output predicted')
        print(pred_data)
        roc_data, accuracy_data, ks_data, y_score, y_pred, y_true, decile_table = calculate_metrics(pred_data, y, data_type)
        draw_roc_plot(mdl_ltrl, y_score, y_true, model_type, data_type)
        decile_table.to_excel( str( model_type) + ' Model ' + str(data_type) + '.xlsx', index=False)
        draw_confusion_matrix(mdl_ltrl, y_pred, y_true, model_type, data_type)
        print('Metrics computed')

        l = [roc_data, accuracy_data, ks_data]
        end_time = time.time()
        print("Model validation process completed in :  %s seconds" % (end_time - start_time))
        return l

def make_confusion_matrix_chart(cf_matrix_train, cf_matrix_test):
        list_values = ['0', '1']

        plt.figure(1, figsize=(10, 5))
        plt.subplot(121)
        sns.heatmap(cf_matrix_train, annot=True, yticklabels=list_values,
                    xticklabels=list_values, fmt='g')
        plt.ylabel("Actual")
        plt.xlabel("Pred")
        plt.ylim([0, len(list_values)])
        plt.title('Train data predictions')

        plt.subplot(122)
        sns.heatmap(cf_matrix_test, annot=True, yticklabels=list_values,
                    xticklabels=list_values, fmt='g')
        plt.ylabel("Actual")
        plt.xlabel("Pred")
        plt.ylim([0, len(list_values)])
        plt.title('Test data predictions')

        plt.tight_layout()
        plt.savefig( 'image/make_confusion_matrix_chart data.png',bbox_inches='tight')
        plt.close()
        return None

from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        results_collect = rdd.collect()
        for row in results_collect:
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)


def plot_roc_pr(df, target_variable_name, plot_type, legend_value, title):
    preds = df.select(target_variable_name, 'probability')
    preds = preds.rdd.map(lambda row: (float(row['probability'][1]), float(row[target_variable_name])))
    # Returns as a list (false positive rate, true positive rate)
    points = CurveMetrics(preds).get_curve(plot_type)
    plt.figure()
    x_val = [x[0] for x in points]
    y_val = [x[1] for x in points]
    plt.title(title)

    if plot_type == 'roc':
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.plot(x_val, y_val, label='AUC = %0.2f' % legend_value)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')

    if plot_type == 'pr':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(x_val, y_val, label='Average Precision = %0.2f' % legend_value)
        plt.plot([0, 1], [0.5, 0.5], color='red', linestyle='--')

    plt.legend(loc='lower right')
    plt.savefig('image/' + 'plot_roc_' + str(title) + '_.png', bbox_inches='tight')
    plt.close()

    return None


def logistic_model(train, x, y):
    lr = LogisticRegression(featuresCol = x, labelCol = y, maxIter = 10)
    lrModel = lr.fit(train)
    return lrModel
