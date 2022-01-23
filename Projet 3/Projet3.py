
###############################################################################
#
# Description : This program used for Project 3
# EDA Exploratory Data Analysis
# Univariate - Multi variate Analysis
# Author : Ali Naama
# Date : 06/11/2021
# Data Set from Kaggle
#
# https://medium.com/@rndayala/eda-on-haberman-data-set-c9ee6d51ab0a
#
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

haberman = pd.read_csv('D:\Projet Perso\Ali\Data Scientist\Projet 1\data\haberman.csv')

# head of the data
print(haberman.head())
column_names = ['Age', 'Year', 'Positive_Axillary_Nodes', 'Survival_Status']
haberman.columns = column_names

print(haberman.head())
print(haberman.shape)
print(haberman.info)
print(haberman.describe)

# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# plot pair-wise relationships in a dataset
# Pair Plots are a really simple way to visualize relationships between two variables.
# It produces a matrix of relationships between each variable in your data for an instant examination of our data.

#sns.pairplot(haberman)

# Note: Draw scatter plots for joint relationships and histograms for uni-variate distributions.

# check for missing values
print('# check for missing values')
print(haberman.isnull().sum())

# print the unique values of the class label column

print(list(haberman['Survival_Status'].unique()))

# modify the target column values to be meaningful as well as categorical
haberman['Survival_Status'] = haberman['Survival_Status'].map({1: "Yes", 2: "No"})
haberman['Survival_Status'] = haberman['Survival_Status'].astype('category')

print(haberman.head())

# check the structure again

print(haberman.info)

# (Q) How many data points for each class are present for Survival_Status column?

print(haberman["Survival_Status"].value_counts())

# plot the histogram for the classes - binary classification, only two classes

count_classes = pd.value_counts(haberman["Survival_Status"])
count_classes.plot(kind = 'bar')
plt.title("Class distribution Histogram")
plt.xlabel("Survival Status")
plt.ylabel("Frequency")
#plt.show()

# percentage of classes
# this gives us the distribution of classes in the data set
print('% of distribution for Survival_Status')
print(haberman["Survival_Status"].value_counts(1).mul(100).round(1).astype(str) + '%')

# (Q) High Level Statistics
# Since Survival_Status is a categorical variable, this column will not be shown here.
# By default, describe() function displays descriptive statistics for numerical columns only
#
print(haberman.describe())

# include descriptive statistics for categorical columns also

print(haberman.describe(include='all'))

# descriptive statistics only for categorical variable
#
print(haberman['Survival_Status'].describe())

print('# Describe : ')
print(haberman.describe())

# Univariate analysis - plotting distribution


#5.3 Box Plots
"""
Box plot takes a less space and visually represents the five number summary of the data points in a box. 
The outliers are displayed as points outside the box.
1. Q1 - 1.5*IQR
2. Q1 (25th percentile)
3. Q2 (50th percentile or median)
4. Q3 (75th percentile)
5. Q3 + 1.5*IQR
Inter Quartile Range = Q3 -Q1
"""


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot( x='Survival_Status', y=feature, data=haberman, ax=axes[idx])
plt.show()



#5.4 Violin Plots
"""
#Violin plot is the combination of box plot and probability density function.
"""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot( x='Survival_Status', y=feature, data=haberman, ax=axes[idx])
plt.show()


#5.1 Distribution plots
"""
#* Distribution plots are used to visually assess how the data points are distributed with respect to its frequency.
#* Usually the data points are grouped into bins and the height of the bars representing each group increases
#with increase in the number of data points
#lie within that group. (histogram)
#* Probality Density Function (PDF) is the probabilty that the variable takes a value x. (smoothed version of the histogram)
#* Kernel Density Estimate (KDE) is the way to estimate the PDF. The area under the KDE curve is 1.
#* Here the height of the bar denotes the percentage of data points under the corresponding group
"""
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    fg = sns.FacetGrid(haberman, hue='Survival_Status', size=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()


#5.2 CDF
"""
#The cumulative distribution function (cdf) is the probability that the variable takes a value less than or equal to x.
"""
plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    plt.subplot(1, 3, idx+1)
    print("********* "+feature+" *********")
    counts, bin_edges = np.histogram(haberman[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)




sns.FacetGrid(haberman, hue="Survival_Status", height=5) \
      .map(sns.distplot, "Age") \
      .add_legend();
plt.show();

# plotting distribution plot for all features
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    fg = sns.FacetGrid(haberman, hue='Survival_Status', size=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()

    # CDF - The cumulative distribution function (cdf) is the probability that
    # the variable takes a value less than or equal to x.

plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    plt.subplot(1, 3, idx + 1)
    print("********* " + feature + " *********")

    counts, bin_edges = np.histogram(haberman[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts / sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)


survived = haberman[haberman['Survival_Status'] == 'Yes']
notsurvived = haberman[haberman['Survival_Status'] == 'No']
plt.figure(figsize=(20, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    plt.subplot(1, 3, idx + 1)
    print("********* " + feature + " *********")
# PDF & CDF for Survived class
    counts, bin_edges =     np.histogram(survived[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts / sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)

# PDF & CDF for not Survived class
    counts, bin_edges = np.histogram(notsurvived[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts / sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)

"""
# Multi Variate

# pair plot
"""
#Pair plot in seaborn plots the scatter plot between every two data columns in a given dataframe.
#It is used to visualize the relationship between two variables

sns.pairplot(haberman, hue='Survival_Status', size=3)
plt.show()





# Multi-variate Analysis

# 2-D Scatter plots are used to visualize relationship between two variables only
# Here 'sns' corresponds to seaborn.# 2-D Scatter plot with color-coding for each Survival type/class.
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=3) \
   .map(plt.scatter, "Age", "Positive_Axillary_Nodes") \
   .add_legend();
plt.show();

# after we have made the categorical variable 'Survival_Status' as of type 'category',
# the default sns pairplot won't show that feature now.
#
sns.pairplot(haberman)

sns.pairplot(haberman, hue = 'Survival_Status', size = 3)
plt.show()

print(list(haberman['Survival_Status'].unique()))

from sklearn.preprocessing import LabelEncoder
# Let’s convert the categorical column to numerical using Label Encoder
le = LabelEncoder()
le.fit(haberman['Survival_Status'])
haberman['Survival_Status'] = le.transform(haberman['Survival_Status'])# check the structure of data
haberman.info()

# print the unique values of the class label column
print(list(haberman['Survival_Status'].unique()))

le.classes_

#(Q) How many data points for each class are present for Survival_Status column?
haberman["Survival_Status"].value_counts()

# generate correlation matrix
#print(haberman.corr()) # for this to work, all columns should be numerical



print(haberman.corr()['Survival_Status']) # numerical correlation matrix


# look at the heatmap of the correlation matrix of our dataset
sns.set(font_scale=1.4)
swarm_plot = sns.heatmap(haberman.corr(), annot = True, cmap='RdYlGn')

fig = swarm_plot.get_figure()
fig.set_figwidth(15)
fig.set_figheight(15)
fig.savefig('saving-a-high-resolution-seaborn-plot.png', dpi=300)


# look at the heatmap of the correlation matrix of our dataset
sns.heatmap(haberman.corr(), annot = True)

print(haberman.corr()['Survival_Status']) # numerical correlation matrix
