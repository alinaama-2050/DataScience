###############################################################################
#
# Description : Ce programme utilisé pour le projet 3
# Analyse exploratoire des données EDA
# Univariée - Analyse multivariée
# Jeu de données : Patients cancéreux
# Author : @rndayala/Used/Translate/Updated By Ali Naama
# Date : 07/11/2021
# Data Set from Kaggle
# From : 
# https://medium.com/@rndayala/eda-on-haberman-data-set-c9ee6d51ab0a
# https://www.kaggle.com/gokulkarthik/haberman-s-survival-exploratory-data-analysis
# https://www.kaggle.com/ashitaggarwal1986/habermans-survival-data-eda
# 
###############################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 1. lecture du fichier csv haberman

haberman = pd.read_csv('Data/haberman.csv')


# 1.1 Exploration des données / Vision du dataframe - premiers enregistrements
print(haberman.head())
# Ajout de ccolonnes d'entête qui sont manquantes
column_names = ['Age', 'Year', 'Positive_Axillary_Nodes', 'Survival_Status']
haberman.columns = column_names

# Résumé des données
print(haberman.head())
# Dimenssion de la matrice de données
print(haberman.shape)
# Résumé des données
print(haberman.info)
# Statistique synthètique des données par colonne
print(haberman.describe)


# Vérifier les valeurs manquantes
print('# vérifier les valeurs manquantes')
print(haberman.isnull().sum())

# Affiche les valeurs uniques de la colonne d'étiquette de classe

print(list(haberman['Survival_Status'].unique()))

# Modifier les valeurs de la colonne cible pour qu'elles soient significatives et catégorielles
haberman['Survival_Status'] = haberman['Survival_Status'].map({1: "Yes", 2: "No"})
haberman['Survival_Status'] = haberman['Survival_Status'].astype('category')

print(haberman.head())

# Revérifier la structure du jeu de données

print(haberman.info)

# Combien de points de données pour chaque classe sont présents pour la colonne Survival_Status ?

print(haberman["Survival_Status"].value_counts())


# Utilisation de librairie seaborn
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# Tracer des relations par paires dans un ensemble de données
# Les graphiques de paires sont un moyen très simple de visualiser les relations entre deux variables.
# Il produit une matrice de relations entre chaque variable de vos données pour 
# un examen rapide de nos données.

sns.pairplot(haberman)

# Tracer l'histogramme des classes - classification binaire, seulement deux classes

count_classes = pd.value_counts(haberman["Survival_Status"])
count_classes.plot(kind = 'bar')
plt.title("Class distribution Histogram")
plt.xlabel("Survival Status")
plt.ylabel("Frequency")
plt.show()

# Pourcentage de distribution
# cela nous donne la distribution des classes dans l'ensemble de données en %
print('% of distribution for Survival_Status')
print(haberman["Survival_Status"].value_counts(1).mul(100).round(1).astype(str) + '%')

# (Q) Statistiques de haut niveau
# Étant donné que Survival_Status est une variable catégorielle, cette colonne ne sera pas affichée ici.
# Par défaut, la fonction describe() affiche des statistiques descriptives pour les colonnes numériques uniquement

print(haberman.describe())

# include descriptive statistics for categorical columns also

print(haberman.describe(include='all'))

# descriptive statistics only for categorical variable
#
print(haberman['Survival_Status'].describe())

print('# Describe : ')
print(haberman.describe())

# 1.2 Univariate analysis - plotting distribution


# Box Plots

# La boîte à moustaches prend moins de place et représente visuellement le résumé à cinq chiffres des points de données dans une boîte.
# Les valeurs aberrantes sont affichées sous forme de points à l'extérieur de la boîte.
# 1. Q1 - 1.5 * IQR
# 2. T1 (25e centile)
# 3. Q2 (50e centile ou médiane)
# 4. Q3 (75e centile)
# 5. Q3 + 1.5 * IQR
# Intervalle interquartile = Q3 -Q1



fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot( x='Survival_Status', y=feature, data=haberman, ax=axes[idx])
plt.show()


# Violin plot est la combinaison d'une boîte à moustaches et d'une fonction de densité de probabilité.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot( x='Survival_Status', y=feature, data=haberman, ax=axes[idx])
plt.show()


# Densité de distribution
# Les diagrammes de distribution sont utilisés pour évaluer visuellement la distribution des points de données par rapport à sa fréquence.
# Habituellement, les points de données sont regroupés dans des bacs et la hauteur des barres représentant chaque groupe augmente
# avec augmentation du nombre de points de données
# se trouvent dans ce groupe. (histogramme)
# La fonction de densité de probabilité (PDF) est la probabilité que la variable prenne une valeur x. (version lissée de l'histogramme)
# Kernel Density Estimate (KDE) est le moyen d'estimer le PDF. L'aire sous la courbe KDE est 1.
# Ici, la hauteur de la barre indique le pourcentage de points de données sous le groupe correspondant


for idx, feature in enumerate(list(haberman.columns)[:-1]):
    fg = sns.FacetGrid(haberman, hue='Survival_Status', size=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()


# CDF

# La fonction de distribution cumulative (cdf) est la probabilité que la variable 
# prenne une valeur inférieure ou égale à x.


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


# 2. Analyse multivariée


# Pair plot in seaborn trace le nuage de points entre toutes les deux colonnes de données dans un dataframe donné.
# Il est utilisé pour visualiser la relation entre deux variables

sns.pairplot(haberman, hue='Survival_Status', size=3)
plt.show()


# Les diagrammes de dispersion 2D sont utilisés pour visualiser la relation entre deux 
# variables uniquement
# Ici 'sns' correspond à seaborn.# Diagramme de dispersion 2-D avec code couleur 
# pour chaque type/classe de survie.



sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=3) \
   .map(plt.scatter, "Age", "Positive_Axillary_Nodes") \
   .add_legend();
plt.show();

# après avoir créé la variable catégorielle 'Survival_Status' en tant que type 'category',
# le pairplot sns par défaut n'affichera pas cette fonctionnalité maintenant.

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

# Combien de points de données pour chaque classe sont présents pour la colonne Survival_Status ?

haberman["Survival_Status"].value_counts()

# Générer une matrice de corrélation

print(haberman.corr()) # for this to work, all columns should be numerical

print(haberman.corr()['Survival_Status']) # numerical correlation matrix


# Visualiser la carte thermique de la matrice de corrélation de notre jeu de données

sns.set(font_scale=1.4)
swarm_plot = sns.heatmap(haberman.corr(), annot = True, cmap='RdYlGn')

fig = swarm_plot.get_figure()
fig.set_figwidth(15)
fig.set_figheight(15)
fig.savefig('saving-a-high-resolution-seaborn-plot.png', dpi=300)


sns.heatmap(haberman.corr(), annot = True)

# Matrice de corrélation numérique

print(haberman.corr()['Survival_Status']) 