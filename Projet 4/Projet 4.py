# # Projet 4 - Analyse  de données de Consommation Electrique
# Analyze Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("whitegrid")
sns.color_palette("crest", as_cmap=True)
import warnings
warnings.filterwarnings('ignore')

import os

data = pd.read_csv("D:/Projet Perso/Ali/Data Scientist/Projet 4/data/building-energy-cleaned.csv")
print(data.head())

print(data.info())
print(data.describe())

print(data[(data.GFAPerBuilding == np.inf) | (data.GFAPerFloor == np.inf)].head())


data['GFAPerBuilding'] = np.where(((data.GFAPerBuilding == np.inf) & (data.NumberofBuildings == 0)),0, data.GFAPerBuilding)
data['GFAPerFloor'] = np.where(((data.GFAPerFloor == np.inf) & (data.NumberofFloors == 0)),0, data.GFAPerFloor)

font_title = {'family': 'serif',
              'color':  '#1d479b',
              'weight': 'bold',
              'size': 18,
             }

fig = plt.figure(figsize=(12,8))
sns.scatterplot(data = data, x='PropertyGFATotal', y='SiteEnergyUse(kBtu)', hue='BuildingType')
plt.title(f"Consommations d'énergie par surface totale au sol et par type de bâtiment\n",
          fontdict=font_title, fontsize=16)
plt.show()

print(data[data['SiteEnergyUse(kBtu)']>8*10**8])


identification_features = ['OSEBuildingID', 'PropertyName', 'Address', 'ZipCode']
data_identification = data[identification_features]
data.drop(identification_features, axis=1, inplace = True)


data_filter = data.drop(['SteamUse(kBtu)','Electricity(kBtu)','NaturalGas(kBtu)'], axis=1)

numerical_features = data_filter.select_dtypes(include=['int64','float64'])
categorical_features = data_filter.select_dtypes(exclude=['int64','float64'])

categorical_features.nunique()
categorical_features = categorical_features.drop(['State','YearsENERGYSTARCertified'], axis=1)
print(list(numerical_features.columns))


energystar_score = numerical_features['ENERGYSTARScore']
numerical_features = numerical_features.drop(['ENERGYSTARScore','DataYear'], axis=1)

data_filter = pd.concat([categorical_features, numerical_features], axis=1)

# Les données numériques doivent être standardisées pour entrer dans nos modèles de prédiction.
# Nous réaliserons un centrage-réduction via la méthode StandardScaler de Scikit-Learn.
#


# Préparation du Preprocessor

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer

target_features = ['BuildingType','PrimaryPropertyType','Neighborhood','LargestPropertyUseType']
target_transformer = TargetEncoder()

numeric_features = ['harvesine_distance','NumberofBuildings','NumberofFloors',
                    'PropertyGFATotal','BuildingAge','TotalUseTypeNumber',
                    'GFABuildingRate','GFAParkingRate','GFAPerBuilding','GFAPerFloor']
numeric_transformer = RobustScaler(unit_variance=True)

preprocessor = ColumnTransformer(transformers=[
    ('target', target_transformer, target_features),
    ('numeric', numeric_transformer, numeric_features)
])



from sklearn.model_selection import train_test_split
#data_filter = data_filter[data_filter.loc[:]!=0].dropna()

X = data_filter.drop(['TotalGHGEmissions','SiteEnergyUse(kBtu)'], axis=1)
X = X.select_dtypes(include=['float64'])
Y = data_filter[['TotalGHGEmissions','SiteEnergyUse(kBtu)']]

# Verifying that all data are in expected type
# Last cleaning on Nan values - preventing infinite errors


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Entrainement: {} lignes,\nTest: {} lignes.\n".format(X_train.shape[0],X_test.shape[0]))


print(Y_train)
print(X_train)



from sklearn.preprocessing import FunctionTransformer

logtransformer = FunctionTransformer(np.log, inverse_func = np.exp, check_inverse = True)
Y_log = logtransformer.transform(Y)

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(20,8))
sns.histplot(data=Y, x='TotalGHGEmissions', stat="density", ax=axes[0])
axes[0].set_title("Données initiales", color='#2cb7b0')
sns.histplot(data=Y_log, x='TotalGHGEmissions', stat="density", ax=axes[1])
axes[1].set_title("Application du logarithme", color='#2cb7b0')
plt.suptitle("Distribution des emissions de CO2 avec changement d'échelle", fontsize=22)
plt.show()

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import set_config
set_config(display='diagram')


param_mlr = {"regressor__fit_intercept": [True, False],
             "regressor__normalize": [True, False]}

mlr_grid_cv = Pipeline([
    ('preprocessor', preprocessor),
    ('grid_search_mlr', GridSearchCV(
                            TransformedTargetRegressor(
                                regressor=LinearRegression(),
                                func=np.log,
                                inverse_func=np.exp),
                            param_grid=param_mlr,
                            cv=5,
                            scoring=('r2','neg_mean_absolute_error'),
                            return_train_score = True,
                            refit='neg_mean_absolute_error',
                            n_jobs = -1))])



#Retour des meilleurs scores NMAE et R2
#Stockage du dataframe de resultats du modèle
def model_scores(pip,step):
    df_results = pd.DataFrame.from_dict(pip.named_steps[step].cv_results_) \
                    .sort_values('rank_test_neg_mean_absolute_error')
    best_nmae = pip.named_steps[step].best_score_
    best_r2 = np.mean(df_results[df_results.rank_test_r2 == 1]['mean_test_r2'])
    best_params = pip.named_steps[step].best_params_
    training_time = round((np.mean(df_results.mean_fit_time)*X_train.shape[0]),2)
    print("Meilleur score MAE : {}\nMeilleur Score R2 : {}\nMeilleurs paramètres : {}\nTemps moyen d'entrainement : {}s"\
         .format(round(best_nmae,3), round(best_r2,3), best_params, training_time))
    return df_results

print(Y_train['TotalGHGEmissions'])

#Entrainement sur les 2 variables à expliquer :
print(X_train.describe())
print(Y_train.describe())



# GHG_mlr_model = mlr_grid_cv.fit(X_train, Y_train)

# fitting the training data

for col in X_train.columns:
    print(col)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# Verifying that all data are in expected type

#Y_train = Y_train.select_dtypes(include=['float64'])

#Y_train = clean_dataset(Y_train)
print(X_train.describe())

#Entrainement sur les 2 variables à expliquer :
#GHG_mlr_model = mlr_grid_cv.fit(X_train, Y_train['TotalGHGEmissions'])
#GHG_mlr_results = model_scores(GHG_mlr_model, 'grid_search_mlr')

import time
# 2. Modèle Linéaire
model_LR = LinearRegression()

X_train.to_csv (r'D:\Projet Perso\Ali\Data Scientist\Projet 4\data\export_X_traindataframe.csv', index = False, header=True)
Y_train.to_csv (r'D:\Projet Perso\Ali\Data Scientist\Projet 4\data\export_Y_traindataframe.csv', index = False, header=True)
model_LR.fit(X_train, Y_train['TotalGHGEmissions'])

# predict the target on train and test data
predict_train = model_LR.predict(X_train)
predict_test  = model_LR.predict(X_test)

# Root Mean Squared Error on train and test date
print('Linear Model :')
print('RMSE on train data: - LR: ', mean_squared_error(Y_train['TotalGHGEmissions'], predict_train)**(0.5))
print('RMSE on test data:  - LR: ',  mean_squared_error(Y_test['TotalGHGEmissions'], predict_test)**(0.5))
start_time = time.time()
LR_pred = model_LR.predict(X_test)
print("Temps d'execution de l'agorithme model_LR : {:.2} s.".format((time.time() - start_time)))
# The linear regression model has a very high RMSE value on both training and validation data.
# Let us see if a tree-based model performs better in this case. Here we will
# # train a random forest and check if we get any improvement in the train and validation errors.

# 3. Modèle ElasticNet
from sklearn.linear_model import ElasticNet
model_ELN = ElasticNet()
model_ELN.fit(X_train, Y_train['TotalGHGEmissions'])

# predict the target on train and test data
predict_train_eln = model_ELN.predict(X_train)
predict_test_eln  = model_ELN.predict(X_test)

# Root Mean Squared Error on train and test date
print('ElasticNet Model :')
print('RMSE on train data: - ELN:  ', mean_squared_error(Y_train['TotalGHGEmissions'], predict_train_eln)**(0.5))
print('RMSE on test data:  - ELN:  ',  mean_squared_error(Y_test['TotalGHGEmissions'], predict_test_eln)**(0.5))
start_time = time.time()
ELN_pred = model_ELN.predict(X_test)
print("Temps d'execution de l'agorithme model_ELN : {:.2} s.".format((time.time() - start_time)))

# 4. Modèle Support Vector Regression (SVR)
from sklearn.svm import LinearSVR
model_SVR = LinearSVR()
model_SVR.fit(X_train, Y_train['TotalGHGEmissions'])

# predict the target on train and test data
predict_train_svr = model_SVR.predict(X_train)
predict_test_svr = model_SVR.predict(X_test)

# Root Mean Squared Error on train and test date
print('Support Vector Regression Model :')
print('RMSE on train data: - SVR:  ', mean_squared_error(Y_train['TotalGHGEmissions'], predict_train_svr)**(0.5))
print('RMSE on test data:  - SVR:   ',  mean_squared_error(Y_test['TotalGHGEmissions'], predict_test_svr)**(0.5))
start_time = time.time()
SVR_pred = model_SVR.predict(X_test)
print("Temps d'execution de l'agorithme model_SVR : {:.2} s.".format((time.time() - start_time)))

# 5. Modèle non-linéaires : RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2, random_state=0)
model_RFS = regr.fit(X_train, Y_train['TotalGHGEmissions'])

# predict the target on train and test data
predict_train_rfs = model_RFS.predict(X_train)
predict_test_rfs = model_RFS.predict(X_test)

# Root Mean Squared Error on train and test date
print('Random Forest Model :')
print('RMSE on train data: - RFS: ', mean_squared_error(Y_train['TotalGHGEmissions'], predict_train_rfs)**(0.5))
print('RMSE on test data : - RFS: ',  mean_squared_error(Y_test['TotalGHGEmissions'], predict_test_rfs)**(0.5))
start_time = time.time()
RFM_pred = model_RFS.predict(X_test)
print("Temps d'execution de l'agorithme model_RFS : {:.2} s.".format((time.time() - start_time)))

# 6. Modèle non-linéaires : XGBoost
import xgboost as xgb
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
param = {
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    "objective": "reg:squarederror",
    'booster': 'gbtree',
    'n_jobs': 10,
    'validate_parameters':'True',
    'alpha': 0.2,
    'lambda': 0.001,
    'colsample_bylevel': 0.9,
    'gamma': 0.01,
    'max_delta_step': 0.1,
}
import xgboost as xgb
rnd_search = xgb.XGBRegressor(**param, n_estimators=999)

model_XGB = rnd_search.fit(X_train, Y_train['TotalGHGEmissions'])
# predict the target on train and test data
predict_train_xgb = model_XGB.predict(X_train)
predict_test_xgb = model_XGB.predict(X_test)
# Root Mean Squared Error on train and test date
print('XGBoost Model : ')
print('RMSE on train data: - XGB: ', mean_squared_error(Y_train['TotalGHGEmissions'], predict_train_xgb)**(0.5))
print('RMSE on test data : - XGB: ',  mean_squared_error(Y_test['TotalGHGEmissions'], predict_test_xgb)**(0.5))
start_time = time.time()
XGB_pred = model_XGB.predict(X_test)
print("Temps d'execution de l'agorithme model_XGB : {:.2} s.".format((time.time() - start_time)))

def plot_pred_true(y_true, y_pred, color=None, title=None):
    X_plot = [y_true.min(), y_true.max()]
    fig = plt.figure(figsize=(12,8))
    plt.scatter(y_true, y_pred, color=color, alpha=.6)
    plt.plot(X_plot, X_plot, color='r')
    plt.xlabel("Valeurs réélles")
    plt.ylabel("Valeurs prédites")
    plt.title("Valeurs prédites VS valeurs réélles | Variable {}".format(title),
              fontdict=font_title, fontsize=18)
    plt.show()

def metrics_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = y_true - y_pred
    mae = np.mean(abs(diff))
    r2 = 1-(sum(diff**2)/sum((y_true-np.mean(y_true))**2))
    dict_metrics = {"Métrique":["MAE", "R²"], "Résultats":[mae, r2]}
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics

#Calcul des métriques pour les émissions de CO2
SEUmetrics = metrics_model(Y_test['SiteEnergyUse(kBtu)'],RFM_pred)
print(SEUmetrics)

#Affichage des valeurs prédites vs valeurs réélles pour émissions de CO2
plot_pred_true(Y_test['TotalGHGEmissions'],RFM_pred, color="#9C3E2D", title="TotalGHGEmissions")


final_SEU_test = pd.concat([X_test,Y_test],axis=1)
final_SEU_test['SEU_pred'] = RFM_pred
print(final_SEU_test.keys())
compare_final_SEU_test = final_SEU_test = final_SEU_test.groupby(by='NumberofFloors').mean()

x = np.arange(len(compare_final_SEU_test.index))
width = 0.35

fig, ax = plt.subplots(figsize=(20,8), sharey=False, sharex=False)

scores1 = ax.bar(x - width/2, compare_final_SEU_test['SiteEnergyUse(kBtu)'], width, label='SiteEnergyUse(kBtu)')
scores2 = ax.bar(x + width/2, compare_final_SEU_test['SEU_pred'], width, label='Prédictions')
ax.set_ylabel('(kBtu)')
ax.set_xticks(x)
ax.set_xticklabels(compare_final_SEU_test.index)
ax.legend()
ax.bar_label(scores1, padding=3)
ax.bar_label(scores2, padding=3)

plt.suptitle("Ecarts de prédictions sur la variable SiteEnergyUse(kBtu) par Nombre d'étage", fontsize=22)
fig.tight_layout()

plt.show()
