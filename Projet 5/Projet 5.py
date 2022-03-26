# ******************************************
#
# Project : Customer Segmentation
# Project 5
# http://localhost:8888/notebooks/CustomerSegmentation.ipynb
# *******************************************


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_excel('/data/2020AnnualSales.xlsx', sheet_name='Sales')

# 2.
print(df.shape)
print(df.head())

# 2. Data Clean-Up
df.loc[df['SBQQ__Quantity__c'] <= 0].shape
print(df.shape)

#### - Total Sales
df['Sales'] = df['SBQQ__Quantity__c'] * df['SBQQ__NetPrice__c']

print(df.head())

#### - Per Customer Data

customer_df = df.groupby('SBQQ__Account__c').agg({
    'Sales': sum,
    'Name': lambda x: x.nunique()
})
#### - Total Sales
customer_df.columns = ['TotalSales', 'OrderCount']
customer_df['AvgOrderValue'] = customer_df['TotalSales']/customer_df['OrderCount']

print(customer_df.head(15))

print(customer_df.describe())

#### - Ranking
rank_df = customer_df.rank(method='first')
print(rank_df.head(15))


#### - Normalize
normalized_df = (rank_df - rank_df.mean()) / rank_df.std()


print(normalized_df.head(15))
print(normalized_df.describe())

# 3. Customer Segmentation via K-Means Clustering

from sklearn.cluster import KMeans

#### - K-Means Clustering

kmeans = KMeans(n_clusters=4).fit(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])

print(kmeans)

kmeans.labels_

kmeans.cluster_centers_

four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = kmeans.labels_
four_cluster_df.head()

four_cluster_df.groupby('Cluster').count()['TotalSales']

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['TotalSales'],
    c='blue'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['TotalSales'],
    c='red'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['TotalSales'],
    c='orange'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['TotalSales'],
    c='green'
)

plt.title('TotalSales vs. OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Total Sales')

plt.grid()
plt.show()


plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['AvgOrderValue'],
    c='blue'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['AvgOrderValue'],
    c='red'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['AvgOrderValue'],
    c='orange'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['OrderCount'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['AvgOrderValue'],
    c='green'
)

plt.title('AvgOrderValue vs. OrderCount Clusters')
plt.xlabel('Order Count')
plt.ylabel('Avg Order Value')

plt.grid()
plt.show()


plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['TotalSales'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['AvgOrderValue'],
    c='blue'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['TotalSales'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['AvgOrderValue'],
    c='red'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['TotalSales'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['AvgOrderValue'],
    c='orange'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['TotalSales'],
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['AvgOrderValue'],
    c='green'
)

plt.title('AvgOrderValue vs. TotalSales Clusters')
plt.xlabel('Total Sales')
plt.ylabel('Avg Order Value')

plt.grid()
plt.show()


#### - Selecting the best number of clusters

from sklearn.metrics import silhouette_score

for n_cluster in [4, 5, 6, 7, 8]:
    kmeans = KMeans(n_clusters=n_cluster).fit(
        normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']]
    )
    silhouette_avg = silhouette_score(
        normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']],
        kmeans.labels_
    )

    print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))


#### - Interpreting Customer Segments

kmeans = KMeans(n_clusters=4).fit(
    normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']]
)

four_cluster_df = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']].copy(deep=True)
four_cluster_df['Cluster'] = kmeans.labels_

four_cluster_df.head(15)

kmeans.cluster_centers_
high_value_cluster = four_cluster_df.loc[four_cluster_df['Cluster'] == 2]
high_value_cluster.head()
customer_df.loc[high_value_cluster.index].describe()
pd.DataFrame(
    df.loc[
        df['SBQQ__Account__c'].isin(high_value_cluster.index)
    ].groupby('Description__c').count()[
        'SBQQ__ProductName__c'
    ].sort_values(ascending=False).head()
)


# 3 D Printing

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
clusts = KMeans(n_clusters=4).fit_predict(normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']])
#Plot the clusters obtained using k means
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
            kmeans.cluster_centers_[:, 1],
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 2],
            s = 250,
            marker='o',
            c='red',
            label='centroids')
scatter = ax.scatter(four_cluster_df['TotalSales'],four_cluster_df['OrderCount'], four_cluster_df['AvgOrderValue'],
                     c=clusts, s=20, cmap='winter')


ax.set_title('K-Means Clustering')
ax.set_xlabel('TotalSales')
ax.set_ylabel('OrderCount')
ax.set_zlabel('AvgOrderValue')
ax.legend()
plt.show()


# Optimize Number of clusters for k-means clustering

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
data = normalized_df[['TotalSales', 'OrderCount', 'AvgOrderValue']]
mms.fit(data)
data_transformed = mms.transform(data)


Sum_of_squared_distances = []
K = range(1,14)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# SIlhouette Vizualizer
from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(data)


plt.show()
