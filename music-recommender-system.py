#!/usr/bin/env python
# coding: utf-8

# In[187]:


import pandas as pd
import numpy as np


# In[188]:


spotify_df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv")


# In[157]:


spotify_df


# In[189]:


features = ["instrumentalness", "tempo", "energy", "key", "valence"]


# In[190]:


spotify_df = spotify_df.dropna(subset=features)


# In[191]:


data = spotify_df[features].copy()


# In[192]:


data


# In[193]:


#1. Scale the Data
#2. Initialize random centroids
#3. Label each data point
#4. Update our centroids
#5. Repeat until centriods stop changing


# In[194]:


#Using min-max scaling, we will set every data point from 0-10
#This just makes the data easier to work with
data = ((data - data.min()) / (data.max() - data.min())) * 10 + 1


# In[195]:


data.describe()


# In[196]:


#Gives us a glimpse at the first 5 rows of data
data.head()


# In[197]:


#Initialize our centroids
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis = 1)
# x.apply goes through each column in data, and we collect x.sample, which is a random value
# We convert it into a float to make it a number
# The return statement combines the centroids into a data frame


# In[198]:


centroids = random_centroids(data, 5)


# In[199]:


centroids


# In[200]:


#Label every data pont


# In[201]:


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


# In[202]:


labels = get_labels(data, centroids)


# In[203]:


labels


# In[204]:


labels.value_counts()


# In[205]:


#Find geometric mean, which is the mean of points, to find center of points.


# In[206]:


def new_centroids(data, labels, k):
    data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids
    #First, we split our data by cluster(labels). To each group, we calculate the geometric mean of each feature


# In[207]:


from sklearn.decomposition import PCA #Principle Components Analysis, helps us visualize. Take 5D to 2D, helps graph
import matplotlib.pyplot as plt #Plotting library
from IPython.display import clear_output # Clears the output, adds a new graph


# In[208]:


def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()


# In[209]:


max_iterations = 100
centroid_count = 10

centroids = random_centroids(data, centroid_count)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    
    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, centroid_count)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1


# In[210]:


centroids


# In[211]:


spotify_df[labels == 8][["track_name"] + features]


# In[212]:


from sklearn.cluster import KMeans


# In[213]:


kmeans = KMeans(10)
kmeans.fit(data)


# In[214]:


centroids = kmeans.cluster_centers_


# In[215]:


pd.DataFrame(centroids, columns=features).T

