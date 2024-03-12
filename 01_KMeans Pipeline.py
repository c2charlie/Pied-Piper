# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:06:49 2024

@author: Lili Zheng
"""

####################################################################################
###          Import the modules needed for all the code in this section          ###
####################################################################################

import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


####################################################################################
###                     Create the data sets for the model                       ###
####################################################################################

# Download and extract the TCGA dataset from UCI
uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
# Build the url
full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)
# Download the file
r = urllib.request.urlretrieve(full_download_url, archive_name)
# Extract the data from the archive
tar = tarfile.open(archive_name, "r:gz")
tar.extractall()
tar.close()

# Load the data from the text file into memory as NumPy arrays
datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

data = np.genfromtxt(
    datafile,
    delimiter=",",
    usecols=range(1, 20532),
    skip_header=1
    )

true_label_names = np.genfromtxt(
    labels_file,
    delimiter=",",
    usecols=(1,),
    skip_header=1,
    dtype="str"
    )

# Convert the abbreviations to integers with LabelEncoder
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_label_names)

n_clusters = len(label_encoder.classes_)


####################################################################################
###                              Build the pipeline                              ###
####################################################################################

# Data to undergo multiple sequences of transformation, as scaling and dimensionality reduction
# Principle Component Analysis (PCA) is one of many dimensionality reduction techniques
preprocessor = Pipeline(
    [
     ("scaler", MinMaxScaler()), # when you do not assume that the shape of all your features follows a normal distribution
     ('pca', PCA(n_components = 2, random_state = 42)),
    ])


# Build the k-means clustering pipeline with user-defined arguments in the KMeans constructor
clusterer = Pipeline(
    [
     ("kmeans",
      KMeans(
          n_clusters = n_clusters,
          init = "k-means++",
          n_init = 50,
          max_iter = 500,
          random_state = 42,
          ),
      )
     ])

# Build an end-to-end k-means clustering pipleline by passing the "preprocessor" and "clusterer" pipelines to Pipeline
pipe = Pipeline(
    [
     ("preprocessor", preprocessor),
     ("clusterer", clusterer)
     ])

pipe.fit(data)

####################################################################################
###                          Evaluate the performance                            ###
####################################################################################

preprocessed_data = pipe['preprocessor'].transform(data)
predicted_labels = pipe["clusterer"]["kmeans"].labels_
silhouette_score(preprocessed_data, predicted_labels)
adjusted_rand_score(true_labels, predicted_labels)


####################################################################################
###                                   PLOT                                       ###
####################################################################################

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns = ["component_1", "component_2"],
    )

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
     x = "component_1",
     y = "component_2",
     s = 50,
     data = pcadf,
     hue = "predicted_cluster",
     style = "true_label",
     palette = "Set2",
     )

scat.set_title(
    "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
    )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()