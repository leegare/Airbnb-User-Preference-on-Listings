import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib as mpl
import seaborn as sns

from bisect import bisect

from scipy import linalg
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn import mixture
from sklearn.neighbors import LocalOutlierFactor


import warnings
warnings.filterwarnings('ignore')

pd.set_option('precision', 3)

'''INDEX
get_district_cluster
    - arrondissement
    - data
    - epsilon=False
    - max_pts=False
Plots data for zipcode = arrondissement and analyzes it to retrieve the
parameters required to run a DBSCANself.

'''

'''----------------Global Variables--------------------------'''

usecols = ['listing_id','host_id', 'host_since', 'host_response_time','host_is_superhost',
       'zipcode', 'latitude', 'longitude',
       'property_type', 'room_type', 'accommodates','bathrooms', 'bed_type',
       'amenities', 'amty_per','price',
       'guests_included', 'minimum_nights','availability_365',
       'first_review', 'last_review', 'review_scores_rating','review_scores_accuracy',
       'review_scores_cleanliness','review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'number_of_reviews',
       'reviews_per_month']

int_data = '../data/processed/data.csv'
data = pd.read_csv(int_data, index_col=0, parse_dates=['host_since','first_review','last_review'], usecols=usecols)

int_data = '../data/raw/paris_reviews.csv'
reviews_data = pd.read_csv(int_data, parse_dates=['date'], usecols=['listing_id','date','id'])

paris_attractions = pd.read_csv('../data/processed/paris_attractions.csv', index_col=0)



'''---------------------FUNCTIONS---------------------'''


'''---------------------FUNCTIONS RELATED TO CLUSTERING---------------------
--------------------- IN COMPRESSING THE LISTINGS IN THEIR ARRONDISSEMENT '''


def print_arrondissement(arrondissement):
    matXY = data.loc[data.zipcode==arrondissement, ['longitude','latitude']].as_matrix()
    fig = plt.figure(figsize=(10,5))
    plt.scatter(matXY[:,0], matXY[:,1], s=50, c='royalblue', alpha=0.9)
    plt.show()

# ----------------------------
# FUNCTION FOR LOF
def get_lof(df, k, p=False):
    # Fit the model
    lof = LocalOutlierFactor(n_neighbors=k)
    X = df.values
    lof.fit(X)
    outlier_scores = lof.negative_outlier_factor_

    # Get outliers whose score is less than -2.... but WHY?
    idx_outliers = [idx for idx, score in enumerate(outlier_scores) if score<-2]
    outliers = df.iloc[idx_outliers,:]

    # Get the inliers
    inliers = df.loc[~df.index.isin(outliers.index.values)]

    # Plot
    if p:
        #Plot inliers as blue circles, and outliers as large X's
        fig = plt.figure(figsize=(15, 7))
        # plot the data observations
        plt.scatter(inliers['longitude'].values, inliers['latitude'].values, c='royalblue', s=50, alpha=0.6)

        # plot the outliers
        plt.scatter(outliers.longitude.values, outliers.latitude.values, c='black', s=50)
        plt.title("Outliers using LOF")
        plt.ylabel("Latitude")
        plt.xlabel("Longitude")
        #plt.ylim(ymax = 115, ymin = -10)
        #plt.xlim(xmax = 35, xmin = -5)
        plt.show()

    return outliers, inliers



# ----------------------------
# FUNCTION USING DBSCAN TO
# CLUSTER DISTRICTS INDIVIDUALLY AND POINT OUT OUTLIERS
def get_district_cluster(arrondissement, data, epsilon=False, max_pts=False):
    matXY = data.loc[(data.zipcode==arrondissement),['longitude','latitude']].as_matrix()
    n_data = matXY.shape[0]

    fig = plt.figure(figsize=(15,5))

    # Pairwise distances in the dataset
    pairDistList = pdist(matXY)

    # Get Density
    ax2 = fig.add_subplot(2,2,2)
    n, b, patches = ax2.hist(pairDistList, bins=100, color='royalblue')
    bin_max = np.where(n == n.max())
    # Plot the maximum value of the histogram
    ax2.axvline(x=b[bin_max][0], linestyle='--', color='gold', linewidth=4)
    # Anotation
    ax2.text(b[bin_max][0]*2,n[bin_max][0]*0.7,r'~Cluster Scale:'+str(b[bin_max][0]), size=13)
    #ax2.xlabel('Pairwise Distance')
    #ax2.ylabel('# of data points')
    #ax2.title('Dataset Separation Distribution')
    if isinstance(epsilon, bool):
        epsilon = b[bin_max][0]


    #Find pairs of data points that are closer than epsilon
    nn = neighborhoods(matXY, epsilon)
    # Get nearest neighbors
    ax3 = fig.add_subplot(2,2,4)
    n, b, patches = ax3.hist(nn, bins=np.arange(min(nn), max(nn) + 1, 1), color='royalblue')
    bin_max = np.where(n == n.max())
    ax3.axvline(x=b[bin_max][0], linestyle='--', color='gold', linewidth=4)
    ax3.text(b[bin_max][0], n[bin_max][0]*0.7, 'maxPts='+str(b[bin_max][0]), size=16, color='dimgray')
    if isinstance(max_pts, bool):
        max_pts = b[bin_max][0]

    # DBSCAN
    dbmodel = DBSCAN(eps=epsilon, min_samples=max_pts).fit(matXY)
    labels = dbmodel.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # ClusterDict has the point name (or index) as value and it's label as key.
    clusterDict = buildClusterDict(dbmodel)
    noise_sample_mask = np.zeros(n_data, dtype=bool)
    noise_sample_mask[clusterDict[-1]] = True
    # Representation in 2D
    ax4 = fig.add_subplot(1,2,1)
    ax4.scatter(matXY[:,0], matXY[:,1], c=dbmodel.labels_)
    ax4.scatter(matXY[noise_sample_mask, 0], matXY[noise_sample_mask, 1], c='k', label="noise")

    plt.show()
