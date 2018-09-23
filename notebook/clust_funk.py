# LIBS FOR EE
from sklearn.covariance import EllipticEnvelope
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

# LIBS FOR GMM
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture

# LIBS FOR KMeans
from sklearn.cluster import KMeans

# LIBS for DBSCAN
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import DBSCAN

# LIBS for lof
from sklearn.neighbors import LocalOutlierFactor

# LIBS for dealing with outliers.
import pandas as pd


'''################################################################
                     LOCAL VARIABLES
################################################################'''

cpzl = {
75001:'yellow',75002:'blue',75003:'dimgray',75004:'darkslateblue',75005:'olive',
75006:'green',75007:'cyan',75008:'lime',75009:'purple',75010:'maroon',
75011:'cyan',75012:'gold',75013:'red',75014:'purple',75015:'royalblue',
75016:'orange',75017:'magenta',75018:'teal',75019:'navy',75020:'violet',
92100:'black', 92110:'black', 92120:'black', 92130:'black', 92150:'black',
92170:'black', 92200:'black', 92240:'black', 92300:'black', 92310:'black',
92600:'black', 93100:'black', 93170:'black', 93260:'black', 93300:'black',
93310:'black', 93400:'black', 93500:'black', 94120:'black', 94130:'black',
94160:'black', 94200:'black', 94220:'black', 94250:'black', 94270:'black',
94300:'black', 94410:'black', 94700:'black', 94800:'black', 95170:'black'
}


'''################################################################
                     FUNCTIONS
################################################################'''

'''Function that get's the center of mass for every district
    It will receive X which is a set of coordinates and it
    performs kmeans algorithm to determine its centroid
'''

def get_centroids(X,k):
    np.random.seed(k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids


'''Function that returns potential outilers of listings
per zipcode using the lof method and also returns the centroids of
every zipcode. Uses local variable cpzl'''

def get_centroids_and_outliers(data):

    outliers = {}
    centroids = {}
    number_of_centroids = 1
    k = 25

    for arrondissement in data.zipcode.unique():

        df = data.loc[data.zipcode==arrondissement,['longitude','latitude']]
        oust, inli = get_lof(df, k, False)
        outliers[arrondissement] = oust.index.values
        centroids[arrondissement] = get_centroids(df.values, number_of_centroids)

    # Set centroids in dataframe
    centr = pd.DataFrame(list(centroids.items()), columns=['zipcode','coordinates'])
    # Break the coordinates into longi and lati
    centr['longitude'] = centr.coordinates.apply(lambda x: x[0][0])
    centr['latitude'] = centr.coordinates.apply(lambda x: x[0][1])
    centr['color'] = centr.zipcode.apply(lambda x: cpzl[x])
    centr.set_index('zipcode', inplace=True)
    #centr.drop('coordinates', axis=1, inplace=True)

    # Set outliers in dataframe
    abrnt = pd.concat({k: pd.DataFrame(v) for k, v in outliers.items()})
    abrnt.reset_index(inplace=True)
    abrnt.set_index(0, inplace=True)
    abrnt.drop('level_1',axis=1, inplace=True)
    abrnt.index.name = 'listing_id'
    abrnt.rename(columns={'level_0': 'fake_zipcode'}, inplace=True)
    # Add color
    abrnt['color'] = abrnt.fake_zipcode.apply(lambda x: cpzl[x])
    # Add latitude and longitude
    abrnt = pd.merge(abrnt, data.loc[:,['latitude','longitude']], right_index=True, left_index=True)

    return centr, abrnt

'''Cluster analysis functions
Does DBSCAN, Covariance ellipses, GMM and Dirichlet MM and LOF
'''

# FUNCTION THAT USES THE ELLIPTICAL ENVELOPE

def get_elliptic_envelope(X1):

# Define "classifiers" to be used
    classifiers = {
        "Elliptic Envelope": EllipticEnvelope(),
        "Empirical Covariance": EllipticEnvelope(support_fraction=1., contamination=0.261),
        "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(contamination=0.261),
        }

    # list color codes for plotting
    colors = ['firebrick','gold','mediumorchid']
    legend1 = {}

    # Learn a frontier for outlier detection with several classifiers
    a = [0.995,1.001]
    b = [0.9999,1.0001]
    lim_min = np.min(X1, axis=0)
    lim_max = np.max(X1, axis=0)

    # create meshgrids for plotting ellipses (contours)
    xx1, yy1 = np.meshgrid(np.linspace(lim_min[0]*a[0], lim_max[0]*a[1], 500), np.linspace(lim_min[1]*b[0], lim_max[1]*b[1], 500))

    # loop over classifiers and fit then plot
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # compute and store plot for X1
        plt.figure(1)
        clf.fit(X1)  # fit current classifier
        # get decision function with outlier thresh = 0 (default)
        Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        # reshape for plotting
        Z1 = Z1.reshape(xx1.shape)
        # plot

        #legend1[clf_name] = plt.contour(
        plt.contour(
            xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])

    #legend1_values_list = list(legend1.values())
    #legend1_keys_list = list(legend1.keys())

    # Plot the results for X1

    plt.figure(1)  # two clusters
    plt.title("Outlier detection on NBA Players")
    plt.scatter(X1[:, 0], X1[:, 1], color='royalblue')  # just data points
    # set figure limits from meshgrids
    plt.xlim((xx1.min(), xx1.max()))
    plt.ylim((yy1.min(), yy1.max()))

    # set labels
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")

    # create legend
    #plt.legend((legend1_values_list[0].collections[0],
    #            legend1_values_list[1].collections[0],
    #            legend1_values_list[2].collections[0]),
    #           (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2]),
    #           loc="lower center",
    #           prop=matplotlib.font_manager.FontProperties(size=12))

    plt.show()


## FUNCTION USED IN GMM

# we need to pass the means and covariances to the function
#  these will be esimated by the mixture model
def plot_results(X, Y_, means, covariances, index, title, color_iter):
    # create subplots
    fig = plt.figure(figsize=(10,10))
    splot = plt.subplot(2, 1, 1 + index)
    # loop over all (mean, covar, color) tuples by zipping them together
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # get eigenvectors (v) and eigenvalues (w) of covar
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)  # scale v
        # create unit vector for computing angles
        u = w[0] / linalg.norm(w[0])

        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        # plot ellipse
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    return u

    # GET GMM

def get_gmm(X, n_components = 1):
    # iterator to cycle over colors while plotting
    color_iter = itertools.cycle(['c', 'cornflowerblue', 'gold', 'orange', 'olive'])

    # Fit a Gaussian mixture with EM using eight components
    # fit mixture model to X, to estimate Y_ and stats (i.e. mean, covar)
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
    w = plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture', color_iter)

    # FUNCTION FOR LOF
def get_lof(df, k, display):
# fit the model
    lof = LocalOutlierFactor(n_neighbors=k)
    X = df.as_matrix()
    lof.fit(X)

    outlier_scores = lof.negative_outlier_factor_
    outlier_idx = [idx for idx, score in enumerate(outlier_scores) if score < -2]

    outliers = df.iloc[outlier_idx]
    inliers = df.iloc[~df.index.isin(outliers.index.values)]

    if display:
        # plot the data observations
        plt.plot(inliers['longitude'], inliers['latitude'], 'ob', alpha=0.2)

        # plot the outliers
        lines = plt.plot(outliers['longitude'], outliers['latitude'], 'Xr')

        plt.title("Outliers using LOF")
        plt.ylabel("latitude")
        plt.xlabel("longitude")
        plt.show()

    return outliers, inliers

# FUNCTION KMeans

def get_optimal_k(testKmeanXZ):
    second_derivative = []
    second_derivative = [abs(testKmeanXZ[elem]-testKmeanXZ[elem+1]) for elem in range(len(testKmeanXZ)-1)]
    return second_derivative.index(max(second_derivative))+2

def test_kmeans(data, nClusterRange, display):
    inertias = np.zeros(len(nClusterRange))
    for i in range(len(nClusterRange)):
        model = KMeans(n_clusters=i+1, init='k-means++').fit(data)
        inertias[i] = model.inertia_
    if display:
        figInertiaWithKNonConvex = plt.figure(figsize=(10,3))
        plt.plot(kRange, testKmeanXZ, 'o-', color='royalblue', linewidth=3)
        plt.plot([k], [testKmeanXZ[k-1]], 'o--', color='gold', linewidth=3)
        plt.show()

    return inertias

def get_kmeans(X, k, threshold):
    np.random.seed(k)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # get array of distances
    # each row is a point
    # each element is the distance to a cluster center
    #  Ex: element at row zero, col zero = distance to cluster label 0
    distances = kmeans.transform(X)


    # separate outliers from inliers
    clusters = [[],[],[],[],[]]
    inliers = [[],[],[],[],[]]
    outliers = [[],[],[],[],[]]

    for i,(l,d) in enumerate(zip(labels, distances)):
        # add correct distance to clusters list
        # based on label and index (i)
        clusters[l].append((i, d.min()))

    # get stats for each cluster
    centroid_median_distances = [np.median([c[1] for c in clust]) for clust in clusters]
    centroid_std_distances = [np.std([c[1] for c in clust]) for clust in clusters]

    # inlier if < 3.5 std from median, else outlier
    colors_o = ['firebrick','g','navy','darkorange','darkviolet']
    colors = ['r','lime', 'b', 'gold', 'magenta']
    for i,(c,s,m) in enumerate(zip(clusters,
            centroid_std_distances,
                     centroid_median_distances)):
        for el in c:
            if el[1] < (threshold*s-m):
                inliers[i].append(el[0])
            else:
                outliers[i].append(el[0])

    for i in range(k):
        # select only data observations with cluster label == i
        ds_in = X[inliers[i]]
        ds_out = X[outliers[i]]

        # plot the data observations
        plt.plot(ds_in[:,0], ds_in[:,1], 'o', alpha=0.4, c=colors[i])
        plt.plot(ds_out[:,0], ds_out[:,1], 'x', alpha=0.5, c=colors_o[i])

        # plot the centroids
        lines = plt.plot(centroids[i,0], centroids[i,1], 'kX')

        # make the centroid x's bigger
        plt.setp(lines, ms=10.0)
        plt.setp(lines, mew=1.0)

    plt.title("Outliers using K-Means")
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    plt.show()
    return centroids


# DBSCAN SUBFUNCTIONS
def neighborhoods(data, epsilon):
    n_data = data.shape[0]
    nn = np.zeros(n_data)
    # Compute all - n(n-1)/2 - pairwise distances
    pMat = pdist(data)
    # Compute the (i,j) indexes of pairs that correspond to each elements in pMat
    indexes = np.array([ (i,j) for i in range(n_data) for j in range(i+1, n_data)])
    # Find pairs of data points that are closer than epsilon
    pairs = indexes[np.where(pMat < epsilon)]
    for pair in pairs:
        nn[pair[0]] += 1
        nn[pair[1]] += 1
    return nn

# Build a dictionary of clusters and their points
def buildClusterDict(model):
    npts = model.labels_.shape[0]
    clusterDict = {}
    for i in range(npts):
        lbl = model.labels_[i]
        if lbl in clusterDict:
            clusterDict[lbl].append(i)
        else:
            clusterDict[lbl] = [i]
    return clusterDict


def get_district_cluster(matXY, epsilon=False, max_pts=False):

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
