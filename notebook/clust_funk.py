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

'''Cluster analysis functions
Does DBSCAN, Covariance ellipses, GMM and Dirichlet MM and LOF
'''

# FUNCTION THAT USES THE ELLIPTICAL ENVELOPE

def get_elliptic_envelope(arrondissement, data):

# Define "classifiers" to be used
    classifiers = {
        "Elliptic Envelope": EllipticEnvelope(),
        "Empirical Covariance": EllipticEnvelope(support_fraction=1., contamination=0.261),
        "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(contamination=0.261),
        }

    # Get data

    X1 = data.loc[(data.zipcode.isin(arrondissement)),['longitude','latitude']].values

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
    plt.ylabel("Three Point Percentage")
    plt.xlabel("Points per Game")

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

def get_gmm(arrondissement, data, n_components = 1):
    # iterator to cycle over colors while plotting
    color_iter = itertools.cycle(['c', 'cornflowerblue', 'gold', 'orange', 'olive'])
    X = data.loc[data.zipcode==arrondissement,['longitude','latitude']].values

    # Fit a Gaussian mixture with EM using eight components
    # fit mixture model to X, to estimate Y_ and stats (i.e. mean, covar)
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
    w = plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture', color_iter)

    # FUNCTION FOR LOF
def get_lof(df, X, k):
# fit the model
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(X)

    outlier_scores = lof.negative_outlier_factor_

    points = []
    threes = []
    # loop, store and print players who are outliers
    for idx, score in enumerate(outlier_scores):
        if score < -2:
            listing = df.iloc[idx, :]

            points.append(listing['longitude'])
            threes.append(listing['latitude'])

    #        print(listing['Name'], listing['PPG'], listing['3P%'])

    #Plot inliers as blue circles, and outliers as large X's

    # plot the data observations
    plt.plot(df['longitude'], df['latitude'], 'o', alpha=0.2)

    # plot the outliers
    lines = plt.plot(points, threes, 'kx')

    # make the centroid x's bigger
    plt.setp(lines, ms=10.0)
    plt.setp(lines, mew=1.0)

    plt.title("Outliers using LOF")
    plt.ylabel("latitude")
    plt.xlabel("longitude")
    #plt.ylim(ymax = 115, ymin = -10)
    #plt.xlim(xmax = 35, xmin = -5)
    plt.show()

# FUNCTION KMeans
def get_kmeans(data, arrondissement, k, threshold):
    np.random.seed(arrondissement)
    X = data.loc[data.zipcode.isin(arrondissement),['longitude','latitude']].values

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


# DBSCAN SUBFUNCTIONS
def get_density(pairDistList):
    figPairwiseDistances = plt.figure(figsize=(15,3))

    n, b, patches = plt.hist(pairDistList, bins=100, color='royalblue')
    bin_max = np.where(n == n.max())

    # Plot the maximum value of the histogram
    plt.axvline(x=b[bin_max][0], linestyle='--', color='gold', linewidth=4)

    # Anotation
    plt.text(b[bin_max][0]*2,n[bin_max][0]*0.7,r'~Cluster Scale:'+str(b[bin_max][0]), size=13)

    plt.xlabel('Pairwise Distance')
    plt.ylabel('# of data points')
    plt.title('Dataset Separation Distribution')
    plt.show()
    return b[bin_max][0]
    #figPairwiseDistances.savefig('./img/k2c-dbscan-pdist-epsilon.png')

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

def get_edge_points(nn, epsilon):
    figNeighborhoods = plt.figure(figsize=(15,4))

    #plt.hist(nn, bins=45, color='darkorange')
    n, b, patches = plt.hist(nn, bins=np.arange(min(nn), max(nn) + 1, 1), color='royalblue')
    bin_max = np.where(n == n.max())

    plt.axvline(x=b[bin_max][0], linestyle='--', color='gold', linewidth=4)
    plt.text(b[bin_max][0], n[bin_max][0]*0.7, 'maxPts='+str(b[bin_max][0]), size=16, color='dimgray')

    plt.xlabel('# of neighbors', size=14)
    plt.ylabel('# of data points', size=14)
    plt.title('Dataset Density Distribution (Eps. = '+str(epsilon)+')', size=14)
    plt.show()
    return b[bin_max][0]
#figNeighborhoods.savefig('./img/k2c-dbscan-neighborhoods.png')

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

def dbscanit(matXY, epsilon, max_pts):
    # DBSCAN
    dbmodel = DBSCAN(eps=epsilon, min_samples=max_pts).fit(matXY)
    labels = dbmodel.labels_
    #print(set(labels))

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print(n_clusters_)

    # ClusterDict has the point name (or index) as value and it's label as key.
    clusterDict = buildClusterDict(dbmodel)

    # Noise
    #clusterDict[-1]

    # Cluster Statistics verbose report
    #for lbl in clusterDict:
    #    if lbl == -1:
    #        print("Model identified {0} data points as noise".format(len(clusterDict[lbl])))
    #    else:
    #        print("Cluster {0} has {1} members".format(lbl, len(clusterDict[lbl])))

    # Number of Core samples
    #print(dbmodel.core_sample_indices_.shape)
    # noise data point mask
    noise_sample_mask = np.zeros(n_data, dtype=bool)
    noise_sample_mask[clusterDict[-1]] = True

    # Representation in 2D
    figDBSCAN2D = plt.figure(figsize=(10,5))

    plt.scatter(matXY[:,0], matXY[:,1], c=dbmodel.labels_)
    plt.scatter(matXY[noise_sample_mask, 0], matXY[noise_sample_mask, 1], c='k', label="noise")
    plt.xlabel('Longitude', size=14)
    plt.ylabel('Latitude', size=14)
    plt.legend()
    plt.title('DBSCAN (eps='+str(epsilon)+', min_sample='+str(max_pts)+')', size=14)
    plt.show()
    #figDBSCAN2D.savefig('./img/k2c-dbscan-basic-2D.png')


# FUNCTION THAT ANALYZES AND RUN DBSCAN

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
