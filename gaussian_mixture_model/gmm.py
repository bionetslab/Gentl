from matplotlib import rc
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as tkr
import scipy.stats as stats
import imageio as iio

# read an image
img = iio.imread("/Users/wushengyang/Documents/Gentl/feature_extraction_methods/cancer.jpg")
x=img.copy()
# x = open("prueba.dat").read().splitlines()

# create the data
x = np.concatenate((np.random.normal(5, 5, 1000),np.random.normal(10, 2, 1000))).reshape(-1,1) # the cancer ROI image

# f = np.ravel(x).astype(np.float)
# f=f.reshape(-1,1)
g = mixture.GaussianMixture(n_components=2,covariance_type='full')
g.fit(x)
weights = g.weights_
means = g.means_
covars = g.covariances_
# [Note: std_dev=(var)^0.5, or var = (std_dev)^2]
# Just use means.

# plt.hist(x, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)

f_axis = x.copy().ravel()
f_axis.sort()
# plt.plot(f_axis,weights[0]*stats.norm.pdf(f_axis,means[0],np.sqrt(covars[0])).ravel(), c='red')

plt.rcParams['agg.path.chunksize'] = 10000

plt.grid()
plt.show()