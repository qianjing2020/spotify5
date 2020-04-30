# Define scree plot function
import numpy as np
import matplotlib.pyplot as plt

def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    INPUT: pca - the result of instantian of PCA in scikit learn    
    OUTPUT: None
    '''
    num_components = len(
        pca.explained_variance_ratio_)  # number of PCA components
    # evenly spaced values in an interval, default start value is 0.
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    fig, axes = plt.subplots(figsize=(18, 6))
    cumvals = np.cumsum(vals)  # cumulative variance explained

    axes.bar(ind, vals, color='g')
    axes.plot(ind, cumvals, color='g')
    for i in range(num_components):
        axes.annotate(r"%s" % (
            (str(vals[i]*100)[:3])), (ind[i], vals[i]), va="bottom", ha="center", fontsize=16)

    axes.tick_params(axis='x', labelsize=16, width=0, )
    axes.tick_params(axis='y', labelsize=16, width=2, length=12)

    axes.set_xlabel("Principal Component", fontsize=16)
    axes.set_ylabel("Variance Explained (%)", fontsize=16)
    plt.title('Explained Variance Per Principal Component', fontsize=16)
    plt.show()

