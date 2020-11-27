# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kpca(fitted_pca, max_lambdas=50):
    """
    Plot PCA eigenvalues
    :param fitted_pca: Fitted PCA/KPCA object
    :param max_lambdas: Maximum number of components in the x_axis
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    lambdas = fitted_pca.lambdas_
    n_lambdas = len(lambdas)
    if n_lambdas > max_lambdas:
        lambdas = lambdas[:max_lambdas]
        n_lambdas = max_lambdas
    x_values = np.arange(1, n_lambdas + 1, step=1)
    plt.plot(x_values, np.log(lambdas), marker='o', linestyle='--', color='k')
    plt.ylim(0, np.log(max(lambdas)) + 1)
    plt.xlabel('Number of Components')
    if n_lambdas <= 20:
        plt.xticks(np.arange(0, n_lambdas + 1, step=1))
    plt.ylabel('Eigenvalue (Log Scale)')
    plt.title('Number of components vs Eigenvalues')
    ax.grid(axis='x')
    plt.show()


def fit_pca(input_data, kpca_type='poly', gamma=10, degree=3, n_components=5, n_samples=1000, random_state=0):
    """
    Fit PCA over given data
    :param input_data: Input numpy array
    :param kpca_type: Set PCA kernel, default is polynomial. Others include 'linear', 'rbf', 'sigmoid', and 'cosine'
    :param gamma: Gamma parameter for polynomial and RBF PCA
    :param degree: Degree parameter for polynomial PCA
    :param n_components: Number of PCA components, if None, all non-zero components are kept
    :param n_samples: Number of samples to randomly choose from the input data for fitting. Full data cannot be used
    because of computational cost
    :param random_state: PRNG seed for sampling for KPCA
    :return: Fitted KPCA object
    """

    fit_kpca = None
    sample_data = input_data.sample(n=n_samples, replace=True, random_state=random_state)
    if n_components is None:
        if kpca_type in ['poly', 'rbf']:
            fit_kpca = KernelPCA(kernel=kpca_type, gamma=gamma, degree=degree, n_jobs=-1).fit(sample_data)
        elif kpca_type in ['linear', 'sigmoid', 'cosine']:
            fit_kpca = KernelPCA(kernel=kpca_type, n_jobs=-1).fit(sample_data)
    else:
        fit_kpca = KernelPCA(kernel=kpca_type, gamma=gamma, degree=degree, n_components=n_components,
                             n_jobs=-1).fit(sample_data)
    return fit_kpca

