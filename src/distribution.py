'''
    Distribution and distribution shift estimation from data.
'''
import warnings
from typing import List, Tuple

import numpy as np
import scipy.stats as st
from KDEpy import TreeKDE
from sklearn.neighbors import NearestNeighbors

def kde(
        train_set: np.ndarray, 
        eval_sets: List[np.ndarray], 
        kernel: str, 
        bandwidth: float,
        ) -> Tuple[float, List[float]]:
    '''
        Kernel density estimation with KDEpy.
    '''
    kde = TreeKDE(kernel=kernel, bw=bandwidth).fit(train_set)

    mll_train = np.log(kde.evaluate(train_set)).mean()
    mll_eval = [np.log(kde.evaluate(set)).mean() for set in eval_sets]

    return mll_train, mll_eval


def gaussian_model(
        train_set: np.ndarray, 
        eval_sets: List[np.ndarray],
        ) -> Tuple[float, List[float]]: 
    '''
        Parametric gaussian estimation.
    '''
    sample_mean = train_set.mean(axis=0)
    sample_cov = np.cov(train_set.transpose())

    mll_train = st.multivariate_normal.logpdf(train_set, mean = sample_mean, cov = sample_cov).mean()
    mll_eval = [st.multivariate_normal.logpdf(set, mean = sample_mean, cov = sample_cov).mean() for set in eval_sets]

    return mll_train, mll_eval


def kl_divergence(
        s1: np.ndarray, 
        s: List[np.ndarray], 
        k: int = 20,
        ) -> List[float]: 
    ### Taken from https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
    """An efficient version of the scikit-learn estimator by @LoryPack
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Contributed by Lorenzo Pacchiardi (@LoryPack)
    """
    # Expects [N, D]
    for s2 in s:
        assert len(s1.shape) == len(s2.shape) == 2
        # Check dimensionality of sample is identical
        assert s1.shape[1] == s2.shape[1]

    n, m = len(s1), len(s[0])
    d = float(s1.shape[1])

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
    s_neighbourhood = [NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2) for s2 in s]

    s1_distances, _ = s1_neighbourhood.kneighbors(s1, k + 1)
    s_distances = [sn.kneighbors(s1, k)[0] for sn in s_neighbourhood]
    rho = s1_distances[:, -1]
    nus = [sd[:, -1] for sd in s_distances]
    if np.any(rho == 0):
        warnings.warn(
            f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
            f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
            f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
            RuntimeWarning,
        )
    Ds = [np.sum(np.log(nu / rho)) for nu in nus]

    return [(d / n) * D + np.log(m / (n - 1))
            for D in Ds]  # this second term should be enough for it to be valid for m \neq n