#  Because pybind11 cannot generate default parameters well, this code is to set them

import inc_pca_cpp


class IncPCA(inc_pca_cpp.IncPCA):
    """Incremental principal components analysis.
    Implementation of the incremental PCA of Ross et al., 2008 and the
    geometric transformation, position estimation, uncertatinty measures by
    Fujiwara et al., 2019.

    Parameters
    ----------
    n_components : int, (default=2)
        Number of components to keep. If n_components is 0,
        then n_components is set to min(n_samples, n_features).
    forgetting_factor: float, (default=1.0)
        A forgetting factor, f,  provides a way to reduce the contributions of
        past observations to the latest result. The value of f ranges from 0 to
        1 where f = 1 means no past results will be forgotten. When 0 <= f < 1,
        the contributions of past observations gradually decrease as new data
        points are obtained. As described in [Ross et al., 2008], the effective
        size of the observation history (a number of observations which affect
        the PCA result) equals to m/(1 - f) (m: number of new data points). For
        example, when f = 0.98 and m = 2, the most recent 100 observations are
        effective.
    Attributes
    ----------
    Examples
    --------
    >>> import math
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from inc_pca import IncPCA

    >>> #
    >>> # 0. load data
    >>> #
    >>> iris = datasets.load_iris()
    >>> shuffled_order = [
    ...     78, 142, 39, 31, 53, 42, 89, 5, 91, 65, 19, 49, 112, 125, 96, 134, 83, 77,
    ...     0, 79, 100, 92, 109, 38, 67, 84, 123, 80, 62, 126, 144, 107, 50, 149, 127,
    ...     46, 21, 136, 41, 35, 20, 139, 82, 27, 18, 66, 118, 145, 124, 93, 129, 97,
    ...     146, 138, 120, 95, 94, 147, 43, 143, 13, 131, 116, 15, 3, 14, 37, 73, 26,
    ...     70, 99, 25, 7, 23, 36, 76, 119, 88, 44, 110, 140, 57, 105, 34, 32, 103, 74,
    ...     114, 87, 106, 111, 40, 117, 81, 86, 63, 56, 133, 33, 4, 9, 1, 51, 24, 30,
    ...     72, 69, 61, 64, 113, 8, 135, 55, 71, 137, 85, 108, 128, 6, 90, 121, 16,
    ...     148, 47, 115, 59, 17, 12, 60, 101, 52, 104, 68, 54, 2, 11, 22, 130, 10, 29,
    ...     102, 45, 141, 122, 58, 132, 28, 75, 98, 48
    ... ]
    >>> X = iris.data[shuffled_order, ]
    >>> group = iris.target[shuffled_order]
    >>> target_names = iris.target_names

    >>> #
    >>> # 1. incremental PCA and geometric transformation examples
    >>> #
    >>> ipca = IncPCA(2, 1.0)
    >>> m = 2  # number of new points
    >>> # process 20 x m new points
    >>> for i in range(20):
    ...     ipca.partial_fit(X[m * i:m * (i + 1), ])
    >>> Y_a = ipca.transform(X[0:m * 20, ])
    >>> # add m new points
    >>> ipca.partial_fit(X[m * 21:m * 22, ])
    >>> Y_b = ipca.transform(X[0:m * 22, ])
    >>> # apply the geometric transformation
    >>> Y_bg = IncPCA.geom_trans(Y_a, Y_b)

    >>> # plot results
    >>> plt.figure()
    >>> colors = ['navy', 'turquoise', 'darkorange']
    >>> lw = 2
    >>> for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ...     plt.scatter(
    ...         Y_a[group[0:len(Y_a)] == i, 0],
    ...         Y_a[group[0:len(Y_a)] == i, 1],
    ...         color=color,
    ...         alpha=.8,
    ...         lw=lw,
    ...         label=target_name)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1)
    >>> plt.title('Inc PCA of IRIS with first obtained data points')
    >>> plt.figure()
    >>> for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ...     plt.scatter(
    ...         Y_b[group[0:len(Y_b)] == i, 0],
    ...         Y_b[group[0:len(Y_b)] == i, 1],
    ...         color=color,
    ...         alpha=.8,
    ...         label=target_name)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1)
    >>> plt.title('Inc PCA of IRIS with additional data points')
    >>> plt.figure()
    >>> for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ...     plt.scatter(
    ...         Y_bg[group[0:len(Y_bg)] == i, 0],
    ...         Y_bg[group[0:len(Y_bg)] == i, 1],
    ...         color=color,
    ...         alpha=.8,
    ...         label=target_name)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1)
    >>> plt.title('Inc PCA of IRIS with additional points & geom trans')
    >>> plt.show()

    >>> #
    >>> # 2. pos estimation example
    >>> #
    >>> # incremental PCA result with only 3 features
    >>> ipca_only_3d = IncPCA(2, 1.0)
    >>> for i in range(22):
    ...     ipca_only_3d.partial_fit(X[m * i:m * (i + 1), 0:3])
    >>> Y_only_3d = ipca_only_3d.transform(X[0:m * 22, 0:3])
    >>> # compare actual position with PCA resutl with full features and estimated pos
    >>> actual_pos = ipca.transform([X[48, ]])[0]
    >>> pos_only_3d = ipca_only_3d.transform([X[48, 0:3]])[0]
    >>> est_pos, uncert_u = IncPCA.pos_est(pos_only_3d, Y_only_3d, Y_b)
    >>> beta = 0.5
    >>> uncert_w = beta * uncert_u + (1.0 - beta) * ipca.get_uncert_v(3)
    >>> print("actual pos:", actual_pos)
    >>> print("estimated pos:", est_pos)
    >>> print("combined uncertainty W:", uncert_w)

    >>> #
    >>> # 3. example of updating beta (combined uncertainty weight)
    >>> #
    >>> # prepare ipca for each dimension l (l: {1, 2, 3, 4})
    >>> ipcas = [
    ...     IncPCA(1, 1.0),
    ...     IncPCA(2, 1.0),
    ...     IncPCA(2, 1.0),
    ...     IncPCA(2, 1.0)
    ... ]
    >>> D = len(ipcas)
    >>> for l in range(len(ipcas)):
    ...     ipcas[l].partial_fit(X[0:10, 0:l + 1])
    >>> def distance(p0, p1):
    ...     return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    >>> # initial parameters
    >>> beta = 0.1
    >>> sq_grad = 0.0
    >>> sq_dbeta = 0.0
    >>> betas_for_plot = [beta]
    >>> for iter in range(50):
    ...     n = 10 + m * iter
    ...     sprimes = []
    ...     uncert_vs = np.zeros(D)
    ...     uncert_us = np.zeros((D, m))
    ...
    ...     Y_D = ipcas[D - 1].transform(X[0:n, 0:D])
    ...
    ...     for l in range(D):
    ...         ipcas[l].partial_fit(X[n:n + m, 0:l + 1])
    ...         Y_l = ipcas[l].transform(X[0:n, 0:l + 1])
    ...         # add column with zeros when l=1 to make 2D points
    ...         if (l == 0):
    ...             Y_l = np.concatenate((Y_l, np.zeros((n, 1))), axis=1)
    ...
    ...         uncert_vs[l] = ipcas[D - 1].get_uncert_v(l)
    ...
    ...         sigma = np.zeros((m, n))
    ...         sprime = np.zeros((m, n))
    ...
    ...         for i in range(n):
    ...             for u in range(m):
    ...                 new_point = [X[n + u, 0:l + 1]]
    ...                 new_point_pos = ipcas[l].transform(new_point)
    ...
    ...                 # add column with zeros when l=1 to make 2D points
    ...                 if (l == 0):
    ...                     new_point_pos = np.concatenate(
    ...                         (new_point_pos, np.zeros((1, 1))), axis=1)
    ...
    ...                 est_pos, uncert_u = IncPCA.pos_est(new_point_pos, Y_l, Y_D)
    ...
    ...                 sigma[u, i] = distance(Y_D[-(m + u)], Y_D[i])
    ...                 sprime[u, i] = distance(Y_l[-(m + u)], Y_l[i])
    ...                 uncert_us[l, u] = uncert_u
    ...         sprimes.append(sprime)
    ...
    ...     beta, sq_grad, sq_dbeta = IncPCA.update_uncert_weight(
    ...         beta, sq_grad, sq_dbeta, sigma, sprimes, uncert_us, uncert_vs)
    ...     betas_for_plot.append(beta)
    >>> plt.scatter(list(range(len(betas_for_plot))), betas_for_plot)
    >>> plt.xlabel('Number of updates')
    >>> plt.ylabel('beta')
    >>> plt.title('Automatic update of beta')
    >>> plt.show()
    Notes
    -----
    The incremental PCA model is from:
    `D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
    pp. 125-141, 2008.`
    The geometric transformation, position estimation, and uncertatinty measures
    are from:
    `T. Fujiwara, J.-K. Chou, Shilpika, P. Xu, L. Ren, K.-L. Ma, Incremental
    Dimensionality Reduction Method for Visualizing Streaming Multidimensional
    Data, arXiv preprint, 2019.`
    The version of implementation in Scikit-learn was refered to implement the
    incremental PCA of Ross et al, 2008. However, this implementation includes
    various modifications (simplifying the parameters, adding forgetting factor,
    etc).
    Incremental PCA in Scikit-learn:
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html
    References
    ----------
     D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
        Tracking, International Journal of Computer Vision, Volume 77,
        Issue 1-3, pp. 125-141, 2008.
     T. Fujiwara, J.-K. Chou, Shilpika, P. Xu, L. Ren, K.-L. Ma, Incremental
        Dimensionality Reduction Method for Visualizing Streaming
        Multidimensional Data, arXiv preprint, 2019.
    """

    def __init__(self, n_components=2, forgetting_factor=1.0):
        super().__init__(n_components, forgetting_factor)

    def initialize(self):
        return super().initialize()

    def partial_fit(self, X):
        """Incremental fit with new datapoints, X. With this, PCs are updated
        from previous results incrementally. X's row (i.e., number of data
        points) must be greater than or equal to 2.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features. n_samples must be >= 2.
            n_features and n_samples must be >= n_components. Also, n_features
            must be the same size with the first X input to partial_fit.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().partial_fit(X)

    def transform(self, X):
        """Obtaining transformed result Y with X and current PCs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing data, where n_samples is the number of samples and
            n_features is the number of features. n_features must be the same
            size with the traiding data's features used for partial_fit.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            Returns the transformed (or projected) result.
        """
        return super().transform(X)

    def get_loadings(self):
        """Obtaining current PC loadings.

        Returns
        -------
        W : array-like, shape (n_components, n_features)
            Returns PC loadings.
        """
        return super().get_loadings()

    @classmethod
    def geom_trans(cls, Y1, Y2):
        """Finding the geometric transformation matrix, which maximizes the
        Y2's overlap to Y1  with a combination of uniform scaling, rotation,
        and sign flipping.

        Parameters
        ----------
        Y1 : array-like, shape (n_samples, n_dimensions)
            Data point positions (n_dimensions). Y1 is used as a base of ovarlapping.
        Y2 : array-like, shape (n_samples, n_dimensions)
            Data point positions (n_dimensions). geom_trans finds a matrix to optimally
            overlap Y2 to Y1.
        Returns
        -------
        Y2_g: array-like, shape (n_samples, n_dimensions)
            Y2 after applied the geometric transformation.
        """
        return super().geom_trans(Y1, Y2)

    def get_uncert_v(self, n_obtained_features):
        """Obtaining the uncertainty measure, V, introduced in
        [Fujiwara et al., xxxx] with current PCA result and a number of
        obtained features of a new data point.

        Parameters
        ----------
        n_obtained_features : int
            Number of obtained features of a new data point.
            n_obtained_features must be <= n_components.
        Returns
        -------
        V : double
            Returns the uncertainty measure, V. V will be 0, 1 if
            n_obtained_features = 0, n_components, respectively.
        """
        return super().get_uncert_v(n_obtained_features)

    @classmethod
    def pos_est(cls, p, Y1, Y2):
        """Estaimated position of p. Refer [Fujiwara et al., xxxx].

        Parameters
        ----------
        p: array-like, shape(1, 2)
            A data point position trasformed with PCA of d=l (l<=D).
        Y1 : array-like, shape (n_samples, 2)
            Data point positions (2D). Y1 shoule be a transformed position with
            PCA of d=l (l<=D).
        Y2 : array-like, shape (n_samples, 2)
            Data point positions (2D). Y2 shoule be a transformed position with
            PCA of d=D.
        Returns
        -------
        est_pos : array-like, shape(1, 2)
            Returns the estimated 2D position.
        uncert_u: float
            Reutrn the uncertainty measure, U
        """
        estPosAndUncertU = super().pos_est(p, Y1, Y2)
        return estPosAndUncertU[0], estPosAndUncertU[1]

    @classmethod
    def update_uncert_weight(cls, current_beta, current_sq_grad,
                             current_sq_dbeta, sigma, sprimes, uncert_us,
                             uncert_vs):
        """Update the combined uncertainty weight, beta.
        Refer [Fujiwara et al., xxxx].

        Parameters
        ----------
        current_beta: float
            Current beta obtained Eq. 12 in [Fujiwara, xxx].
        current_sq_grad: float
            Used in Adadelta calculation in Eq. 11. Set 0 as an initial value.
        current_sq_dbeta: float
            Used in Adadelta calculation in Eq. 11. Set 0 as an initial value.
        sigma: array-like, shape(m, n)
            Distance between m new data positions and n exisiting data positions
             in the PCA result after new data points obtain D dimensions.
        sprimes: array-like, D x shape(m, n)
            Distances between m esitmated data positions and n exisiting data
            positions for each dimension l (1 <= l <= D).
        uncert_us: array-like, shape(D, m)
            Uncertainty measure U for m new data points for each dimension l.
        uncert_vs: array-like, shape(D)
            Uncertainty measure V for each dimension l.
        Returns
        -------
        updated_beta: float
        updated_sq_grad: float
        updated_sq_dbeta: float
            Use these updated values for the next run of update_uncert_weight
        """
        result = super().update_uncert_weight(current_beta, current_sq_grad,
                                              current_sq_dbeta, sigma, sprimes,
                                              uncert_us, uncert_vs)
        return result[0], result[1], result[2]
