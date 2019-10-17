from __future__ import division, print_function
import numpy as np
from statsmodels.stats.stattools import medcouple
from statsmodels.nonparametric.api import KDEUnivariate
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def adjBoxplotStats(x, coeff=1.5, a=-4., b=3.): #creates bounds for outlier detection
    """ Calculates an adjusted interquartile range of distribution and points outside this range
    can be identified as outliers.

    Calculation and default parameters describe in: An adjusted boxplot for skewed distrubtions, Vanderviere
    and Huber, COMPSTAT 2004 Symposium.

    :param x: array
    :param coeff: scalar
    :param a: scalar
    :param b: scalar
    :return: dict with keys:
        fence: list of upper and lower adjusted quartiles
        IQR: unadjusted interquartile range
        MC: skew of distribution using medcouple which is a robust skew measure.
    """
    x = np.array(x)
    MC = medcouple(x)
    [Q1, Q2, Q3] = np.percentile(x, [25, 50, 75])
    IQR = Q3 - Q1
    if (MC >= 0):
        fence = [Q1 - coeff*np.exp(a * MC)*IQR, Q3 + coeff*np.exp(b * MC)*IQR]
    else:
        fence = [Q1 - coeff*np.exp(-b * MC)*IQR, Q3 + coeff*np.exp(-a * MC)*IQR]
    return {'fence': fence, 'IQR': IQR, 'medcouple':MC}

	
def kde(x, x_grid, weights=None, bandwidth='normal_reference', kernel='gau', gridsearch = False, **kwargs):
    """
    Kernel Density Estimation with KDEUnivariate from statsmodels. **kwargs are the named arguments of KDEUnivariate.fit()
    """
    if gridsearch and (weights is None):
        grid = GridSearchCV(KernelDensity(), {'bandwidth' : np.linspace(0.1, 1.0, 30)}, cv = 20)
        grid.fit(x.reshape(-1,1))
        model = grid.best_estimator_
        print('CV bandwidth param is: ', grid.best_params_)
        return model.score_samples(x_grid.reshape(-1,1))
    else:
        x = np.asarray(x)
        density = KDEUnivariate(x)
        if (weights is not None):      # NOTE that KDEUnivariate.fit() cannot perform Fast Fourier Transform with non-zero weights
            weights = np.asarray(weights)
            if (len(x) == 1): # NOTE that KDEUnivariate.fit() cannot cope with one-dimensional weight array
                density.fit(kernel=kernel, weights=None, bw=bandwidth, fft=False, **kwargs)
            else:
                density.fit(kernel = kernel, weights=weights, fft=False, bw=bandwidth, **kwargs)
        else:
            density.fit(kernel=kernel, bw=bandwidth, **kwargs) #when kernel='gau' fft=true
        try:
            n = len(x_grid)
            kde_est = []
            for i in x_grid:
                kde_est.append(density.evaluate(i))
            return np.asarray(kde_est)
        except:
            return density.fit(x_grid)


