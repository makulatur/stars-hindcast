# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:52:28 2015

@author: malte

A module providing functions that calculate and show kappa statistics.
"""
import numpy as np

from libraries.dictionaries import season_dict, measures_dict, network_dict, variable_dict
from libraries.helper import load_all_objects


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Options
#

# standard percentile values for data buckets
BUCKET_PERCENTILES = [1, 10, 25, 75, 90, 99, 100]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions
#


def categorize_data(data1, data2, global_buckets):
    """
    Categorizes observations in data1 and data2 into some provided global buckets
    (categories), or generates new buckets from percentile values over the combined data1 and
    data2 data.

    Arguments:
    ----------
    data1 : list or 1D array
        A list of values, e.g. a vector containing the degree of every node on a network.

    data2 : list or 1D array
        Another list of values that is to be compared to data1.

    global_buckets : array-like or False, optional
        The set of global buckets (or categories) to use, if False two sets will be calculated
        from quantiles specified in `BUCKET_PERCENTILES` of `data1` and `data2`.


    Returns:
    --------
    cat_data1 : list or 1D array
        A list of categorized data, i.e. an array of shape(data1) that contains the appropriate
        category (or bucket) for each data point in data1.

    cat_data2 : list or 1D array
        A list of categorized data.
    """
    # make sure we have arrays
    data1 = np.array(data1)
    data2 = np.array(data2)

    # either we were supplied the categories, or we compute them from quantiles
    if type(global_buckets) == np.ndarray:
        buckets1 = buckets2 = global_buckets
    elif global_buckets is False:
        buckets1 = np.percentile(data1, BUCKET_PERCENTILES)
        buckets2 = np.percentile(data2, BUCKET_PERCENTILES)

    # categorizing the data
    cat_data1 = np.digitize(data1, buckets1)
    cat_data2 = np.digitize(data2, buckets2)

    return cat_data1, cat_data2


def compute_crosstab(cat_data1, cat_data2):
    """
    Computes the crosstab from categorized data.

    .. note::
    This is also known as a confusion matrix. An element :math:`C_{i, j}` is equal to the
    number of obserations that, according to one rater, belong to category :math:`i`, but
    were rated in category :math:`j` by the other.

    Arguments:
    ----------
    cat_data1 : list or 1D array
        A list of categorized data.

    cat_data2 : list or 1D array
        A list of categorized data.


    Returns:
    --------
    ct : 2D array
        The square crosstab (or contingency table) of the data provided.
    """
    # number of categories
    n_cats = max(max(cat_data1), max(cat_data2)) + 1

    # fill crosstab
    ct = np.zeros((n_cats, n_cats), dtype=np.int)

    data = np.vstack((cat_data1, cat_data2))

    for i in range(data.shape[1]):
        ct[data[0, i], data[1, i]] += 1

    return ct


def calculate_kappa(crosstab, weights, fuzzy=False):
    """
    Calculates Cohen's Kappa as well as the complementary measures Kappa_location and
    Kappa_histogram.


    Arguments:
    ----------
    crosstab : 2D array
        A square crosstab (or contingency table) of some data.

    weights : None, string or array
        If any and which weights shall be used. Available options:

        * None: no weights will be used.
        * 'linear': linear weights (Cohen 1968)
        * 'quadratic': quadratic weights (Fleiss and Cohen 1973)
        * 2D array of (n_cats * n_cats): custom weights

    fuzzy : bool
        Whether fuzzy Kappa shall be calculated, for details see below. Experimental.


    Returns:
    --------
    out : tuple of three floats
        Kappa, Kappa_location, Kappa_histogram
    """
    if crosstab.shape[0] != crosstab.shape[1]:
        raise TypeError('confusion matrix must be square')

    n_cats = len(crosstab)

    # construct weights array
    if isinstance(weights, str):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((n_cats, n_cats))
        for i in range(n_cats):
            for j in range(n_cats):
                diff = abs(i - j)
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('invalid weight scheme')

    # sum of elements
    n = float(crosstab.sum())

    # observed proportion in agreement
    ob = crosstab / n
    po = 1.0 - np.sum(ob * weights)

    # margins
    row_sum = ob.sum(axis=1)
    col_sum = ob.sum(axis=0)

    """
    # calculation as defined by Pontius 2000 and Hagen 2003
    
    NQNL = 1 / float(n_cats)
    NQPL = np.sum(np.minimum(NQNL * np.ones(row_sum.shape), row_sum))
    MQNL = np.sum(row_sum * col_sum)

    kstd = (po - MQNL) / float(1 - MQNL)

    MQPL = np.sum(np.minimum(col_sum, row_sum))

    kquanHagen = (MQPL - MQNL) / float(1 - MQNL)

    PQNL = np.sum(row_sum ** 2)

    kloc = (po - MQNL) / float(MQPL - MQNL)

    NQML = NQNL + kloc * (NQPL - NQNL)
    PQML = PQNL + kloc * (1 - PQNL)

    kno = (po - MQNL) / float(1 - NQNL)
    kquanPontius = (po - NQML) / float(PQML - NQML)

    print 'kstd         %.8f' % kstd
    print 'kno          %.8f' % kno
    print 'kloc         %.8f' % kloc
    print 'kquanPontius %.8f' % kquanPontius
    print 'kquanHagen   %.8f' % kquanHagen
    """

    # maximum possible proportion in agreement, given marginal proportions
    po_max = np.sum(np.min(zip(row_sum, col_sum), axis=1))

    # expected proportion in agreement, given chance
    ex = np.outer(row_sum, col_sum)
    pe = 1.0 - np.sum(ex * weights)

    # if all weights are zero, disagreements do not matter
    if np.count_nonzero(weights):
        kappa = (po - pe) / (1 - pe)
        kappa_location = (po - pe) / (po_max - pe)
        kappa_histogram = (po_max - pe) / (1 - pe)
    else:
        kappa = 1.0
        kappa_location = 1.0
        kappa_histogram = 1.0

    return kappa, kappa_location, kappa_histogram


def trim_data(data, target_realizations, silence_level=2):
    """
    Trim the array of data arrays generated by `stars_chunkwise.load_all_objects()` to ensure that 
    only the last `target_realizations` realizations are considered and the data array is handed
    over in the correct way:
    
        ``data.shape = (num_chunks, target_realizations, num_nodes)``
    
    If the data is truncated to ensure proper realizations across all chunks, the first entries
    are truncated, i.e. the most recent results are taken.
    
    
    Arguments:
    ----------
    data : numpy array
        Data array as returned by `stars_chunkwise.load_all_objects()` if called on network measure
        files.
    
    target_realizations : int
        Target number of realizations to enforce across all chunks. If fewer are present in one
        chunk, the data arrays will be truncated so that all have this lower number.
        
    silence_level : int
        Inverse level of verbosity of the object.
        
    
    Returns:
    --------
    data : 3d numpy array
        The data in the desired format ``data.shape = (n_chunks, target_realizations, n_nodes)``.
    """
    if data.ndim == 1:
        min_realizations = min([np.atleast_2d(element).shape[0] for element in data])
        max_realizations = max([np.atleast_2d(element).shape[0] for element in data])

        if min_realizations == 0:
            raise ValueError('at least one chunk is missing all data: %s' % [
                np.atleast_2d(element).shape[0] for element in data])
        elif min_realizations < target_realizations:
            if silence_level < 3:
                txt = ('target number of surrogates has not been met.\n' +
                       'only %i realizations available for all chunks.' % min_realizations)
                print txt

        if min_realizations != max_realizations:
            if min_realizations > target_realizations:
                min_realizations = target_realizations

            new_data = np.zeros((data.shape[0], min_realizations, data[0].shape[1]))

            for i in xrange(data.shape[0]):
                new_data[i, :, :] = np.atleast_2d(data[i])[-min_realizations:, :]

            data = new_data

            if silence_level < 3:
                print 'data has been truncated to %i realizations.' % min_realizations
        else:
            raise TypeError('data array has only one dimension, expected 3 dimensions.')

    elif data.ndim == 2:
        data = data.reshape((data.shape[0], 1, data.shape[1]))

    if data.shape[1] > target_realizations:        
        data = data[:, -target_realizations:, :]
        
        if silence_level < 3:
            print 'data has been truncated to %i realizations.' % target_realizations

    return data


def categorize_and_compute_kappa(data1, data2, weights=None, global_buckets=False, fuzzy=False):
    """
    Calculates Cohen's Kappa for two sets of observations of a particular value, e.g. a network
    measure calculated for each node in a network with optional weighting.

    The first step is to 'cluster' the observations. This is done by binning them according to
    their quantile value with respect to some global quantiles (if an array of global_buckets is
    given), or with respect to quantiles generated from data1 and data2 (if
    global_buckets=False). The buckets are the categories for our 'judgements' in Cohen's Kappa.

    Then, the arrays containing the category for every observation are used to calculate Cohen's
    Kappa, as well as Kappa_location and Kappa_histogram.

    .. note::
        :math:`\kappa_{histogram} = \kappa_{max}` and :math:`\kappa_{location} = \kappa /
        \kappa_{max}`, where :math:`\kappa_{max}` is the maximum Kappa possible given the
        observed frequencies.

    .. note::
        The classic Kappa only distinguishes between agreement and non-agreement. If the
        categories have, however, some ordinal characteristics, we can weight the non-agreements
        by distance. This necessitates, of course, an ordered list of categories, like the one we
        generate from data quantiles.

    .. note::
        We use np.digitize() to bin all values. This function either includes the right intervall
        limit or it doesn't, there is no way of mixing both behaviours as Aljoscha seems to have
        done in his diploma thesis (p. 62).


    Arguments:
    ----------
    data1 : list or 1D array
        A list of values, e.g. a vector containing the degree of every node on a network.

    data2 : list or 1D array
        Another list of values that is to be compared to data1.

    weights : None, string or array
        If any and which weights shall be used. Available options:

        * None: no weights will be used.
        * 'linear': linear weights (Cohen 1968)
        * 'quadratic': quadratic weights (Fleiss and Cohen 1973)
        * 2D array of (n_cats * n_cats): custom weights

    global_buckets : array-like or False, optional
        The set of global buckets (or categories) to use, if False two sets will be calculated
        from quantiles specified in `BUCKET_PERCENTILES` of `data1` and `data2`.
        
    fuzzy : bool
        Whether fuzzy Kappa shall be calculated, for details see below. Experimental.


    Returns:
    --------
    out : tuple of three floats
        Kappa, Kappa_location, Kappa_histogram
    """
    # categorize data
    cat_data1, cat_data2 = categorize_data(data1, data2, global_buckets=global_buckets)

    # compute crosstab
    ct = compute_crosstab(cat_data1, cat_data2)

    # calculate kappa
    kappa = calculate_kappa(ct, weights=weights)

    return kappa


def calculate_kappa_matrix(data, global_buckets=False, weights=None):
    """
    Calculates a matrix containing the Kappa values between all observation sets supplied. Each
    set is compared to each other set.


    Arguments:
    ----------
    data : 2D array
        Array with observation sets as rows, each row will be compared to each other row.

    global_buckets : bool
        Whether a global set of buckets shall be used for all kappa (if True), these will be
        calculated from quantiles over all data sets, or a new sets of buckets will be calculated
        for each pair of data vectors (if False).

    weights : None, string or array
        If any and which weights shall be used. Available options:

        * None: no weights will be used.
        * 'linear': linear weights (Cohen 1968)
        * 'quadratic': quadratic weights (Fleiss and Cohen 1973)
        * 2D array of (n_cats * n_cats): custom weights


    Returns:
    --------
    matrix : 3D array
        Three symmetric matrices containing kappa values between the observation sets:

             * matrix[0,:,:] is kappa
             * matrix[1,:,:] is kappa_location
             * matrix[2,:,:] is kappa_hist
    """
    data = np.array(data)

    if type(global_buckets) == np.ndarray:
        pass
    elif global_buckets is True:
        global_buckets = np.percentile(data, BUCKET_PERCENTILES)
    else:
        global_buckets = False

    n = len(data)
    matrix = np.zeros((3, n, n))
    matrix[0, :, :] = matrix[1, :, :] = matrix[2, :, :] = np.eye(n)

    # fill matrix
    for i in range(n):
        for j in range(i):
            matrix[:, i, j] = matrix[:, j, i] = categorize_and_compute_kappa(data[i], data[j],
                                            global_buckets=global_buckets,
                                            weights=weights)

    return matrix


def calc_kappa_from_measures(variable_location, dir_contents, link_density, shuffle_type, measure,
                             season, target_realizations=1, percentile=False, average=False, 
                             global_buckets=True, weights=None, silence_level=0):
    """
    Load network measures for original or surrogate networks, as generated by stars_cluster.py,
    for each node and each time chunk. :math:`\kappa`-values are calculated between all chunks,
    including the original data, for all realizations. If desired, a percentile value or the
    average is computed over all realizations and returned.

    This function can be used on the original dataset if `shuffle_type` is 'noshu' and
    `percentile` is set to False.

    The function truncates data so that all chunks are processed with the same number of
    realizations, using the most recent realizations available. A hard limit of 
    {target_realizations} maximum realizations per chunk is enforced.


    Arguments:
    ----------
    variable_location : integer
        Column of the variable in the data files:

        * 3: maximum daily temperature
        * 4: mean daily temperature
        * 5: minimum daily temperature
        * 6: total daily precipitation

    dir_contents : list of DirectoryContent objects
        A list containing DirectoryContent objects.

    link_density : int
        Link density of the networks, 0 < LD < 1.

    shuffle_type : string
        What type of surrogate data shall be used.

    measure : string
        What measure to use, e.g. ndg or icr.

    season : string
        Which season to use, either summer, winter, or all.
        
    target_realizations : int
        Target number of realizations that will be worked with. Excess realizations are truncated.

    percentile : False or int
        What percentile is taken from the kappa values over all realizations.

    average : bool
        Instead of the percentile, the average can be returned if this is set to True.

    global_buckets : bool or 'static'
        Whether a global set of buckets shall be used for all kappa (if True), these will be
        calculated from quantiles over all data sets, or a new sets of buckets will be calculated
        for each pair of data vectors (if False).
        
        A static vector of buckets is calculated for 'static'.

    weights : None, string or array
        If any and which weights shall be used. Available options:

        * None: no weights will be used.
        * 'linear': linear weights (Cohen 1968)
        * 'quadratic': quadratic weights (Fleiss and Cohen 1973)
        * 2D array of (n_cats * n_cats): custom weights

    silence_level : int
        Inverse level of verbosity.


    Returns:
    --------
    matrix : 4D array
        If either `average` or `percentage`:

        Two times three symmetric matrices containing :math:`\kappa`-values between the observation
        sets. The first set of matrices contains the top `percentile` or the mean values calculated
        over all realizations.

            * matrix[0, ...] is the percentile/average value
            * matrix[1, ...] is the std. deviation values

        If neither `average` or `percentage`:

        Realizations times three symmetric matrices containing :math:`\kappa`-values between the
        observation sets.

            * matrix[n, ...] is the :math:`\kappa`-matrix for the n-th realization

        A :math:`\kappa`-matrix is filled as follows:

            * matrix[:, 0, :, :] is :math:`\kappa`
            * matrix[:, 1, :, :] is :math:`\kappa_{loc}`
            * matrix[:, 2, :, :] is :math:`\kappa_{his}`
    """
    if bool(percentile) and bool(average):
        raise ValueError('only one, percentile or average, can be set')
    
    # load the network measures from files
    network_type = variable_dict[variable_location][2][0]
    
    filekey = (season_dict[season][0] + 'nm_' + measures_dict[measure][0] + '_' +
        network_dict[network_type][0] + shuffle_type + '_d' + 
        str(int(link_density * 100)).zfill(2))

    data = load_all_objects(dir_contents, filekey, silence_level=2)

    data = trim_data(data, target_realizations)

    # static option for buckets
    if global_buckets == 'static':
        vmax = data.max()
        vmin = data.min()
        global_buckets = np.linspace(vmin, vmax, 7)

    # get dimensions and initialize arrays
    num_files = data.shape[0]
    realizations = data.shape[1]
    A = np.zeros((realizations, 3, num_files, num_files))

    if silence_level < 3 and realizations > 100:
        from progressbar import Percentage, ProgressBar, ETA
        widgets = [Percentage(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=realizations).start()

    # compute the kappa matrix for each realization
    for i in xrange(realizations):
        A[i, :, :, :] = calculate_kappa_matrix(data[:, i, :], global_buckets=global_buckets,
            weights=weights)

        if silence_level < 3 and realizations > 100:
            pbar.update(i)

    if silence_level < 3 and realizations > 100:
        pbar.finish()

    # return the percentile or mean values and std deviation
    if realizations == 1:
        matrix = A
    elif average:
        matrix = np.zeros((2, 3, num_files, num_files))
        matrix[0, ...] = np.mean(A, axis=0)
        matrix[1, ...] = np.std(A, axis=0)
    elif percentile:
        matrix = np.zeros((2, 3, num_files, num_files))
        matrix[0, ...] = np.percentile(A, percentile, axis=0)
        matrix[1, ...] = np.std(A, axis=0)
    else:
        matrix = A

    return matrix


if __name__ == '__main__':
    pass
