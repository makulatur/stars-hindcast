# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:18:22 2014

@author: malte

Read the raw STARS and BASIS data files and compile the observations, anomalies and events 
according to the chunk structure.
"""
import pandas as pd
import os
import numpy as np

from datetime import date
from scipy.stats import scoreatpercentile as sp
from progressbar import Percentage, ProgressBar, ETA
from subprocess import call

from libraries.helper import prettyprint
from libraries.dictionaries import variable_dict, season_dict, node_selection_dict
from libraries.meta import get_directory_content, read_filename


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Options
#

# shall anomalies be calculated from daily or monthly cycles
ANOMALY_CYCLE = 'daily'

# shall anomalies be divided by the climatological std. deviation
ANOMALY_STANDARDIZED = False

# static threshold for event conversion
EVENTIZATION_THRESHOLD = 10


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions
#

def compile_obs_by_meta(data_location, variable_location, meta_file, output_location,
                        start, end, suffix=None, silence_level=0):
    """
    This function creates an observations file from raw STARS observation data,
    reading the separate data files for each station and combining them in
    a single file for easier processing. The stations to read in are
    taken from a metadata file.


    Arguments:
    ----------
    data_location : string
        Root directory of the raw data files for each station in
        the form '(id)(suffix)', e.g. 170007simsz.dat.

    variable_location : integer
        Column of the variable in the data files:

        * 3: maximum daily temperature
        * 4: mean daily temperature
        * 5: minimum daily temperature
        * 6: total daily precipitation

    meta_file : string
        Path to the metadata file.

    output_location : string
        Where the output files shall be saved.

    start : tuple or datetime object
        Date (YYYY,mm,dd) of the first observation to be included.

    end : tuple or datetime object
        Date (YYYY,mm,dd) of the last observation to be included.

    suffix : string, optional
        File name suffix of data, e.g. 'simsz.dat' or 'basz.dat',
        will be guessed from data_location if not entered.

    silence_level : int
        Inverse level of verbosity of the object


    Returns:
    --------
    output_file : string
        Name of the output data file.


    Creates:
    --------
    output_file : text file
        Text file with a row for each station containing a value for each observation
        in the time period as columns.
    """
    if isinstance(start, tuple) and isinstance(end, tuple):
        d1 = date(start[0], start[1], start[2])
        d2 = date(end[0], end[1], end[2])
    else:
        d1 = start
        d2 = end

    if silence_level < 2:
        print 'root directory for data files is %s' % data_location
        print 'reading variable %i between %s and %s' % (variable_location, d1, d2)

    # guess suffix
    if not suffix:
        if data_location.find('stars') != -1:
            suffix = 'simsz.dat'
        elif data_location.find('basis') != -1:
            suffix = 'basz.dat'
        else:
            raise ValueError('could not guess data file suffix')

    # detect which type of node selection we have done
    node_selection = (key for key, value in node_selection_dict.items() if
                      value[1] == meta_file.split('/')[-1]).next()

    if silence_level < 2:
        print 'node selection pattern %s' % node_selection_dict[node_selection][0]

    # check output location, avoid overwriting data
    output_file = (output_location + 'obs_' + node_selection + str(variable_location).zfill(2) +
                   '_' + str(d1.year) + str(d1.month).zfill(2) +
                   str(d1.day).zfill(2) + '_' + str(d2.year) +
                   str(d2.month).zfill(2) + str(d2.day).zfill(2) +
                   '.dat')

    if os.path.exists(output_file):
        raise OSError('%s is not empty' % output_file)
    else:
        if silence_level < 2:
            print 'saving to %s' % output_file

    # pandas has changed its mode of whitespace processing in read_csv
    if pd.__version__ < '0.14.1':
        cols = [1, 2, 3]
        variable_location += 1
    elif pd.__version__ >= '0.14.1':
        cols = [0, 1, 2]

    # map requested time window to data file indices, assuming complete data
    test_file = data_location + os.walk(data_location).next()[2][1]
    dates = pd.read_csv(test_file, delim_whitespace=1, usecols=cols).as_matrix()

    d = dates[:, 0].astype(int)
    m = dates[:, 1].astype(int)
    Y = dates[:, 2].astype(int)

    timeline = np.array([date(Y[i], m[i], d[i]) for i in xrange(len(Y))])

    startrow = np.where(timeline == d1)[0][0]
    endrow = np.where(timeline == d2)[0][0]
    rowcount = endrow - startrow + 1

    """
    # more elegant but assumes stars data (first and last date)
    timeline = pd.period_range(pd.datetime(1500,01,01), pd.datetime(1999,12,31))
    startrow = timeline.get_loc(d1)
    endrow = timeline.get_loc(d2)
    rowcount = endrow - startrow + 1
    """

    # read in metadata
    station_ids = np.loadtxt(meta_file, usecols=[0])

    # importing: each file is one timeseries
    filecount = len(station_ids)

    if silence_level < 2:
        print 'import of %d timeseries with %d data points each' % (filecount, rowcount)

        widgets = [Percentage(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=filecount).start()
        i = 0

    # read in raw data, write output file
    with open(output_file, 'w') as of:
        for station in station_ids:
            line = pd.read_csv(data_location + str(int(station)) + suffix, delim_whitespace=1,
                               usecols=[variable_location], skiprows=startrow,
                               nrows=rowcount).as_matrix()[:, 0]

            np.savetxt(of, (line, ), delimiter=' ', fmt='%.1f')

            if silence_level < 2:
                pbar.update(i)
                i += 1
    of.close()

    if silence_level < 2:
        pbar.finish()
        print 'observation data has been saved'

    return output_file.split('/')[-1]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# functions that convert observations into other data types
#

def events(data, p=10, threstype='l', silence_level=0):
    """
    This function creates events from observations or anomalies, using a
    threshold that is either provided directly or based on percentiles.


    Arguments:
    ----------
    data : 1D or 2D array
        Timeseries, a list of continuous daily observations, if 2D
        every column is a separate timeseries.

    p : float
        Percentile value used to compute the threshold, or the threshold
        itself if threstype = 'l'.

    threstype : string
        Type of threshold to be used, valid choices:

        * 'a': all, score at percentile of all nodes, all observations
        * 't': time, score at percentile of all nodes, per single observation
        * 'n': nodes, score at percentile of all observations, per single node
        * 'l': literal, the given value will be used as a static threshold

    silence_level : int
        Inverse level of verbosity of the object.


    Returns:
    --------
    data : 1D or 2D array
        Event timeseries, a continous list of events, if 2D every column is a
        separate timeseries
    """

    def _comp_thrs(data, percentile, threstype):
        """
        This function computes a threshold from a percentile given a threshold
            type to compute.


        Arguments:
        ----------
        data : 1D or 2D array
            The data from which the threshold is to be computed in the form
            nodes as rows, observations as columns.

        percentile : float
            Percentile value from which the threshold shall be computed.

        threstype : string
            Type of threshold to be used, valid choices:

            * 'a': all, score at percentile of all nodes, all observations
            * 't': time, score at percentile of all nodes, per single observation
            * 'n': nodes, score at percentile of all observations, per single node
            * 'l': literal, the given value will be used as a static threshold


        Returns:
        --------
        thrs : 1D array
            The calculated threshold(s).
        """
        print 'computing threshold from %dth percentile...' % percentile

        # get threshold from percentile
        if threstype == 'a':
            thrs = sp(data[data > 0], percentile)
        elif threstype == 'n':
            thrs = np.apply_along_axis(lambda x: sp(x, percentile), axis=1, arr=data)
        elif threstype == 't':
            thrs = np.apply_along_axis(lambda x: sp(x, percentile), axis=0, arr=data)
        else:
            raise ValueError('incorrect threshold type entered')

        return thrs

    data = data.copy()

    # calculate threshold
    if threstype == 'l':
        thrs = p
    else:
        thrs = _comp_thrs(data, p, threstype)

    if silence_level <= 2:
        print 'threshold(s) in mm: %s' % thrs

    if threstype in ['a', 'l']:
        data[data < thrs] = 0
        data[data >= thrs] = 1
    elif threstype in ['n']:
        for i in range(data.shape[0]):
            row = data[i, :]
            row[row < thrs[i]] = 0
            row[row >= thrs[i]] = 1
    elif threstype in ['t']:
        for i in range(data.shape[1]):
            col = data[:, i]
            col[col < thrs[i]] = 0
            col[col >= thrs[i]] = 1

    return data.astype('int8')


def events_file_to_file(input_file, output_location, p=10, threstype='l', silence_level=0):
    """
    This function creates events from observations or anomalies, using a
    threshold that is either provided directly or based on percentiles.


    Arguments:
    ----------
    input_file : string
        Path to the input file, e.g. an observations file or an
        anomalies file.

    output_location : string
        Path to the folder in which the output file shall be created.

    p : float
        Percentile value used to compute the threshold, or the threshold
        itself if threstype = 'l'.

    threstype : string
        Type of threshold to be used, valid choices:

        * 'a': all, score at percentile of all nodes, all observations
        * 't': time, score at percentile of all nodes, per single observation
        * 'n': nodes, score at percentile of all observations, per single node
        * 'l': literal, the given value will be used as a static threshold

    silence_level : int
        Inverse level of verbosity of the object.


    Returns:
    --------
    output_file : string
        Name of the output file.


    Creates:
    --------
    output_file : text file
        A file containing the events calculated from observations or anomalies.
    """
    output_file = ('evs_' + threstype + str(p) + '_' +
                   input_file.split('/')[-1][:-4] + '.dat')

    if os.path.exists(output_file):
        raise IOError('the file %s already exists, aborting...' % output_file)

    # creating events file
    if silence_level < 2:
        print 'creating events file...'

    with open(output_location + output_file, 'w') as of:
        data = [[float(i) for i in line.split()] for line in open(input_file)]

        evs = events(np.array(data), p=p, threstype=threstype, silence_level=silence_level)

        np.savetxt(of, evs, delimiter=' ', fmt='%i')
    of.close()

    if silence_level < 2:
        print 'events have been saved'

    return output_file


def anomalies(data, start, end, season='all', standardized=False, cycle='daily'):
    """
    This function calculates anomalies from continous daily data using its
    climatology, i.e. the long-term average of the data.

    The anomalies are defined as

    .. math::

        a = d - m

    where a = anomaly, d = daily value, c = climatological mean.

    The standardized anomalies are

    .. math::

        a = (d - m)/s

    where s = std. deviation of climatology.


    Arguments:
    ----------
    data : 1D or 2D array
        Timeseries, a list of continuous daily observations, if 2D
        every row is a separate timeseries.

    start : tuple or datetime object
        Date (YYYY,mm,dd) of the first observation to be included.

    end : tuple or datetime object
        Date (YYYY,mm,dd) of the last observation to be included.

    season : string
        If the data only includes seasonal values, either 'winter' for
        dec, jan, feb, or 'summer' for jun, jul, aug, or 'all' for
        complete data .

    standardized : boolean
        If True, anomalies will be divided by the climatological std deviation.

    cycle : string
        If 'm' or 'monthly', monthly climatologies will be used, i.e. the mean
        values over each month, if 'd' or 'daily', daily climatologies will be used


    Returns:
    --------
    anoms : 1D or 2D array
        The anomaly data in the same shape as data.
    """
    if isinstance(start, tuple) and isinstance(end, tuple):
        d1 = date(start[0], start[1], start[2])
        d2 = date(end[0], end[1], end[2])
    else:
        d1 = start
        d2 = end

    rng = pd.period_range(d1, d2, freq='D')  # works with longer time spans

    if season != 'all':
        months = season_dict[season][1]

        rng = rng[(rng.month == months[0]) | (rng.month == months[1]) | (rng.month == months[2])]

    # to use the groupby feature of the pandas dataframe,
    # we feed the transposed data so that the dates constitute the index
    data = data.T

    if data.shape[0] != rng.shape[0]:
        raise ValueError('data length (%i observations) and date range (%i days) do not match' % (
            len(data), len(rng)))

    dts = pd.DataFrame(data=data, index=rng)

    if cycle in ['d', 'daily']:
        grouping = [dts.index.month, dts.index.day]
    elif cycle in ['m', 'monthly']:
        grouping = dts.index.month

    if standardized:
        anoms = dts.groupby(grouping).transform(lambda x: (x - x.mean()) / x.std()).values
    else:
        anoms = dts.groupby(grouping).transform(lambda x: x - x.mean()).values

    return anoms.T


def anomalies_file_to_file(input_file, output_location, start, end, season='all',
                           standardized=False, cycle='monthly', silence_level=0):
    """
    This function computes anomalies from observation data. Anomalies are used
    instead of raw observations in order to account for seasonal
    variations in, for example, temperature distributions.


    Arguments:
    ----------

    input_file : string
        Path to the input file, e.g. an observations file or an events file

    output_location : string
        Path to the folder in which the output file shall be created

    start : tuple or datetime object
        Date (YYYY,mm,dd) of the first observation to be included.

    end : tuple or datetime object
        Date (YYYY,mm,dd) of the last observation to be included.

    season : string
        If the data only includes seasonal values, either 'winter' for
        dec, jan, feb, or 'summer' for jun, jul, aug, or 'all' for
        complete data.

    standardized : boolean
        If True, anomalies will be divided by the climatological std deviation.

    cycle : string
        If 'm' or 'monthly', monthly climatologies will be used, i.e. the mean
        values over each month, if 'd' or 'daily', daily climatologies will be used

    silence_level : int
        Inverse level of verbosity of the object.


    Returns:
    --------
    output_file : string
        Name of the output file.


    Creates:
    --------
    output_file : text file
        Anomalies file containing the climatological anomaly values calculated for
        all observations in the input_file.
    """
    # check output file
    output_file = 'ano_' + input_file.split('/')[-1]

    if os.path.exists(output_location + output_file):
        raise IOError('the file %s already exists, aborting...' % output_file)

    # creating anomalies file
    if silence_level < 2:
        print 'creating anomalies file...'

    with open(output_location + output_file, 'w') as of:
        data = [[float(i) for i in line.split()] for line in open(input_file)]

        anoms = anomalies(np.array(data), start, end, season=season,
                          standardized=standardized, cycle=cycle)

        np.savetxt(of, anoms, delimiter=' ', fmt='%.2f')
    of.close()

    if silence_level < 2:
        print 'anomalies have been saved'

    return output_file


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# functions that extract or slice a subset of some data
#

def season_extract(start, end, season, data=None):
    """
    Computes which columns contain information belonging to particular seasons. If data is
    provided, the relevant chunks of data are returned. If no data is provided, the columns
    are returned.

    .. note::
        If no data is provided, the function returns the column indices of the requested data
        slice. This is useful as input for a file reader method, e.g. if the goal is to read only
        those data points that are necessary, avoiding to read the whole file.

    Arguments:
    ----------
    start : tuple or datetime object
        Date (YYYY,mm,dd) of the first observation to be included.

    end : tuple or datetime object
        Date (YYYY,mm,dd) of the last observation to be included.

    season : string
        If the data only includes seasonal values, either 'winter' for
        dec, jan, feb, or 'summer' for jun, jul, aug, or 'all' for
        complete data.

    data : 1D or 2D array, optional
        Timeseries, a list of continuous daily observations, if 2D
        every column is a separate timeseries.


    Returns:
    --------
    if data is provided:

    data : 1D or 2D array
        Timeseries, a list of daily observations, only including the selected
        months.

    if no data is provided:

    cols : 1D or 2D array
        The columns of the data pertaining to the selected months, useful for
        file reading methods.
    """
    if isinstance(start, tuple) and isinstance(end, tuple):
        d1 = date(start[0], start[1], start[2])
        d2 = date(end[0], end[1], end[2])
    else:
        d1 = start
        d2 = end

    rng = pd.period_range(d1, d2, freq='D')  # works with longer time spans

    if data is not None:
        if data.ndim == 2:
            if data.shape[1] != len(rng):
                raise ValueError('data length (%i obs) and date range (%i days) do not match' %
                                 (data.shape[1], len(rng)))
        elif data.ndim == 1:
            if data.shape[0] != len(rng):
                raise ValueError('data length (%i obs) and date range (%i days) do not match' %
                                 (data.shape[0], len(rng)))
        else:
            raise TypeError('only 1D and 2D data arrays are supported')

    # which season to extract
    months = season_dict[season][1]

    dts = pd.Series(xrange(len(rng)), rng)

    cols = dts[(rng.month == months[0]) | (
        rng.month == months[1]) | (rng.month == months[2])].values

    if data is not None:
        if data.ndim == 2:
            return data[:, cols]
        else:
            return data[cols]
    else:
        return cols


def season_extract_file_to_file(input_file, output_location, start, end, season, silence_level=0):
    """
    This function extracts the data from the summer or winter season only so
    that events and anomalies can be analyzed seperately.


    Arguments:
    ----------
    input_file : string
        Path to the input file, e.g. an observations file or an events file

    output_location : string
        Path to the folder in which the output file shall be created

    start : tuple or datetime object
        Date (YYYY,mm,dd) of the first observation to be included.

    end : tuple or datetime object
        Date (YYYY,mm,dd) of the last observation to be included.

    season : string
        If the data only includes seasonal values, either 'winter' for
        dec, jan, feb, or 'summer' for jun, jul, aug, or 'all' for
        complete data.

    silence_level : int
        Inverse level of verbosity of the object.


    Returns:
    --------
    output_file : string
        Name of the output file.


    Creates:
    --------
    output_file : text file
        File of the same type as input_file, but containing only a subset
        of the datapoints that corresponds to the chosen season.
    """
    data_desc = read_filename(input_file)

    # check output file
    output_file = (output_location + input_file.split('/')[-1][:-4] + '_' + season +
                   '.dat')

    if os.path.exists(output_file):
        raise IOError('the file %s already exists, aborting...' % output_file)

    cols = season_extract(start, end, season)

    if silence_level <= 2:
        print 'extracting %s season data...' % season

    # iterator to reduce memory consumption, read relevant columns of chunk
    chunksize = 250
    iter_csv = pd.read_csv(input_file, delim_whitespace=1, usecols=cols,
                           iterator=True, chunksize=chunksize, header=None)

    with open(output_file, 'w') as of:
        for chunk in iter_csv:
            np.savetxt(of, chunk.as_matrix(), fmt=data_desc.format)

    of.close()

    if silence_level <= 2:
        print 'saved seasonal data to %s' % output_file

    return output_file.split('/')[-1]

def sort_raw_data_into_chunks(variable_location, base_output_location, chunklength,
                              meta_file, stars_data_location, basis_data_location,
                              season='both', dataset='both'):
    """
    This is a do-all function that reads the raw files and compiles the o, compile
    and calculate any derivative files.

    For clarity there is a folder structure created that puts all files relevant to a certain time
    chunk in a subfolder, e.g. /out/1900-1950/.

    The function uses stars_meta.get_directory_content() to read the names of files present that
    adhere to the standard naming convention, without doing any consistency checks on the files'
    content. It skips the calculation of those files that are already present.


    Arguments:
    ----------
    variable_location : integer
        Column of the variable in the data files:

        * 3: maximum daily temperature
        * 4: mean daily temperature
        * 5: minimum daily temperature
        * 6: total daily precipitation

    base_output_location : string
        Base location of the output structure.    

    chunklength : int
        Length of time chunks in years.

    meta_file : str
        Location of the meta data file.

    stars_data_location : str
        Location of the raw STARS data.
        
    basis_data_location : str
        Location of the raw BASIS data.

    season : string
        Which season to run, either summer, winter, or both.

    dataset : string
        On which dataset to run, either stars, basis, or both.
    """
    variable_short_name = variable_dict[variable_location][1]
 
    folder = base_output_location + str(chunklength) + 'y/'
    if not os.path.exists(folder):
        res = call(['mkdir \'' + folder + '\''], shell=True)
        if res == 0:
            print 'folder %s created' % folder
        else:
            raise IOError('folder %s does not exist and could not be created' % (folder))
            
    folder = base_output_location + str(chunklength) + 'y/' + variable_short_name + '/'
    if not os.path.exists(folder):
        res = call(['mkdir \'' + folder + '\''], shell=True)
        if res == 0:
            print 'folder %s created' % folder
        else:
            raise IOError('folder %s does not exist and could not be created' % (folder))

    output_location = base_output_location + str(chunklength) + 'y/' + variable_short_name + '/'

    if season == 'both':
        seasons = ['summer', 'winter']
    else:
        seasons = [season]
        
    if dataset == 'both':
        datasets = ['stars', 'basis']
    else:
        datasets = [dataset]

    # main loop
    for dataset in datasets:
        if dataset == 'stars':
            startchunks = range(1500, 1999, chunklength)
            data_location = stars_data_location

        elif dataset == 'basis':
            startchunks = range(1951, 1999, chunklength)
            data_location = basis_data_location

        for y1 in startchunks:
            y2 = y1 + chunklength

            if y2 >= 2000:
                y2 = 1999

            prettyprint('now processing %s %s in %s data from %i to %i' % (season,
                variable_short_name, dataset.upper(), y1, y2), 1)

            start = (y1, 6, 1)
            end = (y2, 2, pd.period_range(y2, y2 + 1)
                   [pd.period_range(y2, y2 + 1).month == 2][-1].day)

            folder = str(y1) + '-' + str(y2) + '-' + dataset + '/'

            if not os.path.exists(output_location + folder):
                res = call(['mkdir \'' + output_location + folder[:-1] + '\''], shell=True)
                if res == 0:
                    print 'folder %s created' % folder[:-1]
                else:
                    raise IOError('folder %s does not exist and could not be created' % (
                        output_location + folder[:-1]))

            # get information about files already present
            dco = get_directory_content(output_location + folder, silence_level=2)
            print dco

            # compile observations
            if not hasattr(dco, 'obs'):
                obs_file = compile_obs_by_meta(
                    data_location, variable_location, meta_file, dco.folder, start, end)

                setattr(dco, 'obs', obs_file)

            # prec only: calculate events
            if variable_location in [6]:
                if not hasattr(dco, 'evs'):
                    evs_file = events_file_to_file(
                        dco.folder + getattr(dco, 'obs'), dco.folder,
                        p=EVENTIZATION_THRESHOLD)

                    setattr(dco, 'evs', evs_file)     

            # temp only: calculate anomalies
            if variable_location in [3, 4, 5]:
                if not hasattr(dco, 'ano'):
                    ano_file = anomalies_file_to_file(
                            dco.folder + getattr(dco, 'obs'), dco.folder,
                            start, end, season='all', standardized=ANOMALY_STANDARDIZED, 
                            cycle=ANOMALY_CYCLE)

                    setattr(dco, 'ano', ano_file)   
                    
            for season in seasons:
                if not hasattr(dco, season_dict[season][0] + 'obs'):
                    seasonal_obs_file = season_extract_file_to_file(
                        dco.folder + getattr(dco, 'obs'), dco.folder, start, end, season)

                    setattr(dco, season_dict[season][0] + 'obs', seasonal_obs_file)

                # prec only: calculate events
                if variable_location in [6]:
                    if not hasattr(dco, season_dict[season][0] + 'evs'):
                        evs_file = events_file_to_file(
                            dco.folder + getattr(dco, season_dict[season][0] + 'obs'), 
                            dco.folder, p=EVENTIZATION_THRESHOLD)
    
                        setattr(dco, season_dict[season][0] + 'evs', evs_file)

                # temp only: calculate anomalies
                if variable_location in [3, 4, 5]:
                    if not hasattr(dco, season_dict[season][0] + 'ano'):
                        ano_file = anomalies_file_to_file(
                            dco.folder + getattr(dco, season_dict[season][0] + 'obs'), 
                            dco.folder, start, end, season, standardized=ANOMALY_STANDARDIZED, 
                            cycle=ANOMALY_CYCLE)
    
                        setattr(dco, season_dict[season][0] + 'ano', ano_file)       


if __name__ == '__main__':
    pass

