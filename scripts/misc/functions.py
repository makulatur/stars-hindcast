# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:54:56 2016

@author: malte
"""
import pandas as pd

from datetime import date

from libraries.dictionaries import season_dict


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions
#

def get_obs(data_location, station_id, variable_location, start, end, season='all',
            return_period_index=False, return_full_dataframe=True, suffix=None):
    """
    Function to get data from a particular station between start and end, with
    the possibility of only extracting data from particular months.


    Arguments:
    ----------
    data_location : string
        Root directory of the raw data files for each station in
        the form '(id)(suffix)', e.g. 170007simsz.dat.

    station_id : integer
        Station from which the data is to be extracted.

    variable_location : integer
        Column of the variable in the data files:

        * 3: maximum daily temperature
        * 4: mean daily temperature
        * 5: minimum daily temperature
        * 6: total daily precipitation

    start : tuple or datetime object
        Date (YYYY,mm,dd) of the first observation to be included.

    end : tuple or datetime object
        Date (YYYY,mm,dd) of the last observation to be included.

    season : string
        If the data only includes seasonal values, either 'winter' for
        dec, jan, feb, or 'summer' for jun, jul, aug, or 'all' for
        complete data.

    return_period_index : bool
        Whether a period index is returned with the date index of the
        timeseries in addition to the data.

    return_full_dataframe : bool
        Whether the full dataframe for this station shall be returned.
        In this case {start}, {end}, {season} and {return_period_index}
        are ignored.

    suffix : string, optional
        File name suffix of data, e.g. 'simsz.dat' or 'basz.dat',
        will be guessed from data_location if not entered.


    Returns:
    --------
    if return_period_index = False:

    data : 1D array
        The requested data from start to end, only including selected
        months.

    if return_period_index = True:

    rng : PeriodIndex
        The period index of the requested data.

    data : 1D array
        The requested data from start to end, only including selected
        months.

    if return_full_dataframe = True:

    data : DataFrame
        DataFrame containing the full data of the variable available
        in the source file.
    """
    if isinstance(start, tuple) and isinstance(end, tuple):
        d1 = date(start[0], start[1], start[2])
        d2 = date(end[0], end[1], end[2])
    else:
        d1 = start
        d2 = end

    # guess suffix
    if not suffix:
        if data_location.find('stars') != -1:
            suffix = 'simsz.dat'
        elif data_location.find('basis') != -1:
            suffix = 'basz.dat'
        else:
            raise ValueError('could not guess data file suffix')

    months = season_dict[season][1]

    # pandas has changed its mode of whitespace processing in read_csv
    if pd.__version__ < '0.14.1':
        variable_location += 1
        datecols = [1, 2, 3]
    elif pd.__version__ >= '0.14.1':
        datecols = [0, 1, 2]

    def parser(d, m, y):
        d, m, y = [int(x) for x in [d, m, y]]
        dt = date(y, m, d)
        return dt

    if type(variable_location) == int:
        variable_location = [variable_location]

    # read time and data columns
    data = pd.read_csv(data_location + str(station_id) + suffix,
                       delim_whitespace=1, usecols=datecols + variable_location,
                       parse_dates={'datetime': datecols},
                       date_parser=parser, index_col='datetime')

    if return_full_dataframe:
        return data

    # slice requested window
    data = data.ix[d1:d2].as_matrix()[:, 0]

    rng = pd.period_range(d1, d2, freq='D')

    if season != 'all':
        ts = pd.Series(data, index=rng)

        rng = rng[(rng.month == months[0]) | (rng.month == months[1]) | (rng.month == months[2])]

        data = ts[rng].as_matrix()

    if return_period_index:
        return rng, data
    else:
        return data