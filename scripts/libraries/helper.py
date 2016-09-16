# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:18:22 2014

@author: malte

Read the raw STARS and BASIS data files and compile the observations, anomalies and events 
according to the chunk structure.
"""
import sys

import numpy as np
import pandas as pd

from libraries.dictionaries import season_dict, variable_dict
from libraries.meta import locate_all_files


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions
#

def initialize(variable_location, base_output_location, chunklength=50, get_dir_contents=True, 
               silence_level=0):
    """
    Get the long and short names, output locations and dir_contents object for the variable at
    {variable_location}.


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
    
    chunklength : integer
        The size of the time chunks.
    
    get_dir_contents : bool
        Whether to scan for and return the dir_contents list or just return an empty list, if the
        objects are not needed. Setting of False lets the function run significantly faster.

    silence_level : int
        Inverse level of verbosity.


    Returns:
    --------
    variable_long_name : string
        Long name of the variable.

    variable_short_name : string
        Abbreviated name of the variable.

    output_location : string
        Path to the folder that contains output relating to this variable.

    if get_dir_contents:
    
        dir_contents : list of DirectoryContent object
            A list of objects containing as attributes the names of all derivative
            files found in the directory that was searched.
    
    if not get_dir_contents:
    
        dir contents : list
            An empty list.
    """    
    variable_long_name = variable_dict[variable_location][0]
    variable_short_name = variable_dict[variable_location][1]

    output_location = base_output_location + str(chunklength) + 'y/' + variable_short_name + '/'
    
    if get_dir_contents:
        dir_contents = locate_all_files(output_location, silence_level=2)
    else:
        dir_contents = []

    if silence_level < 2:
        prettyprint('initialized for %s at chunklength %i' % (variable_long_name, chunklength), 1)

    return variable_long_name, variable_short_name, output_location, dir_contents


def create_date_index(start, end, season):
    """
    Creates a pd.period_range index between {start} and {end}, accounting for {seasonal} values.


    Arguments:
    ----------
    start : datetime object
        Start of the period.

    end : datetime object
        End of the period.

    season : string
        What seasons to include, i.e. winter, summer, or all.


    Returns:
    --------
    date_index : PeriodIndex
        Date index of all relevant dates.
    """
    date_index = pd.period_range(start, end)

    if season != 'all':
        months = season_dict[season][1]
        date_index = date_index[(date_index.month == months[0]) | (
            date_index.month == months[1]) | (date_index.month == months[2])]

    return date_index


def day_indexer(start, end, season):
    """
    Return the indices of observations grouped by day of year, i.e. all indices of observations 
    on 1 January, 2 January, etc.
    """
    rng = pd.period_range(start, end, freq='D')  # works with longer time spans
    months = season_dict[season][1]
    rng = rng[(rng.month == months[0]) | (rng.month == months[1]) | (rng.month == months[2])]
    dts = pd.DataFrame(index=rng)
    grouping = [dts.index.month, dts.index.day]
    
    return dts.groupby(grouping).indices


def month_indexer(start, end, season):
    """
    Return the indices of observations grouped by month, i.e. all indices of observations in 
    January, February etc.
    """
    rng = pd.period_range(start, end, freq='D')
    months = season_dict[season][1]
    rng = rng[(rng.month == months[0]) | (rng.month == months[1]) | (rng.month == months[2])]
    dts = pd.DataFrame(index=rng)
    grouping = [dts.index.month]

    return dts.groupby(grouping).indices
    

def generate_seeds(num_chunks, num_dates, num_nodes, shuffle_type, realizations, root_seed):
    """
    Generate seeds that are used to shuffle data. This is necessary to prevent the same shuffle
    patterns across different calls to the shuffle functions in stars_surrogates.py.


    Arguments:
    ----------
    num_chunks : int
        Number of time chunks to be processed.

    num_dates : int
        Maximum length of timeseries across the chunks.

    num_nodes : int
        Number of nodes in the dataset.

    shuffle_type : five character string
        The type of shuffling to be done on the data. The different characters map to these
        options:

            * [0] : c or i for consistent or inconsistent shuffle across chunks
            * [1] : d or n for date or node shuffle
            * [2] : a or s for asynchronous or synchronous shuffle
            * [3] : d, w, m, y or a for the pool type
            * [4] : d, w, m or y for the the block type

        Setting the shuffle type to 'noshu' prevents any data shuffling.

        .. note::
            Consistency determines whether all chunks receive the same rng seeds, i.e. whether the
            shuffling will be consistent across chunks. Inconsistent shuffling across the chunks is
            effectively equivalent to shuffling the data before it is split into chunks,
            emphasizing the contiguous character of the data. Consistent shuffling across the
            chunks reflects the view that each chunk is separate from the others.

            Synchronicity determines whether the shuffling is synchronous along the non-shuffled
            axis, i.e. whether, if the nodes are shuffled, they are shuffled the same way across
            all dates, or, if the dates are shuffled, they are shuffled the same way across all
            nodes. If the shuffling is done in an asynchronous manner it is almost completely
            randomizing the data (pool and block types will still be respected).

    realizations : int
        Number of realizations for which seeds have to be generated

    root_seed : int
        Root seed used by the random number generator in this function.


    Returns:
    --------
    seeds : 2D or 3D array
        An array of integers whose shape depends on the shuffle type chosen.
    """
    if shuffle_type == 'noshu':
        return np.zeros((num_chunks, 1, 1))

    if shuffle_type[0] == 'c':
        if shuffle_type[1] == 'n':
            if shuffle_type[2] == 's':
                n = (1, realizations, 1)
            elif shuffle_type[2] == 'a':
                # exclude this case: no way to consistently shuffle with different counts of
                # dates per chunk
                raise ValueError('cannot combine consistent with async. shuffling for nodes')
        elif shuffle_type[1] == 'd':
            if shuffle_type[2] == 's':
                n = (1, realizations, 1)
            elif shuffle_type[2] == 'a':
                n = (1, realizations, num_nodes)
    elif shuffle_type[0] == 'i':
        if shuffle_type[1] == 'n':
            if shuffle_type[2] == 's':
                n = (num_chunks, realizations, 1)
            elif shuffle_type[2] == 'a':
                n = (num_chunks, realizations, num_dates)
        elif shuffle_type[1] == 'd':
            if shuffle_type[2] == 's':
                n = (num_chunks, realizations, 1)
            elif shuffle_type[2] == 'a':
                n = (num_chunks, realizations, num_nodes)

    np.random.seed(root_seed)
    seeds = np.random.randint(1000000, 5000000, size=n)

    if seeds.shape[2] == 1:
        seeds = np.vstack(num_chunks * [seeds])

    return seeds


def prettyprint(text, stdout_flush=True, force_inline=False, top_padding=True,
        bottom_padding=False):
    """
    Print a line or several lines in a distinctive style. Input can include newline characters,
    the function will print the lines separately.


    Arguments:
    ----------
    text : string
        String to be printed.

    stdout_flush : bool
        Whether or not the stdout shall be flushed after this line.

    force_inline : bool
        Whether or not texts that contain multiple lines shall be formatted as a box (if False),
        or a sequence of lines formatted as if they were separte (if True).

    top_padding : bool
        Whether or not the printout begins with a newline character.

    bottom_padding : bool
        Whether or not the printout ends with a newline character.
    """
    fragments = text.split('\n')

    if force_inline or len(fragments) == 1:
        txt = ''

        for fragment in fragments:
            if fragment != '':
                fragment = ' ' + fragment + ' '
                txt += fragment.center(72, '-') + '\n'
            else:
                txt += '\n'

    else:
        txt = ''.center(72, '-')

        for fragment in fragments:
            if fragment != '':
                fragment = ' ' + fragment + ' '
                txt += '\n' + fragment.center(72, ' ')
            else:
                txt += '\n'

        txt += '\n' + ''.center(72, '-')

    txt = txt.strip('\n')

    if top_padding:
        txt = '\n' + txt.lstrip('\n')
    if bottom_padding:
        txt = txt.rstrip('\n') + '\n'

    print txt

    if stdout_flush:
        sys.stdout.flush()


def call_on_chunks(dir_contents, filekey, func_or_method, load_first=True, *args, **kwargs):
    """
    Call a method or a function on multiple files, objects or data arrays. {dir_contents} is a list of
    DirectoryContent objects - we use it to get the locations of all files with {filekey}.
    Depending on the status of {load_first} we then either load the contents of these files, and
    then call a function on method on these contents, saving the results. Or we call a function
    directly on the file name, saving the results.


    Arguments:
    ----------
    dir_contents : list of DirectoryContent objects
        A list containing DirectoryContent objects.

    filekey : string
        Type of data we are interested in. This string is used to look up all relevant files in
        the list. Valid examples: obs, wobs, sevs, ...

    func_or_method : function or string
        If a function, name of the function to call. If a string, name of the method to call. The 
        args and kwargs are handed down to the method or function.

    load_first : bool
        If true, we load the contents of the file with {filekey} with pd.read_pickle() or
        np.loadtxt(). We then call the method or function on this content. If false, we call the
        function directly on the file string.


    Returns:
    --------
    data : 1D array or list
        Array or list containing the loaded objects.
    """
    data = []

    for dco in dir_contents:
        try:
            data_file = getattr(dco, filekey)
        except AttributeError:
            data_file = None

        if data_file is not None:
            prettyprint('now processing %s' % data_file, 1)

            if isinstance(func_or_method, basestring):
                # method, we need an object
                curr_obj = pd.read_pickle(dco.folder + data_file)
                curr_data = getattr(curr_obj, func_or_method)(*args, **kwargs)

            else:
                # function, we can either call on file ...
                if not load_first:
                    curr_data = func_or_method(dco.folder + data_file, *args, **kwargs)

                # ... or we call on loaded object
                elif load_first:
                    try:
                        curr_obj = pd.read_pickle(dco.folder + data_file)
                    except (ImportError, IndexError):
                        curr_obj = np.loadtxt(dco.folder + data_file)
                    curr_data = func_or_method(curr_obj, *args, **kwargs)

            data.append(curr_data)

    try:
        return np.array(data)
    except ValueError:
        print 'could not broadcast arrays into np.array, returning lists'
        return data


def load_all_objects(dir_contents, filekey, silence_level=0):
    """
    This function loads all objects of {filekey} type that have been found by
    sm.locate_all_files(). Useful for pickled networks.


    Arguments:
    ----------
    dir_contents : list of DirectoryContent objects
        A list containing DirectoryContent objects.

    filekey : string
        Type of data file that we look for. Valid examples: tcn, wtcn, ...

    silence_level : int
        Inverse level of verbosity.


    Returns:
    --------
    objects : 1D array
        Array containing the loaded objects.
    """
    objects = []

    for dco in dir_contents:
        data_file = getattr(dco, filekey)

        if data_file is not None:
            if silence_level < 2:
                prettyprint('now loading %s' % data_file, 1)

            try:
                curr_object = pd.read_pickle(dco.folder + data_file)
            except:
                curr_object = np.loadtxt(dco.folder + data_file)
                # curr_object = pd.read_csv(dco.folder + data_file, delim_whitespace=True)

            objects.append(curr_object)

            del curr_object

    try:
        return np.array(objects)
    except ValueError:
        print 'could not broadcast arrays into np.array, returning lists'
        return objects