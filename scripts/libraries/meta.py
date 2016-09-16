# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:55:47 2014

@author: malte

A module providing utility functions for STARS and BASIS metadata.
"""
import os
import numpy as np
import pandas as pd

from datetime import date, datetime
from socket import getfqdn

from libraries.dictionaries import (filetype_dict, season_dict, threshold_dict, measures_dict, 
                                    link_creation_dict, node_selection_dict, variable_dict, 
                                    shuffle_dict)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Classes
#

class DirectoryContent:
    """
    An object that contains standard-compliant files that are located in a specific directory
    as attributes. Can be printed for overview.
    """
    def __init__(self, folder):
        if folder[-1] == '/':
            self.folder = folder
        else:
            self.folder = folder + '/'

    def __str__(self):
        attrlist = vars(self).items()

        ix = []
        for item in attrlist:
            try:
                ix.append(filetype_dict[item[0]][4])
            except KeyError:
                try:
                    if '_' in item[0]:
                        end = item[0].find('_')
                    else:
                        end = len(item[0])
                    ix.append(filetype_dict[item[0][1:end]][4])
                except KeyError:
                    ix.append(99)

        sortlist = sorted(zip(ix, attrlist))

        text = 'available in \'%s\' on %s:' % (self.folder, self.host)
        text = (' ' + text + ' ').center(72, '-')
        n = 0
        for ix, (attr, value) in sortlist:
            if ix < 99 and value:
                text += '\n%-*s : %s' % (21, attr, value)
                n += 1

        if n == 0:
            text += '\nno files found'

        return text


class FileMetadata:
    """
    An object that contains the metadata that can be inferred from a standard-compliant filename
    as attributes. Can be printed for overview.
    """
    def __init__(self, input_file):
        self.filename = input_file

    def __str__(self):
        attrlist = sorted(vars(self).items())

        text = 'file: \'%s\'' % self.filename
        text = '\n' + (' ' + text + ' ').center(72, '-')
        for attr, value in attrlist:
            if value:
                text += '\n%-*s : %s' % (26, attr, value)

        return text


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions that read and interpret filenames and directory structures
#


def locate_all_files(data_location, silence_level=0):
    """
    This function searches a data directory, assuming a subfolder structure as
    created by sort_raw_data_into_chunks().

    It runs get_directory_content() in every subdirectory and writes the
    filename objects into an array, providing a convenient way to access all
    derivative files.


    Arguments:
    ----------
    data_location : string
        Location of the directory that is to be searched according to the chunk
        subdirectory structure.

    silence_level : integer
        Inverse level of verbosity of the object.


    Returns:
    --------
    dir_contents : list
        A list containing the DirectoryContent objects for each subdirectory, i.e. information
        about any derivative files available in each chunk's subdirectory.
    """
    folder_list = sorted([folder for folder in os.listdir(data_location) if
        folder.endswith('stars') or folder.endswith('basis')])

    res = []

    for folder in folder_list:
        curr_folder = data_location + folder

        dco = get_directory_content(curr_folder, silence_level)

        y1 = int(folder[0:4])
        y2 = int(folder[5:9])

        dco.start = datetime(y1, 06, 01)
        dco.end = datetime(y2, 02, pd.period_range(y2, y2 + 1)[pd.period_range(y2,
                             y2 + 1).month == 2][-1].day)

        if curr_folder.endswith('basis'):
            dco.source = 'basis'
        elif curr_folder.endswith('stars'):
            dco.source = 'stars'

        res.append(dco)

    dir_contents = np.array(res)

    # sort data to put stars data first, basis data second,
    # sort in ascending order by time and length of chunk (large basis chunk at the end)
    dir_contents = sorted(dir_contents, key=lambda x: (x.end.year, x.end.year - x.start.year))
    dir_contents = sorted(dir_contents, key=lambda x: x.source, reverse=True)

    return dir_contents


def get_directory_content(folder, silence_level=0):
    """
    This function searches a directory for files that obey the standard filename conventions and
    returns a filename object containing the names of all those files. An easy way to access
    all information is to print the object.


    Arguments:
    ----------
    folder : string
        Location of the directory that is to be searched according to the standard
        filename convention.

    silence_level : integer
        Inverse level of verbosity of the object.


    Returns:
    --------
    dco : DirectoryContent object
        An object that contains as attributes the names of all derivative
        files found in the directory that was searched. Easily accessed by printing
        the object.
    """
    def find_file(file_list, filekey, season='all', ext=None):
        """
        Finds the file(s) that contain certain data in a given directory using pattern matching on
        the filenames.


        Arguments:
        ----------
        file_list : list of strings
            A list of filenames, the result of a ls command on a directory.

        filekey : string
            Type of data file that we look for. Valid examples: obs, evs, rates,
            wevs, tcn, wtcn, ...

        season : string
            Whether we shall look for seasonal data, default setting is all,
            thus looking for data files with year-long data. Other valid choices are
            winter and summer.

        ext : string, optional
            Extension of the file we are looking for, defaults to the standard file
            extension for this data type.


        Returns:
        --------
        name : string
            Name of the attribute of a DirectoryContent object referring to this file.

        value : string
            Name of the file.
        """
        # if no extension is specified, search for standard extension
        if not ext:
            ext = filetype_dict[filekey][2]

        if season != 'all':
            lis = [s for s in file_list if s.startswith(filetype_dict[filekey][0]) and
                   s.endswith(ext) and (season in s)]
        else:
            lis = [s for s in file_list if s.startswith(filetype_dict[filekey][0]) and
                   s.endswith(ext) and ('winter' not in s) and ('summer' not in s)]

        if len(lis) == 0:
            value = [None]
            name = [None]

        else:
            season_string = season_dict[season][0]

            if filekey is 'nm':
                value = []
                name = []

                for item in lis:
                    value.append(item)
                    fno = read_filename(item)

                    name.append(season_string + filekey + '_' + fno.measure + '_' +
                        fno.source_network_type + fno.shuffle_string + '_' + fno.link_creation)

            else:
                if len(lis) == 1:
                    value = [lis[0]]
                    name = [season_string + filekey]

                elif len(lis) > 1:
                    # if there are several files found, let the user decide
                    for i in xrange(len(lis)):
                        print '(%i) %s' % (i, lis[i])

                    value = [lis[int(raw_input('choose file: '))]]
                    name = [season_string + filekey]

        return name, value

    dco = DirectoryContent(folder)

    file_list = sorted(os.listdir(folder))
    dco.host = getfqdn()

    for filekey in filetype_dict:
        for season in season_dict:
            attr_name, attr_value = find_file(file_list, filekey, season)

            for (name, value) in zip(attr_name, attr_value):
                if value is not None:
                    setattr(dco, name, value)

    return dco


def read_filename(input_file):
    """
    This functions reads a standard-compliant filename and extracts relevant
    information about the data contained in the file. It serves as a metadata
    reader for the derivative files created elsewhere.


    Arguments:
    ----------
    input_file : string
        Path to the filename that shall be read.


    Returns:
    --------
    res : FileMetadata object
        An object that contains all the meta information about the data contained in the filename.
        Can be printed for easy overview.
    """
    input_file = input_file.split('/')[-1]

    res = FileMetadata(input_file)

    input_file = input_file.split('.')[0]
    tokens = input_file.split('_')

    # contained data
    res.datatype = filetype_dict[tokens[0]][1]
    res.format = filetype_dict[tokens[0]][3][0]

    # season
    if 'summer' in input_file:
        res.season = 'summer'
    elif 'winter' in input_file:
        res.season = 'winter'
    else:
        res.season = 'all'

    # root observations file
    if 'obs' in input_file:
        # obs_(var_tok)_(start)_(end)
        res.obs_source = input_file[input_file.find('obs'):input_file.find('obs') +
            season_dict[res.season][2]]

        res.start = date(int(res.obs_source[8:12]), int(res.obs_source[12:14]),
                         int(res.obs_source[14:16]))
        res.end = date(int(res.obs_source[17:21]), int(res.obs_source[21:23]),
                       int(res.obs_source[23:25]))

        date_index = pd.period_range(res.start, res.end)

        if res.season != 'all':
            months = season_dict[res.season][1]
            date_index = date_index[(date_index.month == months[0]) | (
                date_index.month == months[1]) | (date_index.month == months[2])]

        res.num_dates = len(date_index)

        var_tok = res.obs_source[4:7]

    # root events file
    if 'evs' in input_file:
        # evs_(thres_tok)_(obs_source)
        res.evs_source = input_file[input_file.find('evs'):input_file.find('evs') +
            season_dict[res.season][3]]

        thres_tok = res.evs_source[4:7]

    # other file types
    if res.datatype in ['anomalies']:
        # ano_(obs_source)
        pass
    elif res.datatype in ['network measure']:
        # nm_(mes_tok)_(net_tok)_(den_tok)_(evs_source)
        # nm_(mes_tok)_(net_tok)_(den_tok)_(obs_source)
        mes_tok = tokens[1]
        net_tok = tokens[2]
        den_tok = tokens[3]
    elif res.datatype in ['tsonis climate network', 'event synchronization network']:
        # (net_tok)_(den_tok)_(obs_source)
        net_tok = tokens[0]
        den_tok = tokens[1]

    # thres_tok: l10
    try:
        res.threshold_type = threshold_dict[thres_tok[0]][1]
        res.threshold = int(thres_tok[1:])
    except:
        pass

    # mes_tok: str
    try:
        res.measure = mes_tok
        res.measure_long = measures_dict[mes_tok][1]
    except:
        pass

    # net_tok: esnidamd
    try:
        res.source_network_type = net_tok[0:3]
        res.shuffle_string = net_tok[3:8]

        if res.shuffle_string != 'noshu':
            if res.shuffle_string[0:2] != 'un':
                res.shuffle_consistency = shuffle_dict[net_tok[3]][1]
                res.shuffle_object = shuffle_dict[net_tok[4]][1]
                res.shuffle_synchronicity = shuffle_dict[net_tok[5]][1]
                res.shuffle_pool_type = shuffle_dict['pools_and_blocks'][net_tok[6]][1]
                res.shuffle_block_type = shuffle_dict['pools_and_blocks'][net_tok[7]][1]
            else:
                res.shuffle_type = 'uniform random noise'
                res.shuffle_intensity = res.shuffle_string[2:]
        else:
            res.shuffle_type = 'original data'
    except:
        pass

    # den_tok: d50
    try:
        res.link_creation = den_tok
        res.link_creation_type = link_creation_dict[den_tok[0]][1]
        res.link_creation_value = den_tok[1:]
        res.link_creation_value_unit = link_creation_dict[den_tok[0]][2]
    except:
        pass

    # var_tok: m06
    try:
        res.node_selection = node_selection_dict[var_tok[0]][0]
        res.meta_file = node_selection_dict[var_tok[0]][1]
        res.num_nodes = node_selection_dict[var_tok[0]][2]

        res.var = int(var_tok[1:])
        res.varname = variable_dict[res.var][1]
        res.varname_long = variable_dict[res.var][0]
    except:
        pass

    return res
