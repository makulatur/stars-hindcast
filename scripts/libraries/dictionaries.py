# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:39:17 2015

@author: malte
"""
import NetworkMeasures as nm
import numpy as np


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Dictionaries
#

# available variables to be read from the stars dataset
# { column number : (long name, short name, available network types, label, unit) }
variable_dict = {
    3: ('maximum daily temperature', 'tmax', ['tcn'], 'Max. daily temp.', '$^\circ$C'),
    4: ('mean daily temperature', 'tmean', ['tcn'], 'Mean. daily temp.', '$^\circ$C'),
    5: ('minimum daily temperature', 'tmin', ['tcn'], 'Min. daily temp.', '$^\circ$C'),
    6: ('total daily precipitation', 'prec', ['esn'], 'Total daily prec.', 'mm'),
}


# network measures dictionary
# { short name : (short name, long name, method or function used to calculate) }
measures_dict = {
    'ndg': ('ndg', 'nsi degree', 'nsi_degree'),
    'icr': ('icr', 'isochrone rho', nm.UndirectedDenseWeightedDirectionality),
    'icp': ('icp', 'isochrone phi', nm.UndirectedDenseWeightedDirectionality),
    'ice': ('ice', 'isochrone error', nm.UndirectedDenseWeightedDirectionality),
    'str': ('str', 'strength', nm.Strength)
}


# network type dictionary
# { short name : (short name, long name, source data type, [ default network measures ]) }
network_dict = {
    'tcn': ('tcn', 'tsonis climate network', 'obs', ['ndg']),
    'esn': ('esn', 'event synchronization network', 'evs', ['ndg', 'icr', 'icp', 'ice', 'str'])
}


# season dictionary
# { season name : (season key character, list of months included, length of filename I, 
# length of filename II) }
season_dict = {
    'all': ('', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 25, 33),
    'winter': ('w', [1, 2, 12], 32, 40),
    'summer': ('s', [6, 7, 8], 32, 40)
}


# all filetypes currently in use
# { short name : (short name / identifier at beginning of filename, long name, preferred extension,
# (format, datatype), print priority) }
filetype_dict = {
    'obs': ('obs', 'observations', '.dat', ('%.1f', np.float), 1),
    'evs': ('evs', 'events', '.dat', ('%i', np.bool), 2),
    'ano': ('ano', 'anomalies', '.dat', ('%.2f', np.float), 3),
    'tcn': ('tcn', 'tsonis climate network', '.npy', ('binary', np.object), 4),
    'esn': ('esn', 'event synchronization network', '.npy', ('binary', np.object), 5),
    'nm': ('nm', 'network measure', '.dat', ('%.3f', np.float), 6)
}


# threshold dictionary
# { short name : (short name, medium name, long name) }
threshold_dict = {
    'a': ('a', 'all', 'score at percentile of all nodes, all dates'),
    't': ('t', 'time', 'score at percentile of all nodes, per single date'),
    'n': ('n', 'nodes', 'score at percentile of all dates, per single node'),
    'l': ('l', 'literal', 'static threshold')
}


# link creation dictionary
# { short name : (short name, long name) }
link_creation_dict = {
    'd': ('d', 'link density', 'percent'),
    't': ('t', 'threshold', 'literal')
}

# node selection types as enforced through meta_file
# { identifier in filename : (long name, name of meta file, number of nodes) }
node_selection_dict = {
    'm': ('German mainland', 'germany_noisles_nointernational.dat', 2331),
    'g': ('German mainland and ilses', 'germany_isles_nointernational.dat', 2337),
    'f': ('Full set', 'germany_isles_international.dat', 2533),
    'a': ('Aljoscha compatability mode' 'aljoscha.dat', 2337)
}


# shuffle type dictionary
# { short name : (short name, long name) and { short name : (short name, associated
# pd.DataFrame method, relative size ) } }
shuffle_dict = {
    'n': ('n', 'nodes'),
    'd': ('d', 'dates'),
    's': ('s', 'synchronous'),
    'a': ('a', 'asynchronous'),
    'i': ('i', 'inconsistent'),
    'c': ('c', 'consistent'),
    '-': ('-', 'no shuffle'),
    'pools_and_blocks': {
        'd': ('d', 'dayofyear', 0),
        'w': ('w', 'weekofyear', 1),
        'm': ('m', 'month', 2),
        'y': ('y', 'year', 3),
        'a': ('a', 'all', 4),
        '-': ('-', 'no shuffle', None),
    }
}