# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:34:31 2016

@author: malte
"""
import numpy as np
import NetworkMeasures as nm

from libraries.helper import generate_seeds, day_indexer, month_indexer, prettyprint, initialize
from libraries.dictionaries import season_dict, variable_dict
from libraries.meta import read_filename
from misc.paths import meta_file, esync_program

from datetime import datetime
from subprocess import Popen, PIPE


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Options
#

# static threshold for event conversion
EVENTIZATION_THRESHOLD = 10

# misc
LONS, LATS = np.loadtxt(meta_file, usecols=[1, 2], unpack=True)
NODE_WEIGHTS = np.loadtxt(meta_file, usecols=[4]) 
ANGULAR_EPSILON = 0.05 
SPATIAL_GRANULARITY = 1500


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions
#

def calc_daily_anos(data, indices):
    # fast version of raw_data.anomalies() for daily anomalies, using pre-calculated date indices.
    obs_data = data.T
    anom_data = np.empty_like(obs_data)
    
    for day in indices:
        anom_data[indices[day]] = obs_data[indices[day]] - obs_data[indices[day]].mean(axis=0)
    
    return anom_data.T


def calc_fixed_evs(data, thres):
    # fast version of raw_data.events() for fixed thresholds
    evs = data.copy()
    evs[evs < thres] = 0
    evs[evs >= thres] = 1    

    return evs


def shuffle_idamd(data, indices, seed=None):
    # fast shuffle the raw data using the idamd scheme
    np.random.seed(seed)
    
    shuffled_data = np.empty_like(data)
    
    for i in xrange(shuffled_data.shape[0]):
        for pool in indices:
            perms = np.random.permutation(indices[pool])
            shuffled_data[i, indices[pool]] = data[i, perms]
        
    return shuffled_data


def uniform_noise(data, indices, epsilon):
    # add uniformly distributed random noise to the raw data
    random_data = np.empty_like(data)
    
    for i in range(data.shape[0]):
        rn = np.random.uniform(low=-epsilon, high=epsilon, size=data.shape[1])
        noise = rn * data[i, :].mean()
        random_data[i, :] = data[i, :] + noise
    
    return random_data
    

def nsi_degree(corr, node_weights, link_density):
    # calculate nsi degree from correlation matrix using node_weights and fixed link_density

    # threshold from link density
    n_nodes = corr.shape[0]
    
    flat_corr = corr.flatten()
    flat_corr.sort()
    threshold = flat_corr[int((1 - link_density) * (len(flat_corr) - n_nodes))]

    # binary adjacency matrix
    A = np.zeros(corr.shape, dtype=np.bool)
    A[corr > threshold] = 1
    np.fill_diagonal(A, 0)

    # nsi degree with constant weights
    Aplus = A + np.identity(n_nodes)

    wAplus = np.dot(Aplus, node_weights)

    return wAplus


def esync_simple(data):
    # call esync_simple on raw data to calculate synchronizations
    num_nodes = data.shape[0]
    time_len = data.shape[1]

    proc = Popen([esync_program, str(num_nodes), str(time_len)], stdin=PIPE, stdout=PIPE,
                 stderr=PIPE, shell=False)

    for item in data.flatten():
        proc.stdin.write('%i ' % item)

    syncs_matrix = np.zeros((num_nodes, num_nodes))
    out, err = proc.communicate()

    for (i, line) in zip(xrange(1, num_nodes), out.split('\n')):
        syncs_matrix[i, :i] = syncs_matrix[:i, i] = np.fromstring(
            line, dtype=float, sep=' ')
    
    return syncs_matrix


def calc_networks_and_measures_on_chunks(variable_location, base_output_location, chunklength,
                                         season, link_density, realizations):
    """
    Calculates networks and network measures on the raw data that was structured by 
    sort_raw_data_into_chunks().

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
        Size of chunks in years.
        
    season : string
        Which season to run, either summer, winter, or all.

    link_density : int
        Link density of the networks in percent.
    
    realizations : int
        Number of realizations.
    """
    if link_density > 0.5:
        print 'warning, link density above 50 percent, use 0 < link_density < 1'

    _, var_short_name, _, dir_contents = initialize(variable_location, base_output_location, 
                                                    chunklength)
    
    network_type = variable_dict[variable_location][2][0]
    
    num_chunks = len(dir_contents)
    num_dates = max([read_filename(
        getattr(dco, season_dict[season][0] + 'obs')).num_dates for dco in dir_contents])
    num_nodes = read_filename(
        getattr(dir_contents[0], season_dict[season][0] + 'obs')).num_nodes
    
    root_seed = datetime.now().microsecond
    seeds = generate_seeds(num_chunks, num_dates, num_nodes, 'idamd', realizations, root_seed)
    
    for i, dco in enumerate(dir_contents):
        prettyprint('%s%02i, %s: %i-%i' % (network_type, link_density * 100, season, 
                                           dco.start.year, dco.end.year))

        output_folder = (base_output_location + str(chunklength) + 'y/' + 
            '/'.join(dco.folder.split('/')[-3:]))
        
        month_indices = month_indexer(dco.start, dco.end, season)
        day_indices = day_indexer(dco.start, dco.end, season)

        data_file = getattr(dco, season_dict[season][0] + 'obs')   
        observable = np.loadtxt(dco.folder + data_file, dtype=np.float)

        def _save_networkmeasure(measure, data, shuffle_type):
            output_file = (output_folder + 'nm_{measure}' + '_{network}{shuffle}_d{:.0f}_'.format(
                link_density * 100, network=network_type, shuffle=shuffle_type) + data_file)   
            
            with open(output_file.format(measure=measure), 'a') as of:
                np.savetxt(of, (data, ), delimiter=' ', fmt='%.3f') 

        def _save_network(net, data, shuffle_type):
            output_file = (output_folder + 'net_{network}{shuffle}_'.format(network=net, 
                           shuffle=shuffle_type) + data_file)
    
            np.save(output_file[:-3] + 'npy', data)

        for shuffle_type in ['noshu', 'idamd', 'un100', 'un025', 'un005']:
            # shuffle observable
            if shuffle_type == 'noshu':
                if variable_location in [6]:
                    # calculate synchronizations   
                    evs = calc_fixed_evs(observable, thres=EVENTIZATION_THRESHOLD)          
                    syncs = esync_simple(evs)
                    # _save_network('esn', syncs, shuffle_type)
             
                    # calculate network measures 
                    ndg = nsi_degree(syncs, NODE_WEIGHTS, link_density)
                    strength = nm.Strength(syncs.flatten(), LONS, LATS, SPATIAL_GRANULARITY)
                    icr, icp, ice = nm.UndirectedDenseWeightedDirectionality(
                                    syncs.flatten(), LONS, LATS, ANGULAR_EPSILON, 
                                    SPATIAL_GRANULARITY)
                    _save_networkmeasure('ndg', ndg, shuffle_type)
                    _save_networkmeasure('str', strength, shuffle_type)           
                    _save_networkmeasure('icr', icr, shuffle_type)
                    _save_networkmeasure('icp', icp, shuffle_type)
                    _save_networkmeasure('ice', ice, shuffle_type)
                    
                elif variable_location in [3, 4, 5]:
                    # calculate correlations   
                    anos = calc_daily_anos(observable, day_indices)
                    corr = np.corrcoef(anos).astype(np.float32)
                    # _save_network('tcn', corr, shuffle_type)
    
                    # calculate network measures           
                    ndg = nsi_degree(corr, NODE_WEIGHTS, link_density) 
                    strength = nm.Strength(corr.flatten(), LONS, LATS, SPATIAL_GRANULARITY)
                    _save_networkmeasure('ndg', ndg, shuffle_type)
                    _save_networkmeasure('str', 1000 * strength, shuffle_type) 
            else:
                for j in xrange(realizations):
                    if shuffle_type == 'idamd':
                        observable = shuffle_idamd(observable, month_indices, seeds[i, j, :])
                    elif shuffle_type == 'un100':
                        observable = uniform_noise(observable, month_indices, epsilon=1.00)
                    elif shuffle_type == 'un025':
                        observable = uniform_noise(observable, month_indices, epsilon=0.25)
                    elif shuffle_type == 'un005':
                        observable = uniform_noise(observable, month_indices, epsilon=0.05)

                    if variable_location in [6]:
                        # calculate synchronizations   
                        evs = calc_fixed_evs(observable, thres=EVENTIZATION_THRESHOLD)          
                        syncs = esync_simple(evs)
                        # _save_network('esn', syncs, shuffle_type)
                 
                        # calculate network measures 
                        ndg = nsi_degree(syncs, NODE_WEIGHTS, link_density)
                        strength = nm.Strength(syncs.flatten(), LONS, LATS, SPATIAL_GRANULARITY)
                        icr, icp, ice = nm.UndirectedDenseWeightedDirectionality(
                                        syncs.flatten(), LONS, LATS, ANGULAR_EPSILON, 
                                        SPATIAL_GRANULARITY)
                        _save_networkmeasure('ndg', ndg, shuffle_type)
                        _save_networkmeasure('str', strength, shuffle_type)           
                        _save_networkmeasure('icr', icr, shuffle_type)
                        _save_networkmeasure('icp', icp, shuffle_type)
                        _save_networkmeasure('ice', ice, shuffle_type)
                        
                    elif variable_location in [3, 4, 5]:
                        # calculate correlations   
                        anos = calc_daily_anos(observable, day_indices)
                        corr = np.corrcoef(anos).astype(np.float32)
                        # _save_network('tcn', corr, shuffle_type)
        
                        # calculate network measures           
                        ndg = nsi_degree(corr, NODE_WEIGHTS, link_density) 
                        strength = nm.Strength(corr.flatten(), LONS, LATS, SPATIAL_GRANULARITY)
                        _save_networkmeasure('ndg', ndg, shuffle_type)
                        _save_networkmeasure('str', 1000 * strength, shuffle_type)         


if __name__ == '__main__':
    pass
