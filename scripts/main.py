# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:30:45 2016

@author: malte
"""
from libraries.helper import initialize
from processing.structure_data import sort_raw_data_into_chunks
from processing.calculate_networks import calc_networks_and_measures_on_chunks
from processing.compute_kappa import calc_kappa_from_measures
from misc.paths import (meta_file, stars_data_location, basis_data_location, 
                            base_output_location)

if __name__ == '__main__':
    variable_location = 3
    chunklength = 50

    season = 'summer'
    link_density = 0.5
    realizations = 2

    shuffle_type = 'un100'
    measure = 'ndg'
                                             
#    sort_raw_data_into_chunks(
#        variable_location, 
#        base_output_location, 
#        chunklength,
#        meta_file,
#        stars_data_location,
#        basis_data_location,
#        season, 
#        dataset='both')
#        
#    calc_networks_and_measures_on_chunks(
#        variable_location, 
#        base_output_location, 
#        chunklength,
#        season, 
#        link_density,
#        realizations)
#
#    variable_long_name, variable_short_name, output_location, dir_contents = initialize(
#        variable_location, 
#        base_output_location, 
#        chunklength, 
#        get_dir_contents=True)
#    
#    kappa_matrix = calc_kappa_from_measures(
#        variable_location,
#        dir_contents, 
#        link_density, 
#        shuffle_type, 
#        measure, 
#        season, 
#        target_realizations=5, 
#        percentile=False, 
#        average=False,
#        global_buckets=True, 
#        weights=None)