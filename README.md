# Analysis of STARS hindcast data using Python
This is a collection of Python scripts that I used while writing my Bachelor's thesis "Analysis of Spatial Patterns in STARS-Generated Data using Complex Networks and Kappa-Statistics" at the [Potsdam-Institute for Climate Impact Research](https://www.pik-potsdam.de) in 2016 under the supervision of Dr. Norbert Marwan and Prof. Jürgen Kurths.

The STARS model is developed at PIK, more details can be found [here](https://www.pik-potsdam.de/research/climate-impacts-and-vulnerabilities/models/stars). My analysis of the STARS-generated hindcast dataset can be found in [my thesis](thesis.pdf).

Feel free to contact me in case of any questions.

## Usage
The `scripts` directory contains the scripts that can be used to handle STARS and BASIS data. The basic workflow can be seen in `scripts/main.py`. It consists of three steps:

1. Structuring and preprocessing of the raw data
2. Computation of Tsonis Climate Networks and Event Synchronization Networks and the associated network measures Degree, Strength and Directionality.
3. Calculation of Cohen's Kappa between the network measure vectors. 

Please remember to update the `scripts/misc/paths.py` to match your local directory structure. 

The files `scripts/misc/functions.py` and `scripts/misc/plotting.py` provide some additional functionalities that might be useful for others.

The `meta` directory contains the meta data file I used while handling the STARS and BASIS data, it reflects my particular selection of meteorological stations in the dataset.

The `pkg` directory contains work provided by Aljoscha Rheinwalt that was used to calculate Event Synchronization Networks and compute the Strength and Directionality measures.

## Requirements
The project is written in Python 2.  
 
In addition to the standard libraries like Numpy, Scipy, Matplotlib, Progressbar and Pandas, the package 'NetworkMeasures' by Aljoscha Rheinwalt has to be installed, see `pkg/spat_corr`.

## Raw data structure                                   
The scripts expect the raw data in one single folder per dataset containing one file per station, e.g. a number of `(id)simsz.dat` files for STARS data and `(id)basz.dat` for BASIS data.