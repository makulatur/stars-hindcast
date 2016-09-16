# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:16:43 2015

@author: malte

A module providing plotting facilities for station data over Germany.
"""

"""
Reminder: use subplot2grid to center last odd plot

pl.figure(0, (ncols * 8, nrows * 8), dpi=300)
for i in xrange(n):
    if i == (n - 1):
        ax = pl.subplot2grid((nrows, ncols * 4), (i / 2, 2), colspan=4)
    else:
        ax = pl.subplot2grid((nrows, ncols), (i / 2, i % 2))
"""
import sys
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from textwrap import wrap

from misc.paths import meta_file, mask_location, xpts_location, ypts_location


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Labels
#

# standard labels
labels_50y = [
    '1500-1550', '1550-1600', '1600-1650', '1650-1700', '1700-1750', '1750-1800',
    '1800-1850', '1850-1900', '1900-1950', '1950-1999', 'Basis'
]

labels_20y = [
    '1500-1520', '1520-1540', '1540-1560', '1560-1580', '1580-1600', '1600-1620',
    '1620-1640', '1640-1660', '1660-1680', '1680-1700', '1700-1720', '1720-1740',
    '1740-1760', '1760-1780', '1780-1800', '1800-1820', '1820-1840', '1840-1860',
    '1860-1880', '1880-1900', '1900-1920', '1920-1940', '1940-1960', '1960-1980',
    '1980-1999', 'Basis'
]

# can be either n = chunksize or n = number of chunks
c_labels = {20: labels_20y, 26: labels_20y, 11: labels_50y, 50: labels_50y}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Associated files
#

# Germany numpy mask and grid points
MASK = np.loadtxt(mask_location)
XPTS = np.loadtxt(xpts_location)
YPTS = np.loadtxt(ypts_location)

# standard coordinates of the stations
LONS, LATS = np.loadtxt(meta_file, usecols=[1, 2], unpack=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Options
#

# basemap window in meters around central location
MWIN = 1000000

# colormap, cp. http://matplotlib.org/examples/color/colormaps_reference.html
# temperature differences: pl.cm.RdBu, pl.cm.RdBu_r
# general purpose multicolor: pl.cm.PiYG, pl.cm.viridis, pl.cm.jet
# temperature: pl.cm.afmhot
# precipitation: pl.cm.Blues
# general purpose monocolor: pl.cm.Greens
CMAP = pl.cm.RdBu_r

# valuebar color
COLOR = 'yellow'

# nice fonts and default fig sizes
pl.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
pl.rcParams.update(params) 

a4_textwidth = 5.9
std_x = a4_textwidth * 0.5
std_y = a4_textwidth * 0.35
full_x = a4_textwidth

std_fontsize = 11
math_fontsize = 13
map_fontsize = 8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Helper functions
#

def __bmap(ax, labelloc, mwin, quality='i'):
    """
    Internal function that creates a basic Basemap object that is centered on Germany, features
    country borders and colored oceans as well as relevant lat and lon lines.
    """
    bm = Basemap(resolution=quality, area_thresh=500, projection='aeqd',
                 lon_0=10, lat_0=51, width=mwin, height=mwin, ax=ax)
    bm.fillcontinents(color='white', zorder=1)
    # bm.shadedrelief(scale=0.8, zorder=2)
    bm.drawcountries(linewidth=0.5, color='0.5', zorder=4)
    bm.drawcoastlines(linewidth=0.5, zorder=5)
    # bm.drawparallels([47, 51, 55], labels=labelloc[0], color='0.3', zorder=6)
    # bm.drawmeridians([5, 10, 15], labels=labelloc[1], color='0.3', zorder=7)
    bm.drawmapboundary(fill_color='turquoise', zorder=8)

    return bm


def __force_2d_array(var):
    """
    Internal function that converts incoming variables of type list or array into 2D numpy arrays.
    """
    var = np.array(var)

    if var.ndim == 1:
        var = np.reshape(var, (1, var.shape[0]))

    return var

def __statusprint(text, i, n):
    """
    Print some text to stdout and flush afterwards. Rewrite over the last line.


    Arguments:
    ----------
    text : string
        String to be printed.
    
    i : int
        Current iteration.
        
    n : int
        Maximum iteration.
    """    
    sys.stdout.write('\r')
    sys.stdout.write(text + ' %2i/%2i. ' % (i + 1, n))
    sys.stdout.flush()
    
    if i + 1 == n:
        sys.stdout.write('\r' + text + ' %2i/%2i' % (i + 1, n) + '. all done. \n')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Functions that plot data on basemaps
#


def variable_on_map(var, meta_file='standard', method='nearest', n_levels=12, **kwargs):
    """
    Draws a variable on a basemap. Uses pl.contourf() to create the
    visuals, using the interpolation method specified.


    Arguments:
    ----------
    var : 1D or 2D array, dim: (n_stations) or (n_datasets, n_stations)
        The data to be plotted, each row containing one value per station. If 2D
        array is provided, each column is interpreted as a new set of variables that will
        be plotted onto a new map.

    meta_file : string
        Path to the metadata file, or 'standard'.

    method : string
        Interpolation method used by pl.contourf() to fill areas between stations. Valid choices
        are nearest, linear, cubic

    labels : list or 1D array, optional
        Labels to be used on each subplot, corresponding to the columns in var. If not provided,
        and if var is of standard size (10 or 11 columns, i.e. all 50-year-chunks are included),
        the labels are taken from a standard list.

    levels : int
        Number of levels into which the color mapping will be divided. The range is set for each
        subplot via its min and max values. Overridden by levels_array


    Optional Keyword Arguments:
    ---------------------------
    levels_array : 1D array
        The global levels for the color mapping, applied to all subplots. For example:
        >>> levels = np.linspace(var.min(), var.max(), levels=12)
        
    cmap : matplotlib colormap
        Colormap to use on the plot.


    Returns
    -------
    fig : figure object
        The requested plots as a figure.
    """
    var = __force_2d_array(var)

    n = var.shape[0]

    if meta_file == 'standard':
        lons, lats = LONS, LATS
    else:
        lons, lats = np.loadtxt(meta_file, usecols=[1, 2], unpack=True)

    if 'quality' in kwargs:
        quality = kwargs['quality']
    else:
        quality = 'i'
    
    if 'labels' in kwargs:
        labels = kwargs['labels']
    else:
        try:
            labels = c_labels[n]
        except KeyError:
            labels = [''] * n  
   
    if 'cmap_array' in kwargs:
        cmaps = kwargs['cmap_array']
    elif 'cmap' in kwargs:
        cmaps = [kwargs['cmap']] * n
    else:
        cmaps = [CMAP] * n

    if 'nrows_ncols' in kwargs:
        nrows = kwargs['nrows_ncols'][0]
        ncols = kwargs['nrows_ncols'][1]
    else:
        if n == 1:
            ncols = 1
            nrows = 1
        elif not n % 2:
            ncols = 2
            nrows = int(np.ceil(n / 2.0))
        else:
            ncols = 3
            nrows = int(np.ceil(n / 3.0))

    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    else:
        figsize = (std_x * ncols, std_y * nrows)

    if 'levels_array' in kwargs:
        levels_array = kwargs['levels_array']
    else:
        if 'categorical' in kwargs and kwargs['categorical']:
            levels_array = [0.5 + np.arange(min(row) - 1, max(row) + 1) for row in var]
        else:
            levels_array = [np.linspace(min(row), max(row), num=n_levels) for row in var]

    if 'norm_array' in kwargs:
        norm_array = kwargs['norm_array']
    else:
        norm_array = [mpl.colors.Normalize(min(row), max(row)) for row in var]

    fig = pl.figure(figsize=figsize)
    grid = AxesGrid(fig, 111,
                    ngrids=n,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=0.1,       # 0.6,
                    cbar_location='top',
                    cbar_mode='each',
                    cbar_size='5%',     # '2%',
                    cbar_pad=0)         # '1%')

    for i in range(n):
        __statusprint('processing plot', i, n)

        cax = grid[i]
        cbax = grid.cbar_axes[i]
        
        # center last plot if we have 10 plots on a 4x3 grid
        if nrows == 4 and ncols == 3 and n == 10 and i == (n - 1):
            grid[-3].axis('off')
            grid[-1].axis('off')
            grid.cbar_axes[-3].set_axis_off()
            grid.cbar_axes[-1].set_axis_off()
            cax = grid[i + 1]
            cbax = grid.cbar_axes[i + 1]
        
        # label locations for the parallels (sides) [0] and the meridians (bottom) [1]         
        if ncols == 1:
            labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))
        elif ncols == 2:
            if i % 2:
                labelloc = np.vstack(([0, 1, 0, 0], [0, 0, 0, 1]))
            else:
                labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))
        elif ncols == 3:
            if not i % 3:
                labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))
            elif not (i - 2) % 3:
                labelloc = np.vstack(([0, 1, 0, 0], [0, 0, 0, 1]))
            else:
                labelloc = np.vstack(([0, 0, 0, 0], [0, 0, 0, 1]))

        if 'no_coords' in kwargs and kwargs['no_coords']:
            labelloc = np.vstack(([0, 0, 0, 0], [0, 0, 0, 0]))   

        # plot basemap            
        bm = __bmap(cax, labelloc, MWIN, quality)
        x, y = bm(lons, lats)
        xpt, ypt = bm(3.3, 54)
        mz = np.ma.masked_array(griddata((x, y), var[i], (XPTS[None, :], YPTS[:, None]),
                method=method), mask=MASK)

        # plot contour
        im = cax.contourf(XPTS, YPTS, mz, levels_array[i], cmap=cmaps[i], norm=norm_array[i],
             extend='neither', zorder=3)
        im.cmap.set_over(im.cmap(1.0), 1)
        im.cmap.set_under(im.cmap(0.0), 1)
        
        if 'categorical' in kwargs and kwargs['categorical']:
            ticks = np.arange(var[i].min() - 1, var[i].max() + 2)
            cbax.colorbar(im, format='%i', ticks=ticks)
            # cbax.xaxis.set_ticklabels(np.arange(1, len(ticks) + 1))
        else:
            if 'cbax_fmt' in kwargs:
                fmt = kwargs['cbax_fmt']
            else:
                fmt = '%.2f'
            cbax.colorbar(im, format=fmt)
        cbax.tick_params(axis='x', labelsize=map_fontsize, top=False, bottom=False, pad=1)
        
        try:
            cax.text(xpt, ypt, labels[i], fontsize=map_fontsize)
        except:
            print 'error'
            print labels[i]
            pass

    return fig


def isochrones_on_map(rho, phi, err, meta_file='standard', valuebar=False, labels=None):
    """
    Draws isochrones on a basemap, using pl.stream() to create the visuals.


    Arguments:
    ----------
    rho, phi, err : 1D or 2D arrays
        The data to be plotted, each row containing one value per station. This corresponds
        to the standard output of NetworkMeasures.UndirectedDenseWeightedDirectionality. If 2D
        arrays are provided, each column is interpreted as a new set of variables that will
        be plotted onto a new map.

    meta_file : string
        Path to the metadata file

    valuebar : bool
        Whether or not a valuebar shall be plotted under the (normalized) colorbar,
        indicating which values actually occur on this particular subplot

    labels : list or 1D array, optional
        Labels to be used on each subplot, corresponding to the columns in var. If not provided,
        and if var is of standard size (10 or 11 columns, i.e. all 50-year-chunks are included),
        the labels are taken from a standard list.


    Returns:
    --------
    fig : figure object
        The requested plots as a figure.
    """
    # values between which the colors will be normalized,
    # this is to exclude, for example, completely white lines
    VMIN = 60
    VMAX = 1600

    # for indexing purposes make sure incoming data is a 2D array
    for var in [rho, phi, err]:
        var = __force_2d_array(var)
    n = rho.shape[0]

    if meta_file == 'standard':
        lons, lats = LONS, LATS
    else:
        lons, lats = np.loadtxt(meta_file, usecols=[1, 2], unpack=True)

    if not labels:
        try:
            labels = c_labels[n]
        except KeyError:
            pass
    elif len(labels) != n:
        raise ValueError('there must be a label for each dataset')

    if n == 1:
        ncols = 1
        nrows = 1
    elif n % 3:
        ncols = 3
        nrows = int(np.ceil(n / 3.0))
    elif n % 2:
        ncols = 2
        nrows = int(np.ceil(n / 2.0))

    fig = pl.figure(None, (ncols * 8, nrows * 8), dpi=300)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(nrows, ncols),
                    ngrids=n,
                    axes_pad=0.6,
                    cbar_location='top',
                    cbar_mode='each',
                    cbar_size='2%',
                    cbar_pad='1%')

    for i in range(n):
        # label locations for the parallels (sides) [0] and the meridians (bottom) [1]
        if ncols == 2:
            if i % 2:
                labelloc = np.vstack(([0, 1, 0, 0], [0, 0, 0, 1]))
            else:
                labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))
        elif ncols == 3:
            if not i % 3:
                labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))
            elif not (i - 2) % 3:
                labelloc = np.vstack(([0, 1, 0, 0], [0, 0, 0, 1]))
            else:
                labelloc = np.vstack(([0, 0, 0, 0], [0, 0, 0, 1]))

        bm = __bmap(grid[i], labelloc, MWIN)
        x, y = bm(lons, lats)
        xpt, ypt = bm(3.3, 54)

        ze = np.ma.masked_array(griddata((x, y), err[i, :],
               (XPTS[None, :], YPTS[:, None])), mask=MASK)
        zr = np.ma.masked_array(griddata((x, y), rho[i, :],
               (XPTS[None, :], YPTS[:, None])), mask=MASK)
        zx = np.ma.masked_array(griddata((x, y), rho[i, :] * np.cos(phi[i, :]),
               (XPTS[None, :], YPTS[:, None])), mask=MASK)
        zy = np.ma.masked_array(griddata((x, y), rho[i, :] * np.sin(phi[i, :]),
               (XPTS[None, :], YPTS[:, None])), mask=MASK)
        norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)

        im = grid[i].streamplot(XPTS, YPTS, zx, zy, color=zr,
                      linewidth=2.5 * (1 - ze),
                      minlength=0.04,
                      arrowsize=.001,
                      norm=norm,
                      cmap=pl.cm.gist_heat_r,
                      density=2, zorder=2)
        grid.cbar_axes[i].colorbar(im.lines)

        if labels:
            grid[i].text(xpt, ypt, labels[i], fontsize=std_fontsize)

        if valuebar:
            mmin, mmax = rho[i, :].min(), rho[i, :].max()
            if mmin < VMIN:
                mmin = VMIN
            if mmax > VMAX:
                mmax = VMAX
            mmin -= VMIN
            mmax -= VMIN
            mmin *= MWIN / float(VMAX - VMIN)
            mmax *= MWIN / float(VMAX - VMIN)

            grid[i].plot((mmin + 14000, mmax), (MWIN - 14000, MWIN - 14000),
                color=COLOR, linewidth=7, zorder=9)

    return fig


def links_on_map(adjacency_matrix, station_id, meta_file):
    """
    Draws a particular station and all stations that it is linked to according to the
    {adjacency_matrix}.


    Arguments:
    ----------
    adjacency_matrix : 2D numpy array
        Square adjacency matrix. Every value that is nonzero will be counted as a link.

    station_id : int
        ID of the station to examine.

    meta_file : string
        Path to the metadata file.


    Returns:
    --------
    fig : figure object
        The requested plots as a figure.
    """
    # coordinates of the stations
    ids, lons, lats = np.loadtxt(meta_file, usecols=[0, 1, 2], unpack=True)

    # get index of station to display
    station_idx = np.where(ids == station_id)[0][0]

    # get all stations that are linked to this one
    adjacency_vector = adjacency_matrix[station_idx].nonzero()[0]

    # draw basic map
    fig = pl.figure(None, (8, 8), dpi=72)
    grid = AxesGrid(fig, 111,
                    ngrids=1,
                    nrows_ncols=(1, 1),
                    axes_pad=0.6)

    labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))

    bm = __bmap(grid[0], labelloc, MWIN)
    x, y = bm(lons, lats)
    xpt, ypt = bm(3.3, 54)

    # draw line between station and all linked stations
    for target_idx in adjacency_vector:
        bm.plot([x[station_idx], x[target_idx]], [y[station_idx], y[target_idx]], 'D-',
            markersize=1.5, linewidth=0.2, color='k', markerfacecolor='k')

    grid[0].text(xpt, ypt, str(station_id), fontsize=std_fontsize)

    return fig


def stations_on_map(station_ids, meta_file, scatter=False, highlight=None, **kwargs):
    """
    Draw the location of one or several stations on the map.

    Additional keyword arguments in `kwargs` will be handed over to the `plot`-function.


    Arguments:
    ----------
    station_ids : int or list
        IDs of the stations to examine.

    meta_file : string
        Path to the metadata file.


    Returns:
    --------
    fig : figure object
        The requested plots as a figure.

    """
    # coordinates of the stations
    ids, lons, lats = np.loadtxt(meta_file, usecols=[0, 1, 2], unpack=True)

    # get the indices of stations to display
    if type(station_ids) == int:
        station_ids = [station_ids]

    idx = []
    for stat in station_ids:
        idx.append(np.where(ids == stat)[0][0])

    # draw basic map
    fig = pl.figure(figsize=(std_x, 2 * std_y))
    grid = AxesGrid(fig, 111,
                    ngrids=1,
                    nrows_ncols=(1, 1),
                    axes_pad=0.6)

    if 'no_coords' in kwargs and kwargs['no_coords']:
        labelloc = np.vstack(([0, 0, 0, 0], [0, 0, 0, 0]))
    else:
        labelloc = np.vstack(([1, 0, 0, 0], [0, 0, 0, 1]))

    bm = __bmap(grid[0], labelloc, MWIN)
    x, y = bm(lons, lats)
    xpt, ypt = bm(3.3, 54)

    if scatter:
        bm.scatter(x[idx], y[idx], zorder=10, **kwargs)
    else:
        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        
        if 'ms' not in kwargs:
            kwargs['ms'] = 15
        
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        
        for station_idx in idx:
            bm.plot(x[station_idx], y[station_idx], zorder=10, **kwargs)

    if highlight:
        bm.plot(x[np.where(ids == highlight)[0][0]], y[np.where(ids == highlight)[0][0]], 
                zorder=11, color='r', marker='.', ms=7.5, alpha=1)

    if len(idx) == 1:
        grid[0].text(xpt, ypt, 'ID: ' + str(station_ids), fontsize=std_fontsize)

    return fig



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Plot kappa matrix
#

def plot_kappa_matrix(matrix, display=True, mask_diagonal=True, bare=False, **kwargs):
    """
    Plots a Kappa matrix as a heatmap.


    Arguments:
    ----------
    matrix : 2D array
        Matrix of kappa values as created by kappa_matrix().

    display : bool
        If true, the plot is displayed immediately.

    nbins : int, optional
        Number of bins to use for display, defaults to 25 for float matrix or 2 for binary matrix.

    colorbar_range : tuple of floats, optional
        Minimum and maximum of the colorbar range.

    labels : list, optional
        A list of strings containing the plot labels, usually the identification of the time
        chunks compared in the matrix, will be guessed from matrix dimension if not supplied.

    title : string, optional
        Title string for the plot.


    Returns:
    --------
    fig : figure
        The figure containing the plots.
    """
    if matrix.ndim > 2:
        N = matrix.shape[0]
        n_chunks = matrix.shape[1]
    else:
        N = 1
        n_chunks = matrix.shape[0]

    if 'labels' in kwargs:
        labels = kwargs['labels']
        if len(labels) != n_chunks:
            raise ValueError('there must be a label for each dataset')
    else:
        if n_chunks == 11:
            labels = labels_50y
        elif n_chunks == 26:
            labels = labels_20y

    if 'figsize' in kwargs:
        fig_size = kwargs['figsize']
    else:
        fig_size = (6, 5 * N)

    if 'std_fontsize' in kwargs:
        std_fontsize = kwargs['std_fontsize']
    else:
        std_fontsize = 12
    
    """
    if 'math_fontsize' in kwargs:
        math_fontsize = kwargs['math_fontsize']
    else:
        math_fontsize = 16
    """
    
    if 'fig_id' in kwargs:
        fig = pl.figure(kwargs['fig_id'], figsize=fig_size)
    else:
        fig = pl.figure(figsize=fig_size)

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = CMAP

    if not bare:
        grid = ImageGrid(fig, 111, nrows_ncols=(N, 1), axes_pad=0.5, cbar_location='right',
                         cbar_mode='each', cbar_size='7%', cbar_pad='2%')
    else:
        grid = ImageGrid(fig, 111, nrows_ncols=(N, 1), axes_pad=0.5, cbar_mode='none')
                         
    for i in range(N):
        if N > 1:
            M = matrix[i, :, :]
        else:
            M = matrix

        if mask_diagonal:
            # mask the diagonal
            M = np.ma.MaskedArray(M, mask=np.eye(M.shape[0], dtype=bool))

        # support for binning if requested
        if matrix.dtype == 'bool':
            nbins = 2
        elif 'nbins' in kwargs:
            nbins = kwargs['nbins']
        else:
            nbins = 25

        if 'colorbar_range' in kwargs and kwargs['colorbar_range'] is not None:
            cbar_min = kwargs['colorbar_range'][0]
            cbar_max = kwargs['colorbar_range'][1]
        else:
            cbar_min = M.min()
            cbar_max = M.max()

        levels = MaxNLocator(nbins=nbins).tick_values(cbar_min, cbar_max)
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # use rasterized = True if white lines appear in pdfs, or to reduce file size
        hm = grid[i].pcolormesh(M, cmap=cmap, norm=norm, linewidth=0, rasterized=False)
        hm.set_edgecolor('face')

        # include colorbar
        ticks = MaxNLocator(nbins=6).tick_values(cbar_min, cbar_max)
        
        if not bare:
            grid.cbar_axes[i].colorbar(hm, format='%.2f', ticks=ticks)
    
            # # if colorbar range is set, show yellow bar indicating the values that do occur
            # if 'colorbar_range' in kwargs and kwargs['colorbar_range'] is not None:
            #    grid.cbar_axes[i].plot((1.2, 1.2), (M.min(), M.max()),
            #        color=RANGEBAR_COLOR, linewidth=7, zorder=9)
    
            for cax in grid.cbar_axes:
                cax.toggle_label(True)
                cax.tick_params(labelsize=std_fontsize) 

        if 'annot' in kwargs and kwargs['annot']:
            # if all below 1 assume kappa and scale to percent
            if M.max() <= 1:
                M_percent = M * 100
            else:
                M_percent = M

            for y in range(M.shape[0]):
                for x in range(M.shape[1]):
                    if x == y:
                        pass
                    else:                        
                        grid[i].text(x + 0.5, y + 0.5, '%d' % M_percent[y, x],
                                horizontalalignment='center',
                                verticalalignment='center')

        # put the major ticks at the middle of each cell
        grid[i].set_xticks(np.arange(M.shape[0]) + 0.5, minor=False)
        grid[i].set_yticks(np.arange(M.shape[1]) + 0.5, minor=False)

        # table-like display
        grid[i].xaxis.tick_bottom()
        if i == N - 1:
            if 'shortlabels' in kwargs and kwargs['shortlabels']:
                grid[i].set_xticklabels(labels, minor=False)
            else:
                grid[i].set_xticklabels(labels, minor=False, rotation=90)
        else:
            grid[i].set_xticklabels(labels, minor=False, visible=False)
        grid[i].set_yticklabels(labels, minor=False)
        grid[i].tick_params(axis='both', which='both', bottom=False, 
            top=False, left=False, right=False)
        
        if bare:
            grid[i].tick_params(axis='both', which='both', labelbottom=False, 
                labelleft=False, labelright=False, labeltop=False)

        grid[i].set_xlim([0, len(labels)])
        grid[i].set_ylim([0, len(labels)])

        grid[i].tick_params(axis='both', which='major', labelsize=std_fontsize)
        grid[i].tick_params(axis='both', which='minor', labelsize=std_fontsize)

        """
        if i == 0:
            grid[i].set_title(r'a) $\kappa$', fontsize=math_fontsize)
        elif i == 1:
            grid[i].set_title(r'b) $\kappa_{loc}$', fontsize=math_fontsize)
        elif i == 2:
            grid[i].set_title(r'c) $\kappa_{his}$', fontsize=math_fontsize)
        """

        if N > 1:
            if i == 0:
                grid[i].set_title('a)')
            elif i == 1:
                grid[i].set_title('b)')
            elif i == 2:
                grid[i].set_title('c)')

    if 'text' in kwargs:
        txt = '\n'.join(wrap(kwargs['text'], 55))
        fig.text(.07, .02, txt, fontsize=std_fontsize, ha='left', va='top')

    if display:
        if 'text' not in kwargs:
            pl.tight_layout()
        pl.show()

    return fig