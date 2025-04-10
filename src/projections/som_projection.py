import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os

class SOMProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

    def __repr__(self):
        return f'<SOMProjectionResult 1-d-embedding={self.embedding}>'

class SOMProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

    def __repr__(self):
        return f'<SOMProjectionResult 1-d-embedding={self.embedding}>'
    

# def som_sphere_projection():
#     """
#     .. module:: SOMZ
#     .. moduleauthor:: Matias Carrasco Kind

#     """
#     from __future__ import print_function
#     from builtins import zip
#     from builtins import range
#     from builtins import object
#     __author__ = 'Matias Carrasco Kind'
#     import numpy
#     import copy
#     import sys, os, random
#     import warnings

#     warnings.simplefilter("ignore", RuntimeWarning)
#     try:
#         import somF
#         SF90 = True
#     except:
#         SF90 = False


# def get_index(ix, iy, nx, ny):
#     return iy * nx + ix


# def get_pair(ii, nx, ny):
#     iy = int(np.floor(ii / nx))
#     ix = ii % nx
#     return ix, iy


# def get_ns(ix, iy, nx, ny, index=False):
#     """
#     Get neighbors for rectangular grid given its
#     coordinates and size of grid

#     :param int ix: Coordinate in the x-axis
#     :param int iy: Coordinate in the y-axis
#     :param int nx: Number fo cells along the x-axis
#     :param int ny: Number fo cells along the y-axis
#     :param bool index: Return indexes in the map format
#     :return: Array of indexes for direct neighbors
#     """
#     ns = []
#     if ix - 1 >= 0: ns.append((ix - 1, iy))
#     if iy - 1 >= 0: ns.append((ix, iy - 1))
#     if ix + 1 < nx: ns.append((ix + 1, iy))
#     if iy + 1 < ny: ns.append((ix, iy + 1))

#     if ix - 1 >= 0 and iy - 1 >= 0: ns.append((ix - 1, iy - 1))
#     if ix - 1 >= 0 and iy + 1 < ny: ns.append((ix - 1, iy + 1))
#     if ix + 1 < nx and iy + 1 < ny: ns.append((ix + 1, iy + 1))
#     if ix + 1 < nx and iy - 1 >= 0: ns.append((ix + 1, iy - 1))

#     ns = np.array(ns)
#     if not index:
#         return ns
#     if index:
#         ins = []
#         for i in range(len(ns)):
#             ins.append(get_index(ns[i, 0], ns[i, 1], nx, ny))
#         return np.array(ins)


# def get_ns_hex(ix, iy, nx, ny, index=False):
#     """
#     Get neighbors for hexagonal grid given its coordinates
#     and size of grid
#     Same parameters as :func:`get_ns`
#     """
#     ns = []
#     even = False
#     if iy % 2 == 0: even = True
#     if ix - 1 >= 0: ns.append((ix - 1, iy))
#     if ix + 1 < nx: ns.append((ix + 1, iy))
#     if iy - 1 >= 0: ns.append((ix, iy - 1))
#     if iy + 1 < ny: ns.append((ix, iy + 1))
#     if even and ix - 1 >= 0 and iy - 1 >= 0: ns.append((ix - 1, iy - 1))
#     if even and ix - 1 >= 0 and iy + 1 < ny: ns.append((ix - 1, iy + 1))
#     if not even and ix + 1 < nx and iy - 1 >= 0: ns.append((ix + 1, iy - 1))
#     if not even and ix + 1 < nx and iy + 1 < ny: ns.append((ix + 1, iy + 1))
#     ns = np.array(ns)
#     if not index:
#         return ns
#     if index:
#         ins = []
#         for i in range(len(ns)):
#             ins.append(get_index(ns[i, 0], ns[i, 1], nx, ny))
#         return np.array(ins)


# def geometry(top, Ntop, periodic='no'):
#     """
#     Pre-compute distances between cells in a given topology
#     and store it on a distLib array

#     :param str top: Topology ('grid','hex','sphere')
#     :param int Ntop: Size of map,  for grid Size=Ntop*Ntop,
#         for hex Size=Ntop*(Ntop+1[2]) if Ntop is even[odd] and for sphere
#         Size=12*Ntop*Ntop and top must be power of 2
#     :param str periodic: Use periodic boundary conditions ('yes'/'no'), valid for 'hex' and 'grid' only
#     :return: 2D array with distances pre computed between cells and total number of units
#     :rtype: 2D float array, int
#     """
#     if top == 'sphere':
#         try:
#             import healpy as hpx
#         except:
#             print('Error: healpy module not found, use grid or hex topologies')
#             sys.exit(0)
#     if top == 'sphere':
#         nside = Ntop
#         npix = 12 * nside ** 2
#         distLib = np.zeros((npix, npix))
#         for i in range(npix):
#             ai = hpx.pix2ang(nside, i)
#             for j in range(i + 1, npix):
#                 aj = hpx.pix2ang(nside, j)
#                 distLib[i, j] = hpx.rotator.angdist(ai, aj)
#                 distLib[j, i] = distLib[i, j]
#         distLib[np.where(np.isnan(distLib))] = np.pi
#     if top == 'grid':
#         nx = Ntop
#         ny = Ntop
#         npix = nx * ny
#         mapxy = np.mgrid[0:1:complex(0, nx), 0:1:complex(0, ny)]
#         mapxy = np.reshape(mapxy, (2, npix))
#         bX = mapxy[1]
#         bY = mapxy[0]
#         dx = 1. / (nx - 1)
#         dy = 1. / (ny - 1)
#         distLib = np.zeros((npix, npix))
#         if periodic == 'no':
#             for i in range(npix):
#                 for j in range(i + 1, npix):
#                     distLib[i, j] = np.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
#                     distLib[j, i] = distLib[i, j]
#         if periodic == 'yes':
#             for i in range(npix):
#                 for j in range(i + 1, npix):
#                     s0 = np.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
#                     s1 = np.sqrt((bX[i] - (bX[j] + 1. + dx)) ** 2 + (bY[i] - bY[j]) ** 2)
#                     s2 = np.sqrt((bX[i] - (bX[j] + 1. + dx)) ** 2 + (bY[i] - (bY[j] + 1. + dy)) ** 2)
#                     s3 = np.sqrt((bX[i] - (bX[j] + 0.)) ** 2 + (bY[i] - (bY[j] + 1. + dy)) ** 2)
#                     s4 = np.sqrt((bX[i] - (bX[j] - 1. - dx)) ** 2 + (bY[i] - (bY[j] + 1. + dy)) ** 2)
#                     s5 = np.sqrt((bX[i] - (bX[j] - 1. - dx)) ** 2 + (bY[i] - (bY[j] + 0.)) ** 2)
#                     s6 = np.sqrt((bX[i] - (bX[j] - 1. - dx)) ** 2 + (bY[i] - (bY[j] - 1. - dy)) ** 2)
#                     s7 = np.sqrt((bX[i] - (bX[j] + 0.)) ** 2 + (bY[i] - (bY[j] - 1. - dy)) ** 2)
#                     s8 = np.sqrt((bX[i] - (bX[j] + 1. + dx)) ** 2 + (bY[i] - (bY[j] - 1. - dy)) ** 2)
#                     distLib[i, j] = np.min((s0, s1, s2, s3, s4, s5, s6, s7, s8))
#                     distLib[j, i] = distLib[i, j]
#     if top == 'hex':
#         nx = Ntop
#         ny = Ntop
#         xL = np.arange(0, nx, 1.)
#         dy = 0.8660254
#         yL = np.arange(0, ny, dy)
#         ny = len(yL)
#         nx = len(xL)
#         npix = nx * ny
#         bX = np.zeros(nx * ny)
#         bY = np.zeros(nx * ny)
#         kk = 0
#         last = ny * dy
#         for jj in range(ny):
#             for ii in range(nx):
#                 if jj % 2 == 0: off = 0.
#                 if jj % 2 == 1: off = 0.5
#                 bX[kk] = xL[ii] + off
#                 bY[kk] = yL[jj]
#                 kk += 1
#         distLib = np.zeros((npix, npix))
#         if periodic == 'no':
#             for i in range(npix):
#                 for j in range(i + 1, npix):
#                     distLib[i, j] = np.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
#                     distLib[j, i] = distLib[i, j]
#         if periodic == 'yes':
#             for i in range(npix):
#                 for j in range(i + 1, npix):
#                     s0 = np.sqrt((bX[i] - bX[j]) ** 2 + (bY[i] - bY[j]) ** 2)
#                     s1 = np.sqrt((bX[i] - (bX[j] + nx)) ** 2 + (bY[i] - bY[j]) ** 2)
#                     s2 = np.sqrt((bX[i] - (bX[j] + nx)) ** 2 + (bY[i] - (bY[j] + last)) ** 2)
#                     s3 = np.sqrt((bX[i] - (bX[j] + 0)) ** 2 + (bY[i] - (bY[j] + last)) ** 2)
#                     s4 = np.sqrt((bX[i] - (bX[j] - nx)) ** 2 + (bY[i] - (bY[j] + last)) ** 2)
#                     s5 = np.sqrt((bX[i] - (bX[j] - nx)) ** 2 + (bY[i] - (bY[j] + 0)) ** 2)
#                     s6 = np.sqrt((bX[i] - (bX[j] - nx)) ** 2 + (bY[i] - (bY[j] - last)) ** 2)
#                     s7 = np.sqrt((bX[i] - (bX[j] + 0)) ** 2 + (bY[i] - (bY[j] - last)) ** 2)
#                     s8 = np.sqrt((bX[i] - (bX[j] + nx)) ** 2 + (bY[i] - (bY[j] - last)) ** 2)
#                     distLib[i, j] = np.min((s0, s1, s2, s3, s4, s5, s6, s7, s8))
#                     distLib[j, i] = distLib[i, j]
#     return distLib, npix


# def is_power_2(value):
#     """
#     Check if passed value is a power of 2
#     """
#     return value!=0 and ((value & (value- 1)) == 0)


# def get_alpha(t, alphas, alphae, NT):
#     """
#     Get value of alpha at a given time
#     """
#     return alphas * np.power(alphae / alphas, float(t) / float(NT))


# def get_sigma(t, sigma0, sigmaf, NT):
#     """
#     Get value of sigma at a given time
#     """
#     return sigma0 * np.power(sigmaf / sigma0, float(t) / float(NT))


# def h(bmu, mapD, sigma):
#     """
#     Neighborhood function which quantifies how much cells around the best matching one are modified

#     :param int bmu: best matching unit
#     :param float mapD: array of distances computed with :func:`geometry`
#     """
#     return np.exp(-(mapD[bmu] ** 2) / sigma ** 2)


# class SelfMap(object):
#     """
#     Create a som class instance

#     :param float X: Attributes array (all columns used)
#     :param float Y: Attribute to be predicted (not really needed, can be zeros)
#     :param str topology: Which 2D topology, 'grid', 'hex' or 'sphere'
#     :param str som_type: Which updating scheme to use 'online' or 'batch'
#     :param int Ntop: Size of map,  for grid Size=Ntop*Ntop,
#         for hex Size=Ntop*(Ntop+1[2]) if Ntop is even[odd] and for sphere
#         Size=12*Ntop*Ntop and top must be power of 2
#     :param  int iterations: Number of iteration the entire sample is processed
#     :param str periodic: Use periodic boundary conditions ('yes'/'no'), valid for 'hex' and 'grid' only
#     :param dict dict_dim: dictionary with attributes names
#     :param float astar: Initial value of alpha
#     :param float aend: End value of alpha
#     :param str importance: Path to the file with importance ranking for attributes, default is none
#     """

#     def __init__(self, X, Y, topology='grid', som_type='online', Ntop=28, iterations=30, periodic='no', dict_dim='',
#                  astart=0.8, aend=0.5, importance=None):
#         self.np, self.nDim = np.shape(X)
#         self.dict_dim = dict_dim
#         self.X = X
#         self.SF90 = SF90
#         self.Y = Y
#         self.aps = astart
#         self.ape = aend
#         self.top = topology
#         if topology=='sphere' and not is_power_2(Ntop):
#             print('Error, Ntop must be power of 2')
#             sys.exit(0)
#         self.stype = som_type
#         self.Ntop = Ntop
#         self.nIter = iterations
#         self.per = periodic
#         self.distLib, self.npix = geometry(self.top, self.Ntop, periodic=self.per)
#         if importance == None: importance = np.ones(self.nDim)
#         self.importance = importance / np.sum(importance)

#     def som_best_cell(self, inputs, return_vals=1):
#         """
#         Return the closest cell to the input object
#         It can return more than one value if needed
#         """
#         activations = np.sum(np.transpose([self.importance]) * (
#             np.transpose(np.tile(inputs, (self.npix, 1))) - self.weights) ** 2, axis=0)
#         if return_vals == 1:
#             best = np.argmin(activations)
#             return best, activations
#         else:
#             best_few = np.argsort(activations)
#             return best_few[0:return_vals], activations

#     def create_mapF(self, evol='no', inputs_weights=''):
#         """
#         This functions actually create the maps, it uses
#         random values to initialize the weights
#         It uses a Fortran subroutine compiled with f2py
#         """
#         if not self.SF90:
#             print()
#             print('Fortran module somF not found, use create_map instead or try' \
#                   ' f2py -c -m somF som.f90')
#             sys.exit(0)
#         if inputs_weights == '':
#             self.weights = (np.random.rand(self.nDim, self.npix)) + self.X[0][0]
#         else:
#             self.weights = inputs_weights
#         if self.stype == 'online':
#             self.weightsT = somF.map(self.X, self.nDim, self.nIter, self.distLib, self.np, self.weights,
#                                      self.importance, self.npix, self.aps, self.ape)
#         if self.stype == 'batch':
#             self.weightsT = somF.map_b(self.X, self.nDim, self.nIter, self.distLib, self.np, self.weights,
#                                        self.importance, self.npix)
#         self.weights = copy.deepcopy(self.weightsT)

#     def create_map(self, evol='no', inputs_weights='', random_order=True):
#         """
#         This is same as above but uses python routines instead
#         """
#         if inputs_weights == '':
#             self.weights = (np.random.rand(self.nDim, self.npix)) + self.X[0][0]
#         else:
#             self.weights = inputs_weights
#         self.NT = self.nIter * self.np
#         if self.stype == 'online':
#             tt = 0
#             sigma0 = self.distLib.max()
#             sigma_single = np.min(self.distLib[np.where(self.distLib > 0.)])
#             for it in range(self.nIter):
#                 #get alpha, sigma
#                 alpha = get_alpha(tt, self.aps, self.ape, self.NT)
#                 sigma = get_sigma(tt, sigma0, sigma_single, self.NT)
#                 if random_order:
#                     index_random = random.sample(range(self.np), self.np)
#                 else:
#                     index_random = np.arange(self.np)
#                 for i in range(self.np):
#                     tt += 1
#                     inputs = self.X[index_random[i]]
#                     best, activation = self.som_best_cell(inputs)
#                     self.weights += alpha * h(best, self.distLib, sigma) * numpy.transpose(
#                         (inputs - numpy.transpose(self.weights)))
#                 if evol == 'yes':
#                     self.evaluate_map()
#                     self.save_map(itn=it)
#         if self.stype == 'batch':
#             tt = 0
#             sigma0 = self.distLib.max()
#             sigma_single = np.min(self.distLib[np.where(self.distLib > 0.)])
#             for it in range(self.nIter):
#                 #get alpha, sigma
#                 sigma = get_sigma(tt, sigma0, sigma_single, self.NT)
#                 accum_w = np.zeros((self.nDim, self.npix))
#                 accum_n = np.zeros(self.npix)
#                 for i in range(self.np):
#                     tt += 1
#                     inputs = self.X[i]
#                     best, activation = self.som_best_cell(inputs)
#                     for kk in range(self.nDim):
#                         accum_w[kk, :] += h(best, self.distLib, sigma) * inputs[kk]
#                     accum_n += h(best, self.distLib, sigma)
#                 for kk in range(self.nDim):
#                     self.weights[kk] = accum_w[kk] / accum_n

#                 if evol == 'yes':
#                     self.evaluate_map()
#                     self.save_map(itn=it)

#     def evaluate_map(self, inputX='', inputY=''):
#         """
#         This functions evaluates the map created using the input Y or a new Y (array of labeled attributes)
#         It uses the X array passed or new data X as well, the map doesn't change

#         :param float inputX: Use this if another set of values for X is wanted using
#             the weigths already computed
#         :param float inputY: One  dimensional array of the values to be assigned to each cell in the map
#             based on the in-memory X passed
#         """
#         self.yvals = {}
#         self.ivals = {}
#         if inputX == '':
#             inX = self.X
#         else:
#             inX = inputX
#         if inputY == '':
#             inY = self.Y
#         else:
#             inY = inputY
#         for i in range(len(inX)):
#             inputs = inX[i]
#             best, activation = self.som_best_cell(inputs)
#             if best not in self.yvals: self.yvals[best] = []
#             self.yvals[best].append(inY[i])
#             if best not in self.ivals: self.ivals[best] = []
#             self.ivals[best].append(i)

#     def get_vals(self, line):
#         """
#         Get the predictions  given a line search, where the line
#         is a vector of attributes per individual object fot the
#         10 closest cells.

#         :param float line: input data to look in the tree
#         :return: array with the cell content
#         """
#         best, act = self.som_best_cell(line, return_vals=10)
#         for ib in range(10):
#             if best[ib] in self.yvals: return self.yvals[best[ib]]
#         return np.array([-1.])

#     def get_best(self, line):
#         """
#         Get the predictions  given a line search, where the line
#         is a vector of attributes per individual object for THE best cell

#         :param float line: input data to look in the tree
#         :return: array with the cell content
#         """
#         best, act = self.som_best_cell(line, return_vals=10)
#         return best[0]

#     def save_map(self, itn=-1, fileout='SOM', path=''):
#         """
#         Saves the map

#         :param int itn: Number of map to be included on path, use -1 to ignore this number
#         :param str fileout: Name of output file
#         :param str path: path for the output file
#         """
#         if path == '':
#             path = os.getcwd() + '/'
#         if not os.path.exists(path): os.system('mkdir -p ' + path)
#         if itn >= 0:
#             ff = '_%04d' % itn
#             fileout += ff
#         np.save(path + fileout, self)

#     def save_map_dict(self, path='', fileout='SOM', itn=-1):
#         """
#         Saves the map in dictionary format

#         :param int itn: Number of map to be included on path, use -1 to ignore this number
#         :param str fileout: Name of output file
#         :param str path: path for the output file
#         """
#         SOM = {}
#         SOM['W'] = self.weights
#         SOM['yvals'] = self.yvals
#         SOM['ivals'] = self.ivals
#         SOM['topology'] = self.top
#         SOM['Ntop'] = self.Ntop
#         SOM['npix'] = self.npix
#         if path == '':
#             path = os.getcwd() + '/'
#         if not os.path.exists(path): os.system('mkdir -p ' + path)
#         if itn > 0:
#             ff = '_%04d' % itn
#             fileout += ff
#         np.save(path + fileout, SOM)

#     def plot_map(self, min_m=-100, max_m=-100, colbar='yes'):
#         """
#         Plots the map after evaluating, the cells are colored with the mean value inside each
#         one of them

#         :param float min_m: Lower limit for coloring the cells, -100 uses min value
#         :param float max_m: Upper limit for coloring the cells, -100 uses max value
#         :param str colbar: Include a colorbar ('yes','no')
#         """

#         import matplotlib.pyplot as plt
#         import matplotlib as mpl
#         import matplotlib.cm as cm
#         from matplotlib import collections, transforms
#         from matplotlib.colors import colorConverter
#         import healpy

#         if self.top == 'sphere': import healpy as H

#         if self.top == 'grid':
#             M = np.zeros(self.npix) - 20.
#             for i in range(self.npix):
#                 if i in self.yvals:
#                     M[i] = np.mean(self.yvals[i])
#             M2 = np.reshape(M, (self.Ntop, self.Ntop))
#             plt.figure(figsize=(8, 8), dpi=100)
#             if min_m == -100: min_m = M2[np.where(M2 > -10)].min()
#             if max_m == -100: max_m = M2.max()
#             SM2 = plt.imshow(M2, origin='center', interpolation='nearest', cmap=cm.jet, vmin=min_m, vmax=max_m)
#             SM2.cmap.set_under("grey")
#             if colbar == 'yes': plt.colorbar()
#             plt.axis('off')
#         if self.top == 'hex':
#             nx = self.Ntop
#             ny = self.Ntop
#             xL = np.arange(0, nx, 1.)
#             dy = 0.8660254
#             yL = np.arange(0, ny, dy)
#             ny = len(yL)
#             nx = len(xL)
#             npix = nx * ny
#             bX = np.zeros(nx * ny)
#             bY = np.zeros(nx * ny)
#             kk = 0
#             for jj in range(ny):
#                 for ii in range(nx):
#                     if jj % 2 == 0: off = 0.
#                     if jj % 2 == 1: off = 0.5
#                     bX[kk] = xL[ii] + off
#                     bY[kk] = yL[jj]
#                     kk += 1
#             xyo = list(zip(bX, bY))
#             sizes_2 = np.zeros(nx * ny) + ((8. * 0.78 / (self.Ntop + 0.5)) / 2. * 72.) ** 2 * 4. * np.pi / 3.
#             M = np.zeros(npix) - 20.
#             fcolors = [plt.cm.Spectral_r(x) for x in np.random.rand(nx * ny)]
#             for i in range(npix):
#                 if i in self.yvals:
#                     M[i] = np.mean(self.yvals[i])
#             if max_m == -100: max_m = M.max()
#             if min_m == -100: min_m = M[np.where(M > -10)].min()
#             M = M - min_m
#             M = M / (max_m - min_m)
#             for i in range(npix):
#                 if M[i] <= 0:
#                     fcolors[i] = plt.cm.Greys(.5)
#                 else:
#                     fcolors[i] = plt.cm.jet(M[i])
#             figy = ((8. * 0.78 / (self.Ntop + 0.5) / 2.) * (3. * ny + 1) / np.sqrt(3)) / 0.78
#             fig3 = plt.figure(figsize=(8, figy), dpi=100)
#             #fig3.subplots_adjust(left=0,right=1.,top=1.,bottom=0.)
#             a = fig3.add_subplot(1, 1, 1)
#             col = collections.RegularPolyCollection(6, sizes=sizes_2, offsets=xyo, transOffset=a.transData)
#             col.set_color(fcolors)
#             a.add_collection(col, autolim=True)
#             a.set_xlim(-0.5, nx)
#             a.set_ylim(-1, nx + 0.5)
#             plt.axis('off')
#             if colbar == 'yes':
#                 figbar = plt.figure(figsize=(8, 1.), dpi=100)
#                 ax1 = figbar.add_axes([0.05, 0.8, 0.9, 0.15])
#                 cmap = cm.jet
#                 norm = mpl.colors.Normalize(vmin=min_m, vmax=max_m)
#                 cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')
#                 cb1.set_label('')
#         if self.top == 'sphere':
#             M = np.zeros(self.npix) + H.UNSEEN
#             for i in range(self.npix):
#                 if i in self.yvals:
#                     M[i] = np.mean(self.yvals[i])
#             plt.figure(10, figsize=(8, 8), dpi=100)
#             if min_m == -100: min_m = M[np.where(M > -10)].min()
#             if max_m == -100: max_m = M.max()
#             if colbar == 'yes': H.mollview(M, fig=10, title="", min=min_m, max=max_m, cbar=True)
#             if colbar == 'no': H.mollview(M, fig=10, title="", min=min_m, max=max_m, cbar=False)
#         plt.show()



def som_projection(points, lr=0.1, sigma=1.0, maxiter=10000, max_time=None, show_plots=True, labels=None):

    # Convert DataFrame to tensor
    points = torch.tensor(points.values, dtype=torch.float32)
    n, dim = points.shape

    # Initialize embedding
    embedding = torch.randn(n, 2)

    # Record start time for time constraint handling
    start_time = time.time()
    loss_records = []

    for i in range(maxiter):
        # Check the time constraint
        if max_time and (time.time() - start_time) > max_time:
            print("SOM training stopped due to time limit.")
            break

        # Randomly select a data point
        data_point = points[torch.randint(0, n, (1,)).item()]

        # Compute distances from data point to all nodes in the embedding
        distances = torch.sqrt(torch.sum((embedding - data_point[:2]) ** 2, dim=1))

        # Find the best matching unit (BMU) in the SOM embedding
        bmu_idx = torch.argmin(distances).item()

        # Update embedding nodes based on the BMU
        for j in range(n):
            # Distance between the BMU and current node
            node_dist = torch.tensor((j - bmu_idx) ** 2, dtype=torch.float32)  # Ensure node_dist is a tensor
            # Influence based on neighborhood function
            influence = torch.exp(-node_dist / (2 * sigma ** 2))
            # Update node position
            embedding[j] += lr * influence * (data_point[:2] - embedding[j])

        # Record the loss as the average distance to all embedding nodes
        avg_dist = distances.mean().item()
        loss_records.append(avg_dist)

        # Decay learning rate and neighborhood radius
        lr *= 0.995
        sigma *= 0.995

    # Circular mapping from SOM result
    theta = torch.atan2(embedding[:, 1], embedding[:, 0])
    circle_x = torch.cos(theta)
    circle_y = torch.sin(theta)

    # Set up a color palette similar to the one used for cPro
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors = [palette[label] for label in labels] if labels is not None else 'blue'

    if show_plots:
        # Plot the circular layout with cluster colors
        plt.scatter(circle_x.numpy(), circle_y.numpy(), c=colors, edgecolor='white', s=40)
        plt.title('SOM Projection - Circular Mapping')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    # Return SOM projection result
    return SOMProjectionResult(
        embedding=embedding,
        circle_x=circle_x.numpy(),
        circle_y=circle_y.numpy(),
        loss_records=loss_records,
        hd_dist_matrix=None,
        ld_dist_matrix=None
    )

# Example usage
# points = pd.DataFrame(np.random.rand(100, 5))
# result = som_projection(points, max_time=10)  # Limit the SOM process to 10 seconds
# print(result.circle_x, result.circle_y)
