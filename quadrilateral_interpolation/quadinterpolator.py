"""
# Quadrilateral Interpolation

This code provides a fast quadrilateral interpolation to perform satellite image projection from swath geometry to any
geographical projection.

The method and the code implementation are intent to be very fast but may not be as accurate as other backward interpolations (bilinear, ...). 
It is suitable for coarse resolution imagery. The calculation is based on:
[Interpolation using an arbitrary quadrilateral](https://www.particleincell.com/2012/quad-interpolation/)

The API follows the conventions in [xESMF](https://github.com/pangeo-data/xESMF) for the Resampler object and 
in [pyresample](https://github.com/pytroll/pyresample) for the grid definition. See pyresample for useful tools
 to create and manipulate grids.

This package was designed to work for satellite images opened with [satpy](https://github.com/pytroll/satpy) 
but can be used as a standalone projection tool as long as the basic grid definition information are provided.

"""


import math
from warnings import warn

import numba
import numpy as np
import xarray as xr
from pyproj import Proj


class QuadInterpolationResampler:
    """create a Resampler object from source and target grid definitions.
    the target grid must provide the attributes crs, pixel_upper_left, pixel_size_x, and pixel_size_y. 
    The function `load_area` in the pyresample package provides an easy way to build this object.
    The source_def has lons and lats attributes, 
    it can be build from [pyresample.geometry.SwathDefinition](https://pyresample.readthedocs.io/en/latest/geo_def.html).
"""

    def __init__(self, source_def, target_def):
        """initialize a resampler oject.
        :param source_def:
        :param target_def:
"""

        self.source_def = source_def
        self.target_def = target_def

    def resample(self, data):
        """resample the data"""

        # initialize the pyproj projection
        p = Proj(self.target_def.crs)

        # projection of the lon, lat to the X, Y
        X, Y = p(self.source_def.lons.values.flatten(), self.source_def.lats.values.flatten())

        ny, nx = self.source_def.lons.values.shape
        X = X.reshape((ny, nx))
        Y = Y.reshape((ny, nx))

        if hasattr(self.source_def.lons, "dims"):
            # check that the two last dims of data
            lonlatdims = self.source_def.lons.dims
            otherdims = [d for d in data.dims if d not in lonlatdims]
            data = data.transpose(*(otherdims + list(lonlatdims)))
        else:
            warn("# ... hope the order is correct!")
            pass

        def resample(data):
            if len(data.values.shape) < 2:
                return data
            regridded = regrid_quadinterpolation(data.values, X, Y,
                                                 self.target_def.pixel_upper_left,
                                                 (self.target_def.pixel_size_x, self.target_def.pixel_size_y),
                                                 (self.target_def.width, self.target_def.height))

            x = self.target_def.pixel_upper_left[0] + np.arange(self.target_def.width) * self.target_def.pixel_size_x
            y = self.target_def.pixel_upper_left[1] - np.arange(self.target_def.height) * self.target_def.pixel_size_y

            coords = dict(data.coords)
            coords.update({'x': x, 'y': y})

            regridded = xr.DataArray(regridded, coords, dims=data.dims[:-2] + ('y', 'x'))
            return regridded

        if isinstance(data, xr.Dataset):

            def all_same_size(items):
                return all(x == items[0] for x in items if len(x) >= 2)

            if all_same_size([data[v].dims for v in data.variables]):
                warn("It is significantly faster to interpolate a rank 3 (or higher) DataArray (e.g. with coords=(band, lat, lon))"
                     "than multiple Dataset variables of rank 2 (e.g. coords=(lat, lon))."
                     " It is recommended to reshape the input DataSet into a single DataArray."
                     " You can safely ignore this warning if performance is not an issue.")

            return data.map(resample, keep_attrs=True)
        else:
            return resample(data)


@numba.jit(nopython=True, cache=True)
def alpha_beta(p00, p01, p10, p11):
    # quad interpolation: https://www.particleincell.com/2012/quad-interpolation/

    x = [p00[0], p01[0], p11[0], p10[0]]
    y = [p00[1], p01[1], p11[1], p10[1]]

    alpha = [x[0],
             - x[0] + x[1],
             - x[0] + x[3],
             x[0] - x[1] + x[2] - x[3]]

    beta = [y[0],
            - y[0] + y[1],
            - y[0] + y[3],
            y[0] - y[1] + y[2] - y[3]]

    return alpha, beta


@numba.jit(nopython=True, cache=True)
def XtoL(x, y, alpha, beta):
    # quad interpolation: https://www.particleincell.com/2012/quad-interpolation/

    # quadratic equation coeffs, aa*mm^2+bb*m+cc=0
    aa = alpha[3] * beta[2] - alpha[2] * beta[3]
    bb = alpha[3] * beta[0] - alpha[0] * beta[3] + alpha[1] * beta[2] - alpha[2] * beta[1] + x * beta[3] - y * alpha[3]
    cc = alpha[1] * beta[0] - alpha[0] * beta[1] + x * beta[1] - y * alpha[1]

    if aa == 0:
        if bb == 0:
            return -1, -1  # outside
        m = -cc / bb
    else:
        # compute m = (-b+sqrt(b^2-4ac))/(2a)
        discr = bb * bb - 4 * aa * cc
        if discr < 0:
            return -1, -1  # same as outside...
        det = math.sqrt(discr)
        m = (-bb + det) / (2 * aa)

    # compute l
    denom = (alpha[1] + alpha[3] * m)
    if denom != 0:
        l = (x - alpha[0] - alpha[2] * m) / denom
    else:
        denom = (beta[1] + beta[3] * m)
        if denom == 0:
            return -1, -1
        l = (y - beta[0] - beta[2] * m) / denom

    return l, m


def regrid_quadinterpolation(data, *args, **kwargs):

    if len(data.shape) == 2:
        return regrid_quadinterpolation_compiled_rank2(data, *args, **kwargs)
    else:
        return regrid_quadinterpolation_compiled_higherrank(data, *args, **kwargs)


@numba.jit(nopython=True, cache=True)
def regrid_quadinterpolation_compiled_rank2(data, X, Y, center_pixel_upper_left, pixel_size, npixel):

    regridded = np.zeros(data.shape[:-2] + (npixel[1], npixel[0]))
    count = np.zeros((npixel[1], npixel[0]), dtype=np.int8)

    ny, nx = X.shape

    def to_grid(x, y):
        ix = (x - center_pixel_upper_left[0]) / pixel_size[0]
        iy = (y - center_pixel_upper_left[1]) / -pixel_size[1]
        return ix, iy

    for i in range(0, ny - 1):
        for j in range(0, nx - 1):

            if not np.isfinite(data[i, j]):
                continue

            p00 = to_grid(X[i, j], Y[i, j])

            if (p00[0] < 0) or (p00[0] >= npixel[0]) or (p00[1] < 0) or (p00[1] >= npixel[1]):
                continue

            p01 = to_grid(X[i, j + 1], Y[i, j + 1])
            p10 = to_grid(X[i + 1, j], Y[i + 1, j])
            p11 = to_grid(X[i + 1, j + 1], Y[i + 1, j + 1])

            ix0 = math.floor(min(p00[0], p01[0], p10[0], p11[0]))
            ix1 = math.ceil(max(p00[0], p01[0], p10[0], p11[0]))

            iy0 = math.floor(min(p00[1], p01[1], p10[1], p11[1]))
            iy1 = math.ceil(max(p00[1], p01[1], p10[1], p11[1]))

            if (abs(ix0 - ix1) > 6) or (abs(iy0 - iy1) > 6):
                # it's odd... one of the points is too far. This should not happen if the destination grid
                # has a similar resolution to the native resolution
                continue

            alpha, beta = alpha_beta(p00, p01, p10, p11)

            for iy in range(max(iy0, 0), min(iy1 + 1, npixel[1])):
                for ix in range(max(ix0, 0), min(ix1 + 1, npixel[0])):

                    # transform the x,y coordinates into l, m
                    l, m = XtoL(ix, iy, alpha, beta)
                    if (m < 0) or (m > 1) or (l < 0) or (l > 1):
                        # outside!
                        continue
                    # perform the interpolation
                    regridded[..., iy, ix] += (1 - l) * ((1 - m) * data[i, j] + m * data[i + 1, j]) + \
                        l * ((1 - m) * data[i, j + 1] + m * data[i + 1, j + 1])
                    count[iy, ix] += 1

    return regridded / count


@numba.jit(nopython=True, cache=True)
def regrid_quadinterpolation_compiled_higherrank(data, X, Y, center_pixel_upper_left, pixel_size, npixel):

    regridded = np.zeros(data.shape[:-2] + (npixel[1], npixel[0]))
    count = np.zeros((npixel[1], npixel[0]), dtype=np.int8)

    ny, nx = X.shape

    def to_grid(x, y):
        ix = (x - center_pixel_upper_left[0]) / pixel_size[0]
        iy = (y - center_pixel_upper_left[1]) / -pixel_size[1]
        return ix, iy

    for i in range(0, ny - 1):
        for j in range(0, nx - 1):

            if not np.isfinite(data[..., i, j]).any():
                continue

            p00 = to_grid(X[i, j], Y[i, j])

            if (p00[0] < 0) or (p00[0] >= npixel[0]) or (p00[1] < 0) or (p00[1] >= npixel[1]):
                continue

            p01 = to_grid(X[i, j + 1], Y[i, j + 1])
            p10 = to_grid(X[i + 1, j], Y[i + 1, j])
            p11 = to_grid(X[i + 1, j + 1], Y[i + 1, j + 1])

            ix0 = math.floor(min(p00[0], p01[0], p10[0], p11[0]))
            ix1 = math.ceil(max(p00[0], p01[0], p10[0], p11[0]))

            iy0 = math.floor(min(p00[1], p01[1], p10[1], p11[1]))
            iy1 = math.ceil(max(p00[1], p01[1], p10[1], p11[1]))

            if (abs(ix0 - ix1) > 6) or (abs(iy0 - iy1) > 6):
                # it's odd... one of the points is too far. This should not happen if the destination grid
                # has a similar resolution to the native resolution
                continue

            alpha, beta = alpha_beta(p00, p01, p10, p11)

            for iy in range(max(iy0, 0), min(iy1 + 1, npixel[1])):
                for ix in range(max(ix0, 0), min(ix1 + 1, npixel[0])):

                    # transform the x,y coordinates into l, m
                    l, m = XtoL(ix, iy, alpha, beta)
                    if (m < 0) or (m > 1) or (l < 0) or (l > 1):
                        # outside!
                        continue
                    # perform the interpolation
                    regridded[..., iy, ix] += (1 - l) * ((1 - m) * data[..., i, j] + m * data[..., i + 1, j]) + \
                        l * ((1 - m) * data[..., i, j + 1] + m * data[..., i + 1, j + 1])
                    count[iy, ix] += 1

    return regridded / count
